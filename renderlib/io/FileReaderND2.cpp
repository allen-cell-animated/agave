#include "FileReaderND2.h"

#include "BoundingBox.h"
#include "ImageXYZC.h"
#include "Logging.h"
#include "VolumeDimensions.h"

#include <lz4frame.h>
#include <zlib.h>
#include <curl/curl.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <variant>
#include <vector>

// =========================================================================
// ND2 v3 (post-2017) native reader.
//
// Binary format reference: tlambert03/nd2 (BSD-3) — _parse/_chunk_decode.py
// for the chunkmap layout, _parse/_clx_lite.py for the metadata variant
// encoding, and the runtime layer for experiment-loop traversal. See also
// renderlib/io/nd2_reader.py in this repo for cross-reference of the
// semantic decisions (channel-interleaving detection, frame-index formula).
//
// File layout (v3):
//   [0]                               : start-file chunk (16-byte header
//                                       + 32-byte name + 64-byte data;
//                                       data starts with "Ver3.0").
//   ...                               : data chunks (each: 16-byte header
//                                       + name + payload).
//   [last 40 bytes]                   : 32-byte chunkmap-signature
//                                       "ND2 CHUNK MAP SIGNATURE 0000001!"
//                                       + uint64 LE offset of the
//                                       chunkmap chunk.
//
// Each chunk:
//   uint32 magic    = 0x0ABECEDA
//   uint32 nameLen  (bytes of name, includes trailing '!')
//   uint64 dataLen  (bytes of payload)
//   bytes  name     (ASCII, length nameLen)
//   bytes  data     (length dataLen)
//
// Chunkmap chunk payload:
//   Repeating: <name bytes ending in '!'><uint64 offset><uint64 size>
//   Terminated by an entry whose name == the chunkmap signature.
// =========================================================================

namespace {

// ----------------- low-level constants -----------------

constexpr std::uint32_t ND2_CHUNK_MAGIC = 0x0ABECEDAu;
// 32-byte signature appearing at the beginning of every ND2 file.
const char ND2_FILE_SIGNATURE[33] = "ND2 FILE SIGNATURE CHUNK NAME01!";
// 32-byte signature at the very end of an ND2 file (last 40 bytes:
// signature + uint64 chunkmap-offset).
const char ND2_CHUNKMAP_SIGNATURE[33] = "ND2 CHUNK MAP SIGNATURE 0000001!";
// 32-byte signature that is the *name* of the chunkmap chunk itself.
const char ND2_FILEMAP_SIGNATURE[33] = "ND2 FILEMAP SIGNATURE NAME 0001!";

constexpr std::size_t kChunkHeaderSize = 16;
constexpr std::size_t kFooterSize = 40;
constexpr std::size_t kFrameInnerHeaderSkip = 8;

// Compression magics on the inner frame payload (after the 8-byte skip).
constexpr std::uint8_t kZlibFirstByte = 0x78;
constexpr std::uint8_t kLZ4Magic[4] = { 0x04, 0x22, 0x4D, 0x18 };

// All ND2 binary integer fields are little-endian.
inline std::uint32_t
readLE32(const std::uint8_t* p)
{
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}
inline std::uint64_t
readLE64(const std::uint8_t* p)
{
  std::uint64_t lo = readLE32(p);
  std::uint64_t hi = readLE32(p + 4);
  return lo | (hi << 32);
}
inline std::int32_t
readLEi32(const std::uint8_t* p)
{
  return static_cast<std::int32_t>(readLE32(p));
}
inline std::int64_t
readLEi64(const std::uint8_t* p)
{
  return static_cast<std::int64_t>(readLE64(p));
}
inline double
readLEdouble(const std::uint8_t* p)
{
  std::uint64_t bits = readLE64(p);
  double d;
  std::memcpy(&d, &bits, sizeof(d));
  return d;
}

// ----------------- file IO wrapper -----------------

// Returns true if `s` looks like an http:// or https:// URL.
inline bool
isHttpUrl(const std::string& s)
{
  auto starts = [&](const char* prefix) {
    std::size_t n = std::strlen(prefix);
    if (s.size() < n)
      return false;
    for (std::size_t i = 0; i < n; ++i) {
      if (std::tolower(static_cast<unsigned char>(s[i])) != prefix[i]) {
        return false;
      }
    }
    return true;
  };
  return starts("http://") || starts("https://");
}

// One-time global libcurl init.
inline void
ensureCurlGlobalInit()
{
  static std::once_flag flag;
  std::call_once(flag, []() { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

// Owns either a local file handle or a libcurl easy handle bound to an
// HTTP(S) URL, and provides read-at-offset semantics.
//
// Threading: a single FileStream instance is *not* thread-safe and must be
// used from one thread at a time. To read concurrently from the same
// backing file or URL, construct multiple FileStream instances (one per
// worker thread). HTTP clones can pass a known size to skip the HEAD probe.
class FileStream
{
public:
  explicit FileStream(const std::string& path)
    : m_path(path)
    , m_isHttp(isHttpUrl(path))
  {
    if (m_isHttp) {
      openHttp(/*knownSize*/ 0, /*haveSize*/ false);
    } else {
      openLocal();
    }
  }
  // Cheap "clone" constructor: opens an independent handle on the same
  // backing path/URL, reusing a previously-discovered total size to skip
  // the HEAD probe on HTTP. Caller must pass the size from a sibling
  // FileStream of the same resource.
  FileStream(const std::string& path, std::uint64_t knownSize)
    : m_path(path)
    , m_isHttp(isHttpUrl(path))
  {
    if (m_isHttp) {
      openHttp(knownSize, /*haveSize*/ true);
    } else {
      openLocal();
    }
  }
  ~FileStream()
  {
    if (m_curl) {
      curl_easy_cleanup(m_curl);
      m_curl = nullptr;
    }
  }
  FileStream(const FileStream&) = delete;
  FileStream& operator=(const FileStream&) = delete;

  std::uint64_t size() const { return m_size; }

  void readAt(std::uint64_t offset, void* dst, std::uint64_t bytes)
  {
    if (bytes == 0)
      return;
    if (m_isHttp) {
      readHttp(offset, dst, bytes);
    } else {
      readLocal(offset, dst, bytes);
    }
  }
  std::vector<std::uint8_t> readBytes(std::uint64_t offset, std::uint64_t bytes)
  {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(bytes));
    if (bytes > 0) {
      readAt(offset, out.data(), bytes);
    }
    return out;
  }

private:
  // ---- local file backend ----
  void openLocal()
  {
    m_stream.open(m_path, std::ios::binary);
    if (!m_stream.is_open()) {
      throw std::runtime_error("ND2: failed to open file '" + m_path + "'");
    }
    m_stream.seekg(0, std::ios::end);
    m_size = static_cast<std::uint64_t>(m_stream.tellg());
    m_stream.seekg(0, std::ios::beg);
  }
  void readLocal(std::uint64_t offset, void* dst, std::uint64_t bytes)
  {
    m_stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!m_stream) {
      throw std::runtime_error("ND2: seek failed in '" + m_path + "'");
    }
    m_stream.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    if (m_stream.gcount() != static_cast<std::streamsize>(bytes)) {
      throw std::runtime_error("ND2: short read in '" + m_path + "'");
    }
  }

  // ---- HTTP backend ----
  struct WriteCtx
  {
    std::uint8_t* dst;
    std::uint64_t cap;
    std::uint64_t got;
  };
  static std::size_t writeCb(char* ptr, std::size_t size, std::size_t nmemb, void* userdata)
  {
    auto* ctx = static_cast<WriteCtx*>(userdata);
    std::size_t bytes = size * nmemb;
    if (ctx->got + bytes > ctx->cap) {
      // Server sent more than we asked for; clip.
      bytes = static_cast<std::size_t>(ctx->cap - ctx->got);
    }
    if (bytes > 0) {
      std::memcpy(ctx->dst + ctx->got, ptr, bytes);
      ctx->got += bytes;
    }
    return size * nmemb; // tell curl we consumed everything to avoid abort
  }
  void openHttp(std::uint64_t knownSize, bool haveSize)
  {
    ensureCurlGlobalInit();
    m_curl = curl_easy_init();
    if (!m_curl) {
      throw std::runtime_error("ND2: curl_easy_init failed");
    }
    curl_easy_setopt(m_curl, CURLOPT_URL, m_path.c_str());
    curl_easy_setopt(m_curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(m_curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(m_curl, CURLOPT_CONNECTTIMEOUT, 30L);
    // Encourage connection reuse for back-to-back range requests, which is
    // the dominant access pattern when many workers each issue a stream of
    // reads against the same NAS endpoint.
    curl_easy_setopt(m_curl, CURLOPT_TCP_KEEPALIVE, 1L);

    if (haveSize) {
      // Skip the HEAD probe: caller already discovered Content-Length on a
      // sibling handle. Configure for range GETs immediately.
      m_size = knownSize;
      curl_easy_setopt(m_curl, CURLOPT_HTTPGET, 1L);
      curl_easy_setopt(m_curl, CURLOPT_WRITEFUNCTION, &FileStream::writeCb);
      return;
    }

    // HEAD request to discover content length and confirm reachability.
    curl_easy_setopt(m_curl, CURLOPT_NOBODY, 1L);
    CURLcode rc = curl_easy_perform(m_curl);
    if (rc != CURLE_OK) {
      std::string err = curl_easy_strerror(rc);
      curl_easy_cleanup(m_curl);
      m_curl = nullptr;
      throw std::runtime_error("ND2: HEAD failed for '" + m_path + "': " + err);
    }
    long httpCode = 0;
    curl_easy_getinfo(m_curl, CURLINFO_RESPONSE_CODE, &httpCode);
    if (httpCode >= 400) {
      curl_easy_cleanup(m_curl);
      m_curl = nullptr;
      throw std::runtime_error("ND2: HEAD returned HTTP " + std::to_string(httpCode) + " for '" + m_path + "'");
    }
    curl_off_t len = -1;
    curl_easy_getinfo(m_curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &len);
    if (len < 0) {
      curl_easy_cleanup(m_curl);
      m_curl = nullptr;
      throw std::runtime_error("ND2: server did not report Content-Length for '" + m_path + "'");
    }
    m_size = static_cast<std::uint64_t>(len);
    // Switch the handle to GET for subsequent range reads.
    curl_easy_setopt(m_curl, CURLOPT_NOBODY, 0L);
    curl_easy_setopt(m_curl, CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(m_curl, CURLOPT_WRITEFUNCTION, &FileStream::writeCb);
  }
  void readHttp(std::uint64_t offset, void* dst, std::uint64_t bytes)
  {
    char range[64];
    std::snprintf(range,
                  sizeof(range),
                  "%llu-%llu",
                  static_cast<unsigned long long>(offset),
                  static_cast<unsigned long long>(offset + bytes - 1));
    WriteCtx ctx{ static_cast<std::uint8_t*>(dst), bytes, 0 };
    curl_easy_setopt(m_curl, CURLOPT_RANGE, range);
    curl_easy_setopt(m_curl, CURLOPT_WRITEDATA, &ctx);
    CURLcode rc = curl_easy_perform(m_curl);
    if (rc != CURLE_OK) {
      throw std::runtime_error(std::string("ND2: HTTP range read failed: ") + curl_easy_strerror(rc));
    }
    long httpCode = 0;
    curl_easy_getinfo(m_curl, CURLINFO_RESPONSE_CODE, &httpCode);
    // 206 = partial content; 200 is acceptable only if server returned
    // the whole file and we got at least the bytes we wanted.
    if (httpCode != 206 && httpCode != 200) {
      throw std::runtime_error("ND2: unexpected HTTP status " + std::to_string(httpCode) + " for range read of '" +
                               m_path + "'");
    }
    if (ctx.got < bytes) {
      throw std::runtime_error("ND2: short HTTP read (" + std::to_string(ctx.got) + " of " + std::to_string(bytes) +
                               " bytes) from '" + m_path + "'");
    }
  }

  std::string m_path;
  bool m_isHttp{ false };
  std::ifstream m_stream;
  CURL* m_curl{ nullptr };
  std::uint64_t m_size{ 0 };
};

// ----------------- chunk reading -----------------

struct ChunkLoc
{
  std::uint64_t offset; // file offset of the chunk header
  std::uint64_t size;   // byte length of the payload
};
using ChunkMap = std::map<std::string, ChunkLoc>;

// Reads exactly one chunk's payload from `start_position`. Optionally
// validates that the chunk's name *starts with* `expectName`.
void
readNd2ChunkInto(FileStream& fs,
                 std::uint64_t startPosition,
                 std::vector<std::uint8_t>& out,
                 const std::string& expectName = std::string())
{
  std::uint8_t header[kChunkHeaderSize];
  fs.readAt(startPosition, header, kChunkHeaderSize);
  std::uint32_t magic = readLE32(header);
  std::uint32_t nameLen = readLE32(header + 4);
  std::uint64_t dataLen = readLE64(header + 8);
  if (magic != ND2_CHUNK_MAGIC) {
    throw std::runtime_error("ND2: invalid chunk magic at offset " + std::to_string(startPosition));
  }
  if (!expectName.empty()) {
    std::vector<std::uint8_t> nameBuf(nameLen);
    if (nameLen > 0) {
      fs.readAt(startPosition + kChunkHeaderSize, nameBuf.data(), nameLen);
    }
    std::string actual(reinterpret_cast<const char*>(nameBuf.data()), nameBuf.size());
    if (actual.compare(0, expectName.size(), expectName) != 0) {
      throw std::runtime_error("ND2: expected chunk name '" + expectName + "' but got '" + actual + "'");
    }
  }
  out.resize(static_cast<std::size_t>(dataLen));
  if (dataLen > 0) {
    fs.readAt(startPosition + kChunkHeaderSize + nameLen, out.data(), dataLen);
  }
}

std::vector<std::uint8_t>
readNd2Chunk(FileStream& fs, std::uint64_t startPosition, const std::string& expectName = std::string())
{
  std::vector<std::uint8_t> out;
  readNd2ChunkInto(fs, startPosition, out, expectName);
  return out;
}

// Fast path for chunks whose location was looked up in the chunkmap: the
// payload offset is `headerOffset + 16 + nameLen` and the size is already
// known, so no per-chunk header round-trip is needed. This eliminates one
// of the two reads per frame in the inner loop of loadFromFile -- a major
// latency win on networked storage.
void
readKnownChunkPayloadInto(FileStream& fs,
                          std::uint64_t headerOffset,
                          std::size_t nameLen,
                          std::uint64_t payloadSize,
                          std::vector<std::uint8_t>& out)
{
  out.resize(static_cast<std::size_t>(payloadSize));
  if (payloadSize > 0) {
    fs.readAt(headerOffset + kChunkHeaderSize + nameLen, out.data(), payloadSize);
  }
}

// Verify the file header chunk magic and read the (major, minor) version
// out of its 64-byte data block (which begins with e.g. "Ver3.0").
std::pair<int, int>
verifyAndGetVersion(FileStream& fs)
{
  std::uint8_t header[kChunkHeaderSize + 32 + 64];
  if (fs.size() < sizeof(header)) {
    throw std::runtime_error("ND2: file too small");
  }
  fs.readAt(0, header, sizeof(header));
  std::uint32_t magic = readLE32(header);
  if (magic != ND2_CHUNK_MAGIC) {
    throw std::runtime_error("ND2: not an ND2 file (bad start magic)");
  }
  std::string name(reinterpret_cast<const char*>(header + kChunkHeaderSize), 32);
  // tolerate trailing '\0' inside fixed-size name slot
  std::size_t end = name.find('\0');
  if (end != std::string::npos) {
    name.resize(end);
  }
  if (name.compare(0, std::strlen(ND2_FILE_SIGNATURE), ND2_FILE_SIGNATURE) != 0) {
    throw std::runtime_error("ND2: not an ND2 file (bad start name '" + name + "')");
  }
  // data is at header+kChunkHeaderSize+32, length 64. Format "Ver%d.%d".
  const char* d = reinterpret_cast<const char*>(header + kChunkHeaderSize + 32);
  int major = 0;
  int minor = 0;
  // robust-ish: walk to first digit then parse two ints separated by '.'
  std::size_t i = 0;
  while (i < 64 && (d[i] < '0' || d[i] > '9')) {
    ++i;
  }
  while (i < 64 && d[i] >= '0' && d[i] <= '9') {
    major = major * 10 + (d[i] - '0');
    ++i;
  }
  if (i < 64 && d[i] == '.') {
    ++i;
    while (i < 64 && d[i] >= '0' && d[i] <= '9') {
      minor = minor * 10 + (d[i] - '0');
      ++i;
    }
  }
  return { major, minor };
}

// Parse the chunkmap from the trailing footer of the file.
ChunkMap
readChunkMap(FileStream& fs)
{
  if (fs.size() < kFooterSize) {
    throw std::runtime_error("ND2: file too small for chunkmap footer");
  }
  std::uint8_t footer[kFooterSize];
  fs.readAt(fs.size() - kFooterSize, footer, kFooterSize);
  std::string sig(reinterpret_cast<const char*>(footer), 32);
  std::size_t end = sig.find('\0');
  if (end != std::string::npos) {
    sig.resize(end);
  }
  if (sig.compare(0, std::strlen(ND2_CHUNKMAP_SIGNATURE), ND2_CHUNKMAP_SIGNATURE) != 0) {
    throw std::runtime_error("ND2: missing chunkmap footer signature");
  }
  std::uint64_t mapOffset = readLE64(footer + 32);

  // Read the chunkmap chunk itself, validate its name, and walk its body.
  std::vector<std::uint8_t> data = readNd2Chunk(fs, mapOffset, ND2_FILEMAP_SIGNATURE);

  ChunkMap result;
  std::size_t cursor = 0;
  const std::size_t n = data.size();
  while (cursor < n) {
    // Find next '!' starting at cursor (chunk names always end with '!').
    std::size_t bang = cursor;
    while (bang < n && data[bang] != '!') {
      ++bang;
    }
    if (bang >= n) {
      break;
    }
    ++bang; // include trailing '!'
    std::string name(reinterpret_cast<const char*>(data.data() + cursor), bang - cursor);
    cursor = bang;
    if (name.compare(0, std::strlen(ND2_CHUNKMAP_SIGNATURE), ND2_CHUNKMAP_SIGNATURE) == 0) {
      // sentinel: end of chunkmap
      break;
    }
    if (cursor + 16 > n) {
      throw std::runtime_error("ND2: chunkmap entry truncated");
    }
    ChunkLoc loc;
    loc.offset = readLE64(data.data() + cursor);
    loc.size = readLE64(data.data() + cursor + 8);
    cursor += 16;
    result.emplace(std::move(name), loc);
  }
  return result;
}

// ----------------- CLX-Lite metadata variant parser -----------------
//
// CLX-Lite is the binary tagged value format used by ND2 to encode all
// hierarchical metadata (image attributes, experiment loops, channels,
// stage positions). Each record is:
//
//   uint8  type
//   uint8  name_length     (count of utf16 code units; includes the
//                           terminating null)
//   bytes  name            (name_length * 2 bytes utf16, last 2 bytes
//                           are 0x00 0x00)
//   bytes  payload         (depends on type)
//
// Type codes (from tlambert03/nd2 _clx_lite.py):
//   0  UNKNOWN (skip)
//   1  BOOL          1 byte
//   2  INT32         4 bytes
//   3  UINT32        4 bytes
//   4  INT64         8 bytes
//   5  UINT64        8 bytes
//   6  DOUBLE        8 bytes
//   7  VOID*         8 bytes (ignored)
//   8  STRING        utf16 string, terminated by 0x00 0x00 pair
//   9  BYTEARRAY     uint64 size + raw bytes
//  10  DEPRECATED (ignored)
//  11  LEVEL         uint32 item_count + uint64 length, then a nested
//                    sequence of `item_count` records inside the next
//                    `length - header_size` bytes; followed by
//                    item_count*8 trailing bytes that we skip.
//  76  COMPRESS      'L' (= 76). Skip 10 bytes, zlib-inflate the rest,
//                    parse recursively.
//
// Lists (e.g. SLxExperiment::ppNextLevelEx) are encoded as a LEVEL whose
// children all have the empty name "". Our parser collects same-named
// values into a std::vector<LiteValue>.

enum class LType : std::uint8_t
{
  Unknown = 0,
  Bool = 1,
  Int32 = 2,
  UInt32 = 3,
  Int64 = 4,
  UInt64 = 5,
  Double = 6,
  VoidPtr = 7,
  String = 8,
  ByteArray = 9,
  Deprecated = 10,
  Level = 11,
  Compress = 76
};

class LiteValue;
using LiteDict = std::map<std::string, LiteValue>;
using LiteList = std::vector<LiteValue>;

class LiteValue
{
public:
  using Storage =
    std::variant<std::monostate, bool, std::int64_t, std::uint64_t, double, std::string, LiteDict, LiteList>;
  Storage v;

  LiteValue() = default;
  LiteValue(bool x) { v = x; }
  LiteValue(std::int64_t x) { v = x; }
  LiteValue(std::uint64_t x) { v = x; }
  LiteValue(double x) { v = x; }
  LiteValue(std::string x) { v = std::move(x); }
  LiteValue(LiteDict x) { v = std::move(x); }
  LiteValue(LiteList x) { v = std::move(x); }

  bool isDict() const { return std::holds_alternative<LiteDict>(v); }
  bool isList() const { return std::holds_alternative<LiteList>(v); }
  bool isString() const { return std::holds_alternative<std::string>(v); }
  bool isInteger() const
  {
    return std::holds_alternative<std::int64_t>(v) || std::holds_alternative<std::uint64_t>(v) ||
           std::holds_alternative<bool>(v);
  }
  bool isDouble() const { return std::holds_alternative<double>(v); }
  bool isNumber() const { return isInteger() || isDouble(); }

  const LiteDict* asDict() const { return std::get_if<LiteDict>(&v); }
  LiteDict* asDict() { return std::get_if<LiteDict>(&v); }
  const LiteList* asList() const { return std::get_if<LiteList>(&v); }
  const std::string* asString() const { return std::get_if<std::string>(&v); }

  std::int64_t asInt(std::int64_t defaultVal = 0) const
  {
    if (auto p = std::get_if<std::int64_t>(&v)) {
      return *p;
    }
    if (auto p = std::get_if<std::uint64_t>(&v)) {
      return static_cast<std::int64_t>(*p);
    }
    if (auto p = std::get_if<bool>(&v)) {
      return *p ? 1 : 0;
    }
    if (auto p = std::get_if<double>(&v)) {
      return static_cast<std::int64_t>(*p);
    }
    return defaultVal;
  }
  double asDouble(double defaultVal = 0.0) const
  {
    if (auto p = std::get_if<double>(&v)) {
      return *p;
    }
    if (auto p = std::get_if<std::int64_t>(&v)) {
      return static_cast<double>(*p);
    }
    if (auto p = std::get_if<std::uint64_t>(&v)) {
      return static_cast<double>(*p);
    }
    return defaultVal;
  }
  std::string asStr(const std::string& defaultVal = std::string()) const
  {
    if (auto p = std::get_if<std::string>(&v)) {
      return *p;
    }
    return defaultVal;
  }
};

// Convenience: dotted-path lookups into a LiteValue tree.
const LiteValue*
findKey(const LiteValue& root, const std::string& key)
{
  if (auto d = root.asDict()) {
    auto it = d->find(key);
    if (it != d->end()) {
      return &it->second;
    }
  }
  return nullptr;
}
// Look up `key`, or `wrapperKey` -> `key` if present (handles e.g.
// SLxImageAttributes wrapper). Returns nullptr if neither found.
const LiteValue*
findUnwrapped(const LiteValue& root, std::initializer_list<std::string> keys)
{
  const LiteValue* cur = &root;
  for (const std::string& k : keys) {
    const LiteValue* next = findKey(*cur, k);
    if (!next) {
      return nullptr;
    }
    cur = next;
  }
  return cur;
}

// Decode a CLX-Lite utf16 string ending at the first 0x0000 pair.
std::string
decodeUtf16Z(const std::uint8_t* data, std::size_t maxBytes, std::size_t& consumedBytes)
{
  // Find the terminating 0x00 0x00 pair on an even boundary.
  std::size_t i = 0;
  while (i + 1 < maxBytes) {
    if (data[i] == 0 && data[i + 1] == 0) {
      break;
    }
    i += 2;
  }
  std::size_t units = i / 2;
  consumedBytes = i + 2; // include terminator
  // Convert utf16-le -> utf8 (best-effort; assume BMP).
  std::string out;
  out.reserve(units);
  for (std::size_t u = 0; u < units; ++u) {
    std::uint16_t cu = static_cast<std::uint16_t>(data[u * 2]) | (static_cast<std::uint16_t>(data[u * 2 + 1]) << 8);
    if (cu < 0x80) {
      out.push_back(static_cast<char>(cu));
    } else if (cu < 0x800) {
      out.push_back(static_cast<char>(0xC0 | (cu >> 6)));
      out.push_back(static_cast<char>(0x80 | (cu & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xE0 | (cu >> 12)));
      out.push_back(static_cast<char>(0x80 | ((cu >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cu & 0x3F)));
    }
  }
  return out;
}

// Strip leading lower-case prefix used by ND2 (e.g. "uiWidth" -> "Width",
// "dPosX" -> "PosX"). Used by some chunks; we keep raw names by default.
std::string
stripLowerPrefix(const std::string& s)
{
  std::size_t i = 0;
  while (i < s.size() && (s[i] == '_' || (s[i] >= 'a' && s[i] <= 'z'))) {
    ++i;
  }
  return s.substr(i);
}

// Maximum recursion depth for nested CLX-Lite LEVEL / Compress records.
// ND2 metadata is normally only a few levels deep; a hard cap protects
// against stack exhaustion on malformed or pathologically nested input.
constexpr int kLiteMaxDepth = 64;

// Forward decl for recursion.
void
parseLiteSequence(const std::uint8_t* data,
                  std::size_t size,
                  std::size_t count,
                  LiteDict& out,
                  bool stripPrefix,
                  int depth);

// Parse a single CLX-Lite record from `data[offset..size)`. Returns true if
// a record was consumed; false if the buffer is exhausted. On success,
// updates `offset` to point past the record and emplaces (name, value)
// into `out`. Repeated names accumulate into a LiteList.
bool
parseLiteRecord(const std::uint8_t* data,
                std::size_t size,
                std::size_t& offset,
                LiteDict& out,
                bool stripPrefix,
                int depth)
{
  if (depth > kLiteMaxDepth) {
    return false;
  }
  if (offset + 2 > size) {
    return false;
  }
  std::uint8_t typeByte = data[offset];
  std::uint8_t nameLen = data[offset + 1];
  std::size_t curs = offset; // start-of-record (used by LEVEL)
  offset += 2;

  std::string name;
  if (typeByte == static_cast<std::uint8_t>(LType::Compress)) {
    // No name; payload = 10-byte skip + zlib-deflated stream.
    if (offset + 10 > size) {
      return false;
    }
    offset += 10;
    if (offset > size) {
      return false;
    }
    std::vector<std::uint8_t> deflated(data + offset, data + size);
    offset = size;
    // zlib-inflate
    z_stream zs{};
    zs.next_in = deflated.data();
    zs.avail_in = static_cast<uInt>(deflated.size());
    if (inflateInit(&zs) != Z_OK) {
      return false;
    }
    std::vector<std::uint8_t> inflated;
    inflated.reserve(deflated.size() * 4);
    // NOTE: previously this buffer was on the stack as `std::uint8_t buf[64 *
    // 1024]`. Because parseLiteRecord recurses (Compress payloads can
    // themselves contain Compress/Level records), a chain of nested Compress
    // records would consume 64 KB of stack per frame and overflow the
    // default 1 MB Windows thread stack within ~16 levels. Heap-allocate it.
    std::vector<std::uint8_t> buf(64 * 1024);
    int rc = Z_OK;
    do {
      zs.next_out = buf.data();
      zs.avail_out = static_cast<uInt>(buf.size());
      rc = inflate(&zs, Z_NO_FLUSH);
      if (rc != Z_OK && rc != Z_STREAM_END) {
        inflateEnd(&zs);
        return false;
      }
      inflated.insert(inflated.end(), buf.data(), buf.data() + (buf.size() - zs.avail_out));
    } while (rc != Z_STREAM_END);
    inflateEnd(&zs);
    parseLiteSequence(inflated.data(), inflated.size(), 1u, out, stripPrefix, depth + 1);
    return true;
  }

  // For all other types, read the utf16 name.
  if (offset + static_cast<std::size_t>(nameLen) * 2u > size) {
    return false;
  }
  // The name field is `name_length` utf16 code units, the last of which is
  // the null terminator. Decode, then drop the trailing null character.
  std::string raw;
  if (nameLen > 0) {
    std::size_t consumed = 0;
    raw = decodeUtf16Z(data + offset, static_cast<std::size_t>(nameLen) * 2u, consumed);
    // raw should already exclude the terminator by virtue of decodeUtf16Z
    // stopping at 0x0000, but if the field contains no terminator we keep
    // what we got.
    (void)consumed;
  }
  offset += static_cast<std::size_t>(nameLen) * 2u;
  name = stripPrefix ? stripLowerPrefix(raw) : raw;

  auto emit = [&](LiteValue&& value) {
    auto it = out.find(name);
    if (it == out.end()) {
      out.emplace(name, std::move(value));
    } else {
      // Existing value present -> coalesce into a list.
      if (auto* list = std::get_if<LiteList>(&it->second.v)) {
        list->push_back(std::move(value));
      } else {
        LiteList l;
        l.push_back(std::move(it->second));
        l.push_back(std::move(value));
        it->second = LiteValue(std::move(l));
      }
    }
  };

  switch (static_cast<LType>(typeByte)) {
    case LType::Bool: {
      if (offset + 1 > size) {
        return false;
      }
      bool b = data[offset] != 0;
      offset += 1;
      emit(LiteValue(b));
      return true;
    }
    case LType::Int32: {
      if (offset + 4 > size) {
        return false;
      }
      emit(LiteValue(static_cast<std::int64_t>(readLEi32(data + offset))));
      offset += 4;
      return true;
    }
    case LType::UInt32: {
      if (offset + 4 > size) {
        return false;
      }
      emit(LiteValue(static_cast<std::int64_t>(readLE32(data + offset))));
      offset += 4;
      return true;
    }
    case LType::Int64: {
      if (offset + 8 > size) {
        return false;
      }
      emit(LiteValue(static_cast<std::int64_t>(readLEi64(data + offset))));
      offset += 8;
      return true;
    }
    case LType::UInt64: {
      if (offset + 8 > size) {
        return false;
      }
      emit(LiteValue(static_cast<std::uint64_t>(readLE64(data + offset))));
      offset += 8;
      return true;
    }
    case LType::Double: {
      if (offset + 8 > size) {
        return false;
      }
      emit(LiteValue(readLEdouble(data + offset)));
      offset += 8;
      return true;
    }
    case LType::VoidPtr: {
      if (offset + 8 > size) {
        return false;
      }
      offset += 8; // ignore
      return true;
    }
    case LType::String: {
      std::size_t consumed = 0;
      if (offset > size) {
        return false;
      }
      std::string s = decodeUtf16Z(data + offset, size - offset, consumed);
      offset += consumed;
      emit(LiteValue(std::move(s)));
      return true;
    }
    case LType::ByteArray: {
      if (offset + 8 > size) {
        return false;
      }
      std::uint64_t bsz = readLE64(data + offset);
      offset += 8;
      if (offset + bsz > size) {
        return false;
      }
      // We don't need byte-arrays for AGAVE's purposes (used internally by
      // ND2 for thumbnails / job definitions). Skip the bytes.
      offset += static_cast<std::size_t>(bsz);
      return true;
    }
    case LType::Level: {
      if (offset + 12 > size) {
        return false;
      }
      std::uint32_t itemCount = readLE32(data + offset);
      std::uint64_t length = readLE64(data + offset + 4);
      offset += 12;
      // The full LEVEL block (header + nested data) occupies `length`
      // bytes starting from `curs`. Nested children fill the remainder
      // after the header.
      if (length < (offset - curs)) {
        return false;
      }
      std::size_t bodySize = static_cast<std::size_t>(length) - (offset - curs);
      if (offset + bodySize > size) {
        return false;
      }
      LiteDict nested;
      parseLiteSequence(data + offset, bodySize, itemCount, nested, stripPrefix, depth + 1);
      offset += bodySize;
      // Trailing footer: itemCount * 8 bytes of (offset/index?) we skip.
      std::size_t trailer = static_cast<std::size_t>(itemCount) * 8u;
      if (offset + trailer > size) {
        // Some files appear to under-report; clamp.
        offset = size;
      } else {
        offset += trailer;
      }
      // List heuristic: if the nested dict is just one anonymous entry "",
      // unwrap into a LiteList. Otherwise use the dict.
      auto it = nested.find("");
      if (nested.size() == 1 && it != nested.end()) {
        LiteValue inner = std::move(it->second);
        if (auto* list = std::get_if<LiteList>(&inner.v)) {
          // already a list
          emit(LiteValue(std::move(*list)));
        } else {
          LiteList l;
          l.push_back(std::move(inner));
          emit(LiteValue(std::move(l)));
        }
      } else {
        emit(LiteValue(std::move(nested)));
      }
      return true;
    }
    case LType::Unknown:
    case LType::Deprecated:
    default:
      // Unknown -> bail out; we may still have read some valid records.
      return false;
  }
}

void
parseLiteSequence(const std::uint8_t* data,
                  std::size_t size,
                  std::size_t count,
                  LiteDict& out,
                  bool stripPrefix,
                  int depth)
{
  if (depth > kLiteMaxDepth) {
    return;
  }
  std::size_t offset = 0;
  for (std::size_t i = 0; i < count; ++i) {
    if (offset >= size) {
      break;
    }
    std::size_t prevOffset = offset;
    if (!parseLiteRecord(data, size, offset, out, stripPrefix, depth)) {
      break;
    }
    // Safety: parseLiteRecord must always advance. Without this guard a
    // record that returns true without consuming bytes would spin forever.
    if (offset == prevOffset) {
      break;
    }
  }
}

// Public entry: parse a chunk's payload as CLX-Lite or XML variant.
LiteValue
decodeChunkVariant(const std::vector<std::uint8_t>& data)
{
  // XML variant starts with '<'. We don't have an XML parser dependency
  // here; for AGAVE's needs (post-2017 ND2) the binary CLX-Lite path is
  // sufficient. If we ever encounter the XML variant we return an empty
  // dict and let upper layers fall back to defaults.
  if (!data.empty() && data[0] == '<') {
    LOG_WARNING << "ND2: XML metadata variant not implemented; ignoring";
    return LiteValue(LiteDict{});
  }
  LiteDict root;
  parseLiteSequence(data.data(), data.size(), 1u, root, false, 0);
  return LiteValue(std::move(root));
}

// ----------------- frame chunk decompression -----------------

// Decompress (or pass through) a frame's pixel-data payload into a caller-
// provided output buffer. Input is the raw chunk data with the leading
// 8-byte inner header *already skipped*. `expectedBytes` is the known
// uncompressed size (sizeX * sizeY * bytesPerPixel * componentCount) and is
// used to pre-size `out` so the decoder writes contiguously into it without
// reallocations or staging buffers.
void
decompressFrameBytesInto(const std::uint8_t* src,
                         std::size_t srcSize,
                         std::size_t expectedBytes,
                         std::vector<std::uint8_t>& out)
{
  if (srcSize >= 4 && std::memcmp(src, kLZ4Magic, 4) == 0) {
    // LZ4 frame format. Decompress directly into `out` when the expected
    // size is known; otherwise fall back to a growing output.
    LZ4F_decompressionContext_t ctx = nullptr;
    if (LZ4F_isError(LZ4F_createDecompressionContext(&ctx, LZ4F_VERSION))) {
      throw std::runtime_error("ND2: LZ4 ctx creation failed");
    }
    const std::uint8_t* in = src;
    std::size_t remaining = srcSize;
    if (expectedBytes > 0) {
      out.resize(expectedBytes);
      std::size_t written = 0;
      while (remaining > 0 && written < expectedBytes) {
        std::size_t outAvail = expectedBytes - written;
        std::size_t inUsed = remaining;
        std::size_t hint = LZ4F_decompress(ctx, out.data() + written, &outAvail, in, &inUsed, nullptr);
        if (LZ4F_isError(hint)) {
          LZ4F_freeDecompressionContext(ctx);
          throw std::runtime_error("ND2: LZ4 decompress error");
        }
        written += outAvail;
        in += inUsed;
        remaining -= inUsed;
        if (hint == 0) {
          break; // frame complete
        }
        if (inUsed == 0 && outAvail == 0) {
          break; // defensive: no progress
        }
      }
      out.resize(written);
    } else {
      out.clear();
      out.reserve(srcSize * 4);
      std::uint8_t buf[64 * 1024];
      while (remaining > 0) {
        std::size_t outAvail = sizeof(buf);
        std::size_t inUsed = remaining;
        std::size_t hint = LZ4F_decompress(ctx, buf, &outAvail, in, &inUsed, nullptr);
        if (LZ4F_isError(hint)) {
          LZ4F_freeDecompressionContext(ctx);
          throw std::runtime_error("ND2: LZ4 decompress error");
        }
        out.insert(out.end(), buf, buf + outAvail);
        in += inUsed;
        remaining -= inUsed;
        if (hint == 0) {
          break;
        }
        if (inUsed == 0 && outAvail == 0) {
          break;
        }
      }
    }
    LZ4F_freeDecompressionContext(ctx);
    return;
  }
  if (srcSize >= 1 && src[0] == kZlibFirstByte) {
    // zlib stream. When expected size is known, inflate directly into `out`.
    z_stream zs{};
    zs.next_in = const_cast<Bytef*>(src);
    zs.avail_in = static_cast<uInt>(srcSize);
    if (inflateInit(&zs) != Z_OK) {
      throw std::runtime_error("ND2: zlib init failed");
    }
    if (expectedBytes > 0) {
      out.resize(expectedBytes);
      zs.next_out = out.data();
      zs.avail_out = static_cast<uInt>(expectedBytes);
      int rc = inflate(&zs, Z_FINISH);
      if (rc != Z_STREAM_END && rc != Z_OK && rc != Z_BUF_ERROR) {
        inflateEnd(&zs);
        throw std::runtime_error("ND2: zlib inflate error");
      }
      out.resize(expectedBytes - zs.avail_out);
    } else {
      out.clear();
      out.reserve(srcSize * 4);
      std::uint8_t buf[64 * 1024];
      int rc = Z_OK;
      do {
        zs.next_out = buf;
        zs.avail_out = sizeof(buf);
        rc = inflate(&zs, Z_NO_FLUSH);
        if (rc != Z_OK && rc != Z_STREAM_END) {
          inflateEnd(&zs);
          throw std::runtime_error("ND2: zlib inflate error");
        }
        out.insert(out.end(), buf, buf + (sizeof(buf) - zs.avail_out));
      } while (rc != Z_STREAM_END);
    }
    inflateEnd(&zs);
    return;
  }
  // Raw / uncompressed.
  out.assign(src, src + srcSize);
}

std::vector<std::uint8_t>
decompressFrameBytes(const std::uint8_t* src, std::size_t srcSize, std::size_t expectedBytes)
{
  std::vector<std::uint8_t> out;
  decompressFrameBytesInto(src, srcSize, expectedBytes, out);
  return out;
}

// ----------------- experiment-loop / dimension extraction -----------------

// ND2 experiment-loop type codes (from tlambert03/nd2 structures.py).
constexpr std::int64_t kLoopTime = 1;
constexpr std::int64_t kLoopXYPos = 2;
constexpr std::int64_t kLoopXYDiscrete = 3;
constexpr std::int64_t kLoopZStack = 4;
constexpr std::int64_t kLoopSpect = 6;
constexpr std::int64_t kLoopNETime = 8;
constexpr std::int64_t kLoopManTime = 9;
constexpr std::int64_t kLoopZStackAccurate = 10;

struct LoopInfo
{
  std::int64_t type{ 0 };
  std::uint32_t count{ 1 };
  double zStep{ 0.0 }; // microns
};

// Walk nested ppNextLevelEx structures and flatten into outermost->innermost
// order. We tolerate missing levels and treat unknown loop types as a
// generic "1-count" passthrough.
void
collectLoops(const LiteValue& node, std::vector<LoopInfo>& out)
{
  // Each level is a dict; "eType" + "uLoopPars" + "ppNextLevelEx".
  const LiteValue* etype = findKey(node, "eType");
  const LiteValue* params = findKey(node, "uLoopPars");
  if (etype && params) {
    LoopInfo info;
    info.type = etype->asInt(0);
    if (auto* p = findKey(*params, "uiCount")) {
      info.count = static_cast<std::uint32_t>(p->asInt(1));
    }
    if (auto* p = findKey(*params, "dZStep")) {
      info.zStep = p->asDouble(0.0);
    }
    if (info.count == 0) {
      info.count = 1;
    }
    out.push_back(info);
  }
  // Recurse into nested level(s). ND2 stores them under "ppNextLevelEx" or
  // "pNextLevelEx"; either may be a dict (a single child) or a list (with
  // anonymous keys).
  for (const char* k : { "ppNextLevelEx", "pNextLevelEx" }) {
    if (auto* next = findKey(node, k)) {
      if (auto* l = next->asList()) {
        for (const auto& item : *l) {
          collectLoops(item, out);
        }
      } else if (next->isDict()) {
        // dict of child keys -> recurse into each value
        for (const auto& kv : *next->asDict()) {
          collectLoops(kv.second, out);
        }
      }
    }
  }
}

struct Nd2Layout
{
  // Per-axis size (from experiment loops + image attributes).
  std::uint32_t sizeX{ 0 };
  std::uint32_t sizeY{ 0 };
  std::uint32_t sizeZ{ 1 };
  std::uint32_t sizeC{ 1 };
  std::uint32_t sizeT{ 1 };
  std::uint32_t sizeS{ 1 }; // scenes / XY positions

  std::uint32_t componentCount{ 1 };
  std::uint32_t bitsPerComponent{ 16 };
  std::uint32_t totalSequenceCount{ 0 };

  bool channelsInterleaved{ false };
  std::vector<int> channelToComponent; // interleaved: channel->component idx
  std::vector<std::string> channelNames;

  float physicalSizeX{ 1.f };
  float physicalSizeY{ 1.f };
  float physicalSizeZ{ 1.f };
  std::string spatialUnits{ "um" };

  // Loops in outermost->innermost order. Used to compute the linear frame
  // index for a given (T, S, C, Z) tuple.
  std::vector<LoopInfo> loops;
};

// Compute the linear ImageDataSeq frame index for a given coordinate.
// Channels are passed in only when channels are *not* interleaved (i.e.
// each frame holds a single channel).
std::uint64_t
frameIndex(const Nd2Layout& L, std::uint32_t t, std::uint32_t s, std::uint32_t c, std::uint32_t z)
{
  // Walk loops outermost->innermost; map each loop's coordinate based on
  // its type. Linear index = sum(coord_i * stride_i) where stride_i is the
  // product of all inner loop counts.
  std::uint64_t idx = 0;
  std::uint64_t stride = 1;
  // Compute strides right-to-left.
  for (std::size_t i = L.loops.size(); i-- > 0;) {
    const auto& loop = L.loops[i];
    std::uint32_t coord = 0;
    switch (loop.type) {
      case kLoopTime:
      case kLoopNETime:
      case kLoopManTime:
        coord = t;
        break;
      case kLoopXYPos:
      case kLoopXYDiscrete:
        coord = s;
        break;
      case kLoopZStack:
      case kLoopZStackAccurate:
        coord = z;
        break;
      case kLoopSpect:
        coord = c;
        break;
      default:
        coord = 0;
        break;
    }
    if (coord >= loop.count) {
      coord = loop.count - 1;
    }
    idx += static_cast<std::uint64_t>(coord) * stride;
    stride *= loop.count;
  }
  return idx;
}

// Fill in the layout from the ImageAttributesLV and ImageMetadataLV /
// ImageMetadataSeqLV|0 chunks.
Nd2Layout
buildLayout(FileStream& fs, const ChunkMap& chunks)
{
  Nd2Layout L;

  auto findChunk = [&](const std::string& name) -> const ChunkLoc* {
    auto it = chunks.find(name);
    if (it == chunks.end()) {
      return nullptr;
    }
    return &it->second;
  };

  // ImageAttributesLV! -> SLxImageAttributes -> dimensions, dtype, components.
  if (const ChunkLoc* loc = findChunk("ImageAttributesLV!")) {
    auto data = readNd2Chunk(fs, loc->offset);
    LiteValue root = decodeChunkVariant(data);
    const LiteValue* attrs = &root;
    if (auto* w = findKey(root, "SLxImageAttributes")) {
      attrs = w;
    }
    if (auto* p = findKey(*attrs, "uiWidth")) {
      L.sizeX = static_cast<std::uint32_t>(p->asInt(0));
    }
    if (auto* p = findKey(*attrs, "uiHeight")) {
      L.sizeY = static_cast<std::uint32_t>(p->asInt(0));
    }
    if (auto* p = findKey(*attrs, "uiBpcSignificant")) {
      L.bitsPerComponent = static_cast<std::uint32_t>(p->asInt(16));
    }
    if (auto* p = findKey(*attrs, "uiComp")) {
      L.componentCount = static_cast<std::uint32_t>(p->asInt(1));
    }
    if (auto* p = findKey(*attrs, "uiSequenceCount")) {
      L.totalSequenceCount = static_cast<std::uint32_t>(p->asInt(0));
    }
  }

  // ImageMetadataLV! -> SLxExperiment -> ppNextLevelEx (loops).
  if (const ChunkLoc* loc = findChunk("ImageMetadataLV!")) {
    auto data = readNd2Chunk(fs, loc->offset);
    LiteValue root = decodeChunkVariant(data);
    const LiteValue* exp = findKey(root, "SLxExperiment");
    if (!exp) {
      exp = &root;
    }
    collectLoops(*exp, L.loops);

    // Physical Z step: pick the first ZStack loop's dZStep.
    for (const auto& lp : L.loops) {
      if ((lp.type == kLoopZStack || lp.type == kLoopZStackAccurate) && lp.zStep > 0.0) {
        L.physicalSizeZ = static_cast<float>(lp.zStep);
        break;
      }
    }

    // Derive sizeT, sizeZ, sizeS from loops.
    for (const auto& lp : L.loops) {
      switch (lp.type) {
        case kLoopTime:
        case kLoopNETime:
        case kLoopManTime:
          L.sizeT *= lp.count;
          break;
        case kLoopXYPos:
        case kLoopXYDiscrete:
          L.sizeS *= lp.count;
          break;
        case kLoopZStack:
        case kLoopZStackAccurate:
          L.sizeZ *= lp.count;
          break;
        case kLoopSpect:
          // Channel-as-loop case (not interleaved).
          L.sizeC *= lp.count;
          break;
        default:
          break;
      }
    }
    // sizeT/sizeZ/sizeS/sizeC start at 1 and we *= count, so a loop of
    // count=1 leaves the dim at 1. This is correct.
  }

  // ImageMetadataSeqLV|0! -> per-frame metadata containing channel info.
  if (const ChunkLoc* loc = findChunk("ImageMetadataSeqLV|0!")) {
    auto data = readNd2Chunk(fs, loc->offset);
    LiteValue root = decodeChunkVariant(data);
    // Common path: SLxPictureMetadata::sPicturePlanes::sPlaneNew (or
    // sPlane), which is a dict whose values are the per-channel plane
    // descriptions.
    const LiteValue* md = findKey(root, "SLxPictureMetadata");
    if (!md) {
      md = &root;
    }
    const LiteValue* planes = nullptr;
    if (auto* pp = findKey(*md, "sPicturePlanes")) {
      if (auto* sp = findKey(*pp, "sPlaneNew")) {
        planes = sp;
      } else if (auto* sp = findKey(*pp, "sPlane")) {
        planes = sp;
      }
    }
    if (planes && planes->isDict()) {
      for (const auto& kv : *planes->asDict()) {
        std::string chanName;
        if (auto* p = findKey(kv.second, "sDescription")) {
          chanName = p->asStr();
        }
        if (chanName.empty()) {
          chanName = "Ch" + std::to_string(L.channelNames.size());
        }
        L.channelNames.push_back(chanName);
        // The dict key is the component index as a string ("0", "1", ...).
        try {
          L.channelToComponent.push_back(std::stoi(kv.first));
        } catch (...) {
          L.channelToComponent.push_back(static_cast<int>(L.channelNames.size()) - 1);
        }
      }
    }

    // Physical X/Y pixel size.
    if (auto* p = findKey(*md, "dCalibration")) {
      double cal = p->asDouble(1.0);
      if (cal > 0.0) {
        L.physicalSizeX = static_cast<float>(cal);
        L.physicalSizeY = static_cast<float>(cal);
      }
    }
  }

  // Determine channel-storage mode. If the file's components-per-frame
  // matches the number of channels we extracted from sPicturePlanes, the
  // channels are interleaved into each frame. Otherwise channels are
  // stored as separate frames (and likely already accounted for via a
  // SpectLoop in the loop list).
  if (L.channelNames.empty()) {
    // Fall back: assume one channel; component_count slots are valid.
    L.sizeC = std::max<std::uint32_t>(1u, L.componentCount);
    L.channelsInterleaved = (L.componentCount > 1);
    L.channelNames.clear();
    L.channelToComponent.clear();
    for (std::uint32_t c = 0; c < L.sizeC; ++c) {
      L.channelNames.push_back("Ch" + std::to_string(c));
      L.channelToComponent.push_back(static_cast<int>(c));
    }
  } else {
    bool interleaved = (L.componentCount > 1) && (L.componentCount == L.channelNames.size());
    L.channelsInterleaved = interleaved;
    if (interleaved) {
      // sizeC comes from channelNames.
      L.sizeC = static_cast<std::uint32_t>(L.channelNames.size());
    } else {
      // sizeC may already have been set by a SpectLoop. If not, derive
      // from channelNames.
      if (L.sizeC <= 1u && L.channelNames.size() > 1u) {
        L.sizeC = static_cast<std::uint32_t>(L.channelNames.size());
      }
    }
  }

  // Some files don't expose all dims via loops but do via uiSequenceCount.
  // Sanity-check: product of loop counts should equal uiSequenceCount.
  if (L.totalSequenceCount > 0) {
    std::uint64_t prod = 1;
    for (const auto& lp : L.loops) {
      prod *= lp.count;
    }
    if (prod != L.totalSequenceCount) {
      LOG_WARNING << "ND2: loop product (" << prod << ") != uiSequenceCount (" << L.totalSequenceCount << ")";
    }
  }

  return L;
}

// ----------------- frame -> destination buffer -----------------
//
// AGAVE's in-memory format is uint16 packed (IN_MEMORY_BPP=16). The
// destination buffer is laid out as channel-major Z-slices:
//   data[(channel * sizeZ + z) * sizeX * sizeY * 2] is the start of
//   plane (channel, z) in uint16.

void
copyFrameToDest(const std::vector<std::uint8_t>& frameBytes,
                const Nd2Layout& L,
                std::uint32_t channelIdx,
                std::uint8_t* dst)
{
  const std::size_t pixels = static_cast<std::size_t>(L.sizeX) * static_cast<std::size_t>(L.sizeY);
  const std::size_t bppSrcComp = (L.bitsPerComponent <= 8) ? 1u : 2u;
  const std::size_t srcStride = bppSrcComp * L.componentCount;
  const std::size_t expected = pixels * srcStride;
  if (frameBytes.size() < expected) {
    LOG_WARNING << "ND2: frame data short (" << frameBytes.size() << " < " << expected << ")";
    return;
  }

  // Component within the frame to extract, when interleaved.
  std::size_t comp = 0;
  if (L.channelsInterleaved && channelIdx < L.channelToComponent.size()) {
    comp = static_cast<std::size_t>(L.channelToComponent[channelIdx]);
  }

  std::uint16_t* dst16 = reinterpret_cast<std::uint16_t*>(dst);
  if (bppSrcComp == 2) {
    // 16-bit source -> 16-bit dest.
    if (!L.channelsInterleaved && L.componentCount == 1) {
      // Bulk copy.
      std::memcpy(dst16, frameBytes.data(), expected);
    } else {
      const std::uint8_t* sp = frameBytes.data() + comp * 2u;
      for (std::size_t i = 0; i < pixels; ++i) {
        std::uint16_t v;
        std::memcpy(&v, sp, 2);
        dst16[i] = v;
        sp += srcStride;
      }
    }
  } else {
    // 8-bit source -> 16-bit dest (promote).
    if (!L.channelsInterleaved && L.componentCount == 1) {
      const std::uint8_t* sp = frameBytes.data();
      for (std::size_t i = 0; i < pixels; ++i) {
        dst16[i] = static_cast<std::uint16_t>(sp[i]);
      }
    } else {
      const std::uint8_t* sp = frameBytes.data() + comp;
      for (std::size_t i = 0; i < pixels; ++i) {
        dst16[i] = static_cast<std::uint16_t>(*sp);
        sp += srcStride;
      }
    }
  }
}

// Interleaved-source de-interleave: a single pass over `frameBytes` writes
// pixels for every requested channel into its respective destination plane.
// This replaces N strided gather passes (one per requested channel) with a
// single pass, cutting memory bandwidth on the source frame from O(N*size)
// to O(size).
void
copyFrameToDestMulti(const std::vector<std::uint8_t>& frameBytes,
                     const Nd2Layout& L,
                     const std::vector<std::uint32_t>& srcChannels,
                     const std::vector<std::uint8_t*>& dstPlanes)
{
  const std::size_t pixels = static_cast<std::size_t>(L.sizeX) * static_cast<std::size_t>(L.sizeY);
  const std::size_t bppSrcComp = (L.bitsPerComponent <= 8) ? 1u : 2u;
  const std::size_t srcStride = bppSrcComp * L.componentCount;
  const std::size_t expected = pixels * srcStride;
  const std::size_t nch = srcChannels.size();
  if (frameBytes.size() < expected) {
    LOG_WARNING << "ND2: frame data short (" << frameBytes.size() << " < " << expected << ")";
    return;
  }

  // Resolve the source-component byte offset for each requested channel
  // once, outside the per-pixel loop.
  std::vector<std::size_t> compByteOffsets(nch);
  for (std::size_t k = 0; k < nch; ++k) {
    std::size_t comp = 0;
    if (srcChannels[k] < L.channelToComponent.size()) {
      comp = static_cast<std::size_t>(L.channelToComponent[srcChannels[k]]);
    }
    compByteOffsets[k] = comp * bppSrcComp;
  }

  if (bppSrcComp == 2) {
    const std::uint8_t* base = frameBytes.data();
    for (std::size_t i = 0; i < pixels; ++i) {
      const std::uint8_t* sp = base + i * srcStride;
      for (std::size_t k = 0; k < nch; ++k) {
        std::uint16_t v;
        std::memcpy(&v, sp + compByteOffsets[k], 2);
        reinterpret_cast<std::uint16_t*>(dstPlanes[k])[i] = v;
      }
    }
  } else {
    const std::uint8_t* base = frameBytes.data();
    for (std::size_t i = 0; i < pixels; ++i) {
      const std::uint8_t* sp = base + i * srcStride;
      for (std::size_t k = 0; k < nch; ++k) {
        reinterpret_cast<std::uint16_t*>(dstPlanes[k])[i] = static_cast<std::uint16_t>(sp[compByteOffsets[k]]);
      }
    }
  }
}

} // namespace

// ============================================================
// FileReaderND2 public API
// ============================================================

FileReaderND2::FileReaderND2(const std::string& /*filepath*/) {}
FileReaderND2::~FileReaderND2() = default;

uint32_t
FileReaderND2::loadNumScenes(const std::string& filepath)
{
  try {
    FileStream fs(filepath);
    auto version = verifyAndGetVersion(fs);
    (void)version;
    ChunkMap chunks = readChunkMap(fs);
    Nd2Layout L = buildLayout(fs, chunks);
    return std::max<std::uint32_t>(1u, L.sizeS);
  } catch (const std::exception& e) {
    LOG_ERROR << "ND2 loadNumScenes failed for '" << filepath << "': " << e.what();
    return 0;
  }
}

VolumeDimensions
FileReaderND2::loadDimensions(const std::string& filepath, uint32_t scene)
{
  VolumeDimensions vdims;
  try {
    FileStream fs(filepath);
    verifyAndGetVersion(fs);
    ChunkMap chunks = readChunkMap(fs);
    Nd2Layout L = buildLayout(fs, chunks);
    if (scene >= L.sizeS) {
      LOG_ERROR << "ND2 scene index " << scene << " out of range (sizeS=" << L.sizeS << ")";
      return VolumeDimensions();
    }
    vdims.sizeX = L.sizeX;
    vdims.sizeY = L.sizeY;
    vdims.sizeZ = L.sizeZ;
    vdims.sizeC = L.sizeC;
    vdims.sizeT = L.sizeT;
    vdims.physicalSizeX = L.physicalSizeX;
    vdims.physicalSizeY = L.physicalSizeY;
    vdims.physicalSizeZ = L.physicalSizeZ;
    vdims.spatialUnits = L.spatialUnits;
    vdims.bitsPerPixel = L.bitsPerComponent <= 8 ? 8 : 16;
    vdims.channelNames = L.channelNames;
    if (!vdims.validate()) {
      LOG_ERROR << "ND2 invalid dimensions for '" << filepath << "'";
      return VolumeDimensions();
    }
    return vdims;
  } catch (const std::exception& e) {
    LOG_ERROR << "ND2 loadDimensions failed for '" << filepath << "': " << e.what();
    return VolumeDimensions();
  }
}

std::vector<MultiscaleDims>
FileReaderND2::loadMultiscaleDims(const std::string& filepath, uint32_t scene)
{
  std::vector<MultiscaleDims> out;
  VolumeDimensions vdims = loadDimensions(filepath, scene);
  if (!vdims.validate()) {
    return out;
  }
  MultiscaleDims md;
  md.shape = { vdims.sizeT, vdims.sizeC, vdims.sizeZ, vdims.sizeY, vdims.sizeX };
  md.scale = { 1.0, 1.0, vdims.physicalSizeZ, vdims.physicalSizeY, vdims.physicalSizeX };
  md.dimensionOrder = { "T", "C", "Z", "Y", "X" };
  md.dtype = "uint16";
  md.path = "";
  md.channelNames = vdims.channelNames;
  out.push_back(md);
  return out;
}

std::shared_ptr<ImageXYZC>
FileReaderND2::loadFromFile(const LoadSpec& loadSpec)
{
  std::shared_ptr<ImageXYZC> empty;
  const std::string& filepath = loadSpec.filepath;
  const std::uint32_t scene = loadSpec.scene;
  const std::uint32_t time = loadSpec.time;

  auto tStart = std::chrono::high_resolution_clock::now();

  try {
    FileStream fs(filepath);
    verifyAndGetVersion(fs);
    ChunkMap chunks = readChunkMap(fs);
    Nd2Layout L = buildLayout(fs, chunks);
    if (scene >= L.sizeS) {
      LOG_ERROR << "ND2 scene out of range";
      return empty;
    }
    if (time >= L.sizeT) {
      LOG_ERROR << "ND2 time out of range";
      return empty;
    }

    // Channel selection (from loadSpec.channels). Empty -> all.
    std::size_t nch = loadSpec.channels.empty() ? L.sizeC : loadSpec.channels.size();
    std::vector<std::uint32_t> chans;
    chans.reserve(nch);
    for (std::size_t i = 0; i < nch; ++i) {
      chans.push_back(loadSpec.channels.empty() ? static_cast<std::uint32_t>(i) : loadSpec.channels[i]);
    }

    // Allocate destination buffer (uint16 packed, channel-major Z).
    std::size_t planeBytes = static_cast<std::size_t>(L.sizeX) * L.sizeY * 2u;
    std::size_t total = planeBytes * L.sizeZ * nch;
    std::unique_ptr<std::uint8_t[]> dataPtr(new std::uint8_t[total]);
    std::memset(dataPtr.get(), 0, total);

    // Per-frame uncompressed size, used both to pre-size the decompression
    // output and to validate frames.
    const std::size_t bppSrcComp = (L.bitsPerComponent <= 8) ? 1u : 2u;
    const std::size_t framePixels = static_cast<std::size_t>(L.sizeX) * L.sizeY;
    const std::size_t expectedFrameBytes = framePixels * bppSrcComp * L.componentCount;

    // ---- Build the flat list of work units. -----------------------------
    // Each work unit reads exactly one ImageDataSeq frame chunk, and
    // scatters its components into one or more disjoint destination
    // planes. Tasks for different work units never write to the same
    // destination bytes, so the parallel loop below is data-race free.
    struct WorkUnit
    {
      std::uint64_t headerOffset;             // file offset of frame chunk header
      std::uint64_t payloadSize;              // chunk payload byte count
      std::size_t nameLen;                    // chunk name byte count (incl. trailing '!')
      std::vector<std::uint32_t> srcChannels; // source channel indices to extract
      std::vector<std::uint8_t*> dstPlanes;   // destination plane pointers (one per srcChannels entry)
    };

    auto findFrameChunk = [&](std::uint64_t fidx, WorkUnit& wu) -> bool {
      std::string name = "ImageDataSeq|" + std::to_string(fidx) + "!";
      auto it = chunks.find(name);
      if (it == chunks.end()) {
        LOG_ERROR << "ND2 missing frame chunk " << name;
        return false;
      }
      wu.headerOffset = it->second.offset;
      wu.payloadSize = it->second.size;
      wu.nameLen = it->first.size();
      return true;
    };

    std::vector<WorkUnit> workUnits;
    if (L.channelsInterleaved) {
      workUnits.reserve(L.sizeZ);
      for (std::uint32_t z = 0; z < L.sizeZ; ++z) {
        WorkUnit wu;
        if (!findFrameChunk(frameIndex(L, time, scene, 0, z), wu)) {
          return empty;
        }
        wu.srcChannels = chans;
        wu.dstPlanes.resize(nch);
        for (std::size_t outCh = 0; outCh < nch; ++outCh) {
          wu.dstPlanes[outCh] = dataPtr.get() + (outCh * L.sizeZ + z) * planeBytes;
        }
        workUnits.push_back(std::move(wu));
      }
    } else {
      workUnits.reserve(static_cast<std::size_t>(L.sizeZ) * nch);
      for (std::size_t outCh = 0; outCh < nch; ++outCh) {
        std::uint32_t srcCh = chans[outCh];
        for (std::uint32_t z = 0; z < L.sizeZ; ++z) {
          WorkUnit wu;
          if (!findFrameChunk(frameIndex(L, time, scene, srcCh, z), wu)) {
            return empty;
          }
          wu.srcChannels = { srcCh };
          wu.dstPlanes = { dataPtr.get() + (outCh * L.sizeZ + z) * planeBytes };
          workUnits.push_back(std::move(wu));
        }
      }
    }

    // ---- Worker count. --------------------------------------------------
    // Goal is IO concurrency (hide NAS round-trip latency by keeping
    // multiple in-flight requests), not CPU parallelism. Cap at a modest
    // value so we don't overwhelm the storage backend; allow override via
    // AGAVE_ND2_THREADS for benchmarking.
    unsigned hwc = std::thread::hardware_concurrency();
    if (hwc == 0) {
      hwc = 4;
    }
    unsigned numWorkers = std::min<unsigned>(8u, hwc);
    if (const char* env = std::getenv("AGAVE_ND2_THREADS")) {
      int v = std::atoi(env);
      if (v > 0) {
        numWorkers = static_cast<unsigned>(v);
      }
    }
    numWorkers = std::max<unsigned>(1u, numWorkers);
    if (workUnits.size() < numWorkers) {
      numWorkers = static_cast<unsigned>(workUnits.size());
    }

    // ---- Run workers. ---------------------------------------------------
    // Each worker owns its own FileStream (independent file handle / curl
    // easy handle) and its own scratch buffers. Work is pulled from a
    // shared atomic counter; first failure flips the abort flag so later
    // tasks short-circuit cleanly.
    std::atomic<std::size_t> nextUnit{ 0 };
    std::atomic<bool> aborted{ false };
    std::mutex errMu;
    std::string firstError;
    const std::uint64_t fileSize = fs.size();

    auto workerBody = [&](FileStream& workerFs) {
      std::vector<std::uint8_t> rawScratch;
      std::vector<std::uint8_t> bodyScratch;
      rawScratch.reserve(expectedFrameBytes + kFrameInnerHeaderSkip);
      bodyScratch.reserve(expectedFrameBytes);
      while (!aborted.load(std::memory_order_relaxed)) {
        std::size_t i = nextUnit.fetch_add(1, std::memory_order_relaxed);
        if (i >= workUnits.size()) {
          return;
        }
        const WorkUnit& wu = workUnits[i];
        try {
          readKnownChunkPayloadInto(workerFs, wu.headerOffset, wu.nameLen, wu.payloadSize, rawScratch);
          if (rawScratch.size() < kFrameInnerHeaderSkip) {
            throw std::runtime_error("ND2 frame too small");
          }
          decompressFrameBytesInto(rawScratch.data() + kFrameInnerHeaderSkip,
                                   rawScratch.size() - kFrameInnerHeaderSkip,
                                   expectedFrameBytes,
                                   bodyScratch);
          if (L.channelsInterleaved) {
            copyFrameToDestMulti(bodyScratch, L, wu.srcChannels, wu.dstPlanes);
          } else {
            copyFrameToDest(bodyScratch, L, wu.srcChannels.front(), wu.dstPlanes.front());
          }
        } catch (const std::exception& e) {
          {
            std::lock_guard<std::mutex> lk(errMu);
            if (firstError.empty()) {
              firstError = e.what();
            }
          }
          aborted.store(true, std::memory_order_relaxed);
          return;
        }
      }
    };

    if (numWorkers <= 1) {
      // Single-threaded fast path: reuse the already-open `fs`, no clones.
      workerBody(fs);
    } else {
      // Spawn N-1 worker threads, each with its own FileStream clone, and
      // run one worker on this thread. The first FileStream (`fs`) is used
      // by the calling thread.
      std::vector<std::thread> threads;
      threads.reserve(numWorkers - 1);
      // Pre-construct clones on this thread so any open failure is reported
      // synchronously rather than from inside a worker.
      std::vector<std::unique_ptr<FileStream>> clones;
      clones.reserve(numWorkers - 1);
      for (unsigned w = 1; w < numWorkers; ++w) {
        clones.emplace_back(std::make_unique<FileStream>(filepath, fileSize));
      }
      for (unsigned w = 0; w < numWorkers - 1; ++w) {
        FileStream* cs = clones[w].get();
        threads.emplace_back([&, cs]() { workerBody(*cs); });
      }
      workerBody(fs);
      for (auto& t : threads) {
        t.join();
      }
    }

    if (aborted.load()) {
      LOG_ERROR << "ND2 frame read failed: " << firstError;
      return empty;
    }

    auto* im = new ImageXYZC(L.sizeX,
                             L.sizeY,
                             L.sizeZ,
                             static_cast<uint32_t>(nch),
                             ImageXYZC::IN_MEMORY_BPP,
                             dataPtr.release(),
                             L.physicalSizeX,
                             L.physicalSizeY,
                             L.physicalSizeZ,
                             L.spatialUnits);

    std::vector<std::string> channelNames;
    channelNames.reserve(nch);
    for (auto c : chans) {
      if (c < L.channelNames.size()) {
        channelNames.push_back(L.channelNames[c]);
      } else {
        channelNames.push_back("Ch" + std::to_string(c));
      }
    }
    im->setChannelNames(channelNames);

    auto tEnd = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    double mb = static_cast<double>(total) / (1024.0 * 1024.0);
    LOG_DEBUG << "ND2 loaded '" << filepath << "' in " << ms << "ms (" << workUnits.size() << " frames, " << mb
              << " MB out, " << numWorkers << " workers, " << (ms > 0.0 ? mb / (ms / 1000.0) : 0.0) << " MB/s)";

    return std::shared_ptr<ImageXYZC>(im);
  } catch (const std::exception& e) {
    LOG_ERROR << "ND2 loadFromFile failed for '" << filepath << "': " << e.what();
    return empty;
  }
}
