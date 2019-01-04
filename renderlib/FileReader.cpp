#include "FileReader.h"

#include "ImageXYZC.h"
#include "Logging.h"
#include "cudarndr/BoundingBox.h"

#include <QDomDocument>
#include <QElapsedTimer>
#include <QString>
#include <QtDebug>

#include <tiff.h>
#include <tiffio.h>

#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"

#include <map>

std::map<std::string, std::shared_ptr<ImageXYZC>> FileReader::sPreloadedImageCache;

FileReader::FileReader() {}

FileReader::~FileReader() {}

uint32_t
requireUint32Attr(QDomElement& el, const QString& attr, uint32_t defaultVal)
{
  QString attrval = el.attribute(attr);
  bool ok;
  uint32_t retval = attrval.toUInt(&ok);
  if (!ok) {
    retval = defaultVal;
  }
  return retval;
}
float
requireFloatAttr(QDomElement& el, const QString& attr, float defaultVal)
{
  QString attrval = el.attribute(attr);
  bool ok;
  float retval = attrval.toFloat(&ok);
  if (!ok) {
    retval = defaultVal;
  }
  return retval;
}

void
FileReader::getZCT(uint32_t i,
                   QString dimensionOrder,
                   uint32_t sizeZ,
                   uint32_t sizeC,
                   uint32_t sizeT,
                   uint32_t& z,
                   uint32_t& c,
                   uint32_t& t)
{
  // assume t = 0 for everything.
  assert(sizeT == 1);
  QString order = dimensionOrder.remove("XY");
  t = 0;
  if (order == "CTZ") {
    c = i % (sizeC);
    z = i / sizeC;
  } else if (order == "CZT") {
    c = i % (sizeC);
    z = i / sizeC;
  } else if (order == "ZCT") {
    c = i / (sizeZ);
    z = i % sizeZ;
  } else if (order == "ZTC") {
    c = i / (sizeZ);
    z = i % sizeZ;
  } else if (order == "TCZ") {
    c = i % (sizeC);
    z = i / sizeC;
  } else if (order == "TZC") {
    c = i / (sizeZ);
    z = i % sizeZ;
  }
}

std::shared_ptr<ImageXYZC>
FileReader::loadOMETiff_4D(const std::string& filepath, bool addToCache)
{
  // check cache first of all.
  auto cached = sPreloadedImageCache.find(filepath);
  if (cached != sPreloadedImageCache.end()) {
    return cached->second;
  }

  QElapsedTimer twhole;
  twhole.start();

  QElapsedTimer timer;
  timer.start();

  // Loads tiff file
  TIFF* tiff = TIFFOpen(filepath.c_str(), "r");
  if (!tiff) {
    QString msg = "Failed to open TIFF: '" + QString(filepath.c_str()) + "'";
    LOG_ERROR << msg.toStdString();
    // throw new Exception(NULL, msg, this, __FUNCTION__, __LINE__);
    return std::shared_ptr<ImageXYZC>();
  }

  char* omexmlstr = nullptr;
  // ome-xml is in ImageDescription of first IFD in the file.
  if (TIFFGetField(tiff, TIFFTAG_IMAGEDESCRIPTION, &omexmlstr) != 1) {
    QString msg = "Failed to read imagedescription of TIFF: '" + QString(filepath.c_str()) + "'";
    LOG_ERROR << msg.toStdString();

    // throw new Exception(NULL, msg, this, __FUNCTION__, __LINE__);
  }

  // Temporary variables
  uint32 width, height;
  //  tsize_t scanlength;

  // Read dimensions of image
  if (TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width) != 1) {
    QString msg = "Failed to read width of TIFF: '" + QString(filepath.c_str()) + "'";
    LOG_ERROR << msg.toStdString();
    // throw new Exception(NULL, msg, this, __FUNCTION__, __LINE__);
  }
  if (TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height) != 1) {
    QString msg = "Failed to read height of TIFF: '" + QString(filepath.c_str()) + "'";
    LOG_ERROR << msg.toStdString();
    // throw new Exception(NULL, msg, this, __FUNCTION__, __LINE__);
  }

  uint32_t bpp = 0;
  if (TIFFGetField(tiff, TIFFTAG_BITSPERSAMPLE, &bpp) != 1) {
    QString msg = "Failed to read bpp of TIFF: '" + QString(filepath.c_str()) + "'";
    LOG_ERROR << msg.toStdString();
    // throw new Exception(NULL, msg, this, __FUNCTION__, __LINE__);
  }

  uint32_t sizeT = 1;
  uint32_t sizeX = width;
  uint32_t sizeY = height;
  uint32_t sizeZ = 1;
  uint32_t sizeC = 1;
  float physicalSizeX = 1.0f;
  float physicalSizeY = 1.0f;
  float physicalSizeZ = 1.0f;
  std::vector<QString> channelNames;
  QString dimensionOrder = "XYCZT";

  // check for plain tiff with ImageJ imagedescription:
  QString qomexmlstr(omexmlstr);
  if (qomexmlstr.startsWith("ImageJ=")) {
    // "ImageJ=\nhyperstack=true\nimages=7900\nchannels=1\nslices=50\nframes=158"
    QStringList sl = qomexmlstr.split('\n');
    QRegExp reChannels("channels=(\\w+)");
    QRegExp reSlices("slices=(\\w+)");
    int ich = sl.indexOf(reChannels);
    if (ich == -1) {
      QString msg = "Failed to read number of channels of ImageJ TIFF: '" + QString(filepath.c_str()) + "'";
      LOG_WARNING << msg.toStdString();
    }

    int isl = sl.indexOf(reSlices);
    if (isl == -1) {
      QString msg = "Failed to read number of slices of ImageJ TIFF: '" + QString(filepath.c_str()) + "'";
      LOG_WARNING << msg.toStdString();
    }

    // get the n channels and n slices:
    if (ich != -1) {
      int pos = reChannels.indexIn(sl.at(ich));
      if (pos > -1) {
        QString value = reChannels.cap(1); // "189"
        sizeC = value.toInt();
      }
    }
    if (isl != -1) {
      int pos = reSlices.indexIn(sl.at(isl));
      if (pos > -1) {
        QString value = reSlices.cap(1); // "189"
        sizeZ = value.toInt();
      }
    }
    for (uint32_t i = 0; i < sizeC; ++i) {
      channelNames.push_back(QString::number(i));
    }
  } else if (qomexmlstr.startsWith("{\"shape\":")) {
    // expect a 4d shape array of C,Z,Y,X
    int firstBracket = qomexmlstr.indexOf('[');
    int lastBracket = qomexmlstr.lastIndexOf(']');
    QString shape = qomexmlstr.mid(firstBracket + 1, lastBracket - firstBracket - 1);
    LOG_INFO << shape.toStdString();
    QStringList shapelist = shape.split(',');
    assert(shapelist.size() == 4);
    if (shapelist.size() != 4) {
      QString msg = "Expected shape to be 4D TIFF: '" + QString(filepath.c_str()) + "'";
      LOG_ERROR << msg.toStdString();
    }
    dimensionOrder = "XYZCT";
    sizeX = shapelist[3].toInt();
    sizeY = shapelist[2].toInt();
    sizeZ = shapelist[1].toInt();
    sizeC = shapelist[0].toInt();
    for (uint32_t i = 0; i < sizeC; ++i) {
      channelNames.push_back(QString("%1").arg(i));
    }

  } else {
    // convert c to xml doc.  if this fails then we don't have an ome tif.
    QDomDocument omexml;
    bool ok = omexml.setContent(qomexmlstr);
    if (!ok) {
      QString msg = "Bad ome xml content";
      LOG_ERROR << msg.toStdString();
      // throw new Exception(NULL, msg, this, __FUNCTION__, __LINE__);
    }

    // extract some necessary info from the xml:
    QDomElement pixelsEl = omexml.elementsByTagName("Pixels").at(0).toElement();
    if (pixelsEl.isNull()) {
      QString msg = "No <Pixels> element in ome xml";
      LOG_ERROR << msg.toStdString();
    }

    // skipping "complex", "double-complex", and "bit".
    std::map<std::string, uint32_t> mapPixelTypeBPP = { { "uint8", 8 },  { "uint16", 16 }, { "uint32", 32 },
                                                        { "int8", 8 },   { "int16", 16 },  { "int32", 32 },
                                                        { "float", 32 }, { "double", 64 } };

    QString pixelType = pixelsEl.attribute("PixelType", "uint16").toLower();
    bpp = mapPixelTypeBPP[pixelType.toStdString()];
    sizeX = requireUint32Attr(pixelsEl, "SizeX", 0);
    sizeY = requireUint32Attr(pixelsEl, "SizeY", 0);
    sizeZ = requireUint32Attr(pixelsEl, "SizeZ", 0);
    sizeC = requireUint32Attr(pixelsEl, "SizeC", 0);
    sizeT = requireUint32Attr(pixelsEl, "SizeT", 0);
    // one of : "XYZCT", "XYZTC","XYCTZ","XYCZT","XYTCZ","XYTZC"
    dimensionOrder = pixelsEl.attribute("DimensionOrder", dimensionOrder);
    physicalSizeX = requireFloatAttr(pixelsEl, "PhysicalSizeX", 1.0f);
    physicalSizeY = requireFloatAttr(pixelsEl, "PhysicalSizeY", 1.0f);
    physicalSizeZ = requireFloatAttr(pixelsEl, "PhysicalSizeZ", 1.0f);
    QString physicalSizeXunit = pixelsEl.attribute("PhysicalSizeXUnit", "");
    QString physicalSizeYunit = pixelsEl.attribute("PhysicalSizeYUnit", "");
    QString physicalSizeZunit = pixelsEl.attribute("PhysicalSizeZUnit", "");

    // find channel names
    QDomNodeList channels = omexml.elementsByTagName("Channel");
    for (int i = 0; i < channels.length(); ++i) {
      QDomNode dn = channels.at(i);
      QDomElement chel = dn.toElement();
      QString chid = chel.attribute("ID");
      QString chname = chel.attribute("Name");
      if (!chname.isEmpty()) {
        channelNames.push_back(chname);
      } else if (!chid.isEmpty()) {
        channelNames.push_back(chid);
      } else {
        channelNames.push_back(QString("%1").arg(i));
      }
    }
  }

  assert(sizeX == width);
  assert(sizeY == height);

  // allocate the destination buffer!!!!
  assert(sizeC >= 1);
  assert(sizeX >= 1);
  assert(sizeY >= 1);
  assert(sizeZ >= 1);
  size_t planesize = sizeX * sizeY * bpp / 8;
  uint8_t* data = new uint8_t[planesize * sizeZ * sizeC];
  memset(data, 0, planesize * sizeZ * sizeC);

  uint8_t* destptr = data;

  // use dimensionOrderZCT to determine whether Z or C comes first!
  // assume XY are not transposed.
  QString dimensionOrderZCT = dimensionOrder.remove("XY");

  if (TIFFIsTiled(tiff)) {
    tsize_t tilesize = TIFFTileSize(tiff);
    uint32 ntiles = TIFFNumberOfTiles(tiff);
    assert(ntiles == 1);
    // assuming ntiles == 1 for all IFDs
    tdata_t buf = _TIFFmalloc(tilesize);
    for (uint32_t i = 0; i < sizeZ * sizeC; ++i) {
      int setdirok = TIFFSetDirectory(tiff, i);
      if (setdirok == 0) {
        LOG_ERROR << "Bad tiff directory specified: " << (i);
      }
      int readtileok = TIFFReadEncodedTile(tiff, 0, buf, tilesize);
      if (readtileok < 0) {
        LOG_ERROR << "Error reading tiff tile";
      }
      // copy buf into data.
      uint32_t t = 0;
      uint32_t z = 0;
      uint32_t c = 0;
      getZCT(i, dimensionOrderZCT, sizeZ, sizeC, sizeT, z, c, t);
      uint32_t planeindexinbuffer = c * sizeZ + z;
      destptr = data + (planesize * (planeindexinbuffer));
      memcpy(destptr, buf, readtileok);
    }
    _TIFFfree(buf);
  } else {
    // stripped.
    // Number of bytes in a decoded scanline
    tsize_t striplength = TIFFStripSize(tiff);
    tdata_t buf = _TIFFmalloc(striplength);

    for (uint32_t i = 0; i < sizeZ * sizeC; ++i) {
      uint32_t planeindexintiff = i;
      int setdirok = TIFFSetDirectory(tiff, planeindexintiff);
      if (setdirok == 0) {
        LOG_ERROR << "Bad tiff directory specified: " << (i);
      }
      // ensure channels are coalesced (transposing from xycz to xyzc)
      uint32_t t = 0;
      uint32_t z = 0;
      uint32_t c = 0;
      getZCT(i, dimensionOrderZCT, sizeZ, sizeC, sizeT, z, c, t);
      uint32_t planeindexinbuffer = c * sizeZ + z;
      destptr = data + (planesize * (planeindexinbuffer));
      uint32 nstrips = TIFFNumberOfStrips(tiff);
      for (tstrip_t strip = 0; strip < nstrips; strip++) {
        int readstripok = TIFFReadEncodedStrip(tiff, strip, buf, striplength);
        if (readstripok < 0) {
          LOG_ERROR << "Error reading tiff strip";
        }

        // copy buf into data.
        memcpy(destptr, buf, readstripok);
        destptr += readstripok;
      }
    }
    _TIFFfree(buf);
  }

  TIFFClose(tiff);

  LOG_DEBUG << "TIFF loaded in " << timer.elapsed() << "ms";

  timer.start();
  ImageXYZC* im =
    new ImageXYZC(sizeX, sizeY, sizeZ, sizeC, uint32_t(bpp), data, physicalSizeX, physicalSizeY, physicalSizeZ);
  LOG_DEBUG << "ImageXYZC prepared in " << timer.elapsed() << "ms";

  im->setChannelNames(channelNames);

  LOG_DEBUG << "Loaded " << filepath << " in " << twhole.elapsed() << "ms";

  std::shared_ptr<ImageXYZC> sharedImage(im);
  if (addToCache) {
    sPreloadedImageCache[filepath] = sharedImage;
  }
  return sharedImage;
}

Assimp::Importer*
FileReader::loadAsset(const char* path, CBoundingBox* bb)
{
  QElapsedTimer t;
  t.start();

  Assimp::Importer* importer = new Assimp::Importer;

  const aiScene* scene =
    importer->ReadFile(path,
                       aiProcess_Triangulate
                         //| aiProcess_JoinIdenticalVertices
                         | aiProcess_SortByPType | aiProcess_ValidateDataStructure | aiProcess_SplitLargeMeshes |
                         aiProcess_FixInfacingNormals | aiProcess_GenSmoothNormals);
  if (scene) {
    // getBoundingBox(&scene_min, &scene_max);
    // scene_center.x = (scene_min.x + scene_max.x) / 2.0f;
    // scene_center.y = (scene_min.y + scene_max.y) / 2.0f;
    // scene_center.z = (scene_min.z + scene_max.z) / 2.0f;

    // float3 optixMin = { scene_min.x, scene_min.y, scene_min.z };
    // float3 optixMax = { scene_max.x, scene_max.y, scene_max.z };
    // aabb.set(optixMin, optixMax);

    unsigned int numVerts = 0;
    unsigned int numFaces = 0;

    if (scene->mNumMeshes > 0) {
      printf("Number of meshes: %d\n", scene->mNumMeshes);

      // get the running total number of vertices & faces for all meshes
      for (unsigned int i = 0; i < scene->mNumMeshes; i++) {
        numVerts += scene->mMeshes[i]->mNumVertices;
        numFaces += scene->mMeshes[i]->mNumFaces;
      }
      printf("Found %d Vertices and %d Faces\n", numVerts, numFaces);

      for (unsigned int m = 0; m < scene->mNumMeshes; m++) {
        aiMesh* mesh = scene->mMeshes[m];
        if (!mesh->HasPositions()) {
          throw std::runtime_error("Mesh contains zero vertex positions");
        }
        if (!mesh->HasNormals()) {
          throw std::runtime_error("Mesh contains zero vertex normals");
        }

        printf("Mesh #%d\n\tNumVertices: %d\n\tNumFaces: %d\n", m, mesh->mNumVertices, mesh->mNumFaces);

        // add points
        for (unsigned int i = 0u; i < mesh->mNumVertices; i++) {
          aiVector3D pos = mesh->mVertices[i];
          aiVector3D norm = mesh->mNormals[i];

          *bb += glm::vec3(pos.x, pos.y, pos.z);
        }
      }

      printf("BBOX: X:(%f,%f)  Y:(%f,%f)  Z:(%f,%f)\n",
             bb->GetMinP().x,
             bb->GetMaxP().x,
             bb->GetMinP().y,
             bb->GetMaxP().y,
             bb->GetMinP().z,
             bb->GetMaxP().z);
    }
  }
  LOG_DEBUG << "Loaded mesh in " << t.elapsed() << "ms";
  return importer;
}
