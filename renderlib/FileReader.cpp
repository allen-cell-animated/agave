#include "FileReader.h"

#include "ImageXYZC.h"
#include "Logging.h"
#include "cudarndr/BoundingBox.h"

#include <QDomDocument>
#include <QString>
#include <QElapsedTimer>
#include <QtDebug>

#include <tiff.h>
#include <tiffio.h>

#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"

#include <map>

FileReader::FileReader()
{
}


FileReader::~FileReader()
{
}

uint32_t requireUint32Attr(QDomElement& el, const QString& attr, uint32_t defaultVal) {
	QString attrval = el.attribute(attr);
	bool ok;
	uint32_t retval = attrval.toUInt(&ok);
	if (!ok) {
		retval = defaultVal;
	}
	return retval;
}
float requireFloatAttr(QDomElement& el, const QString& attr, float defaultVal) {
	QString attrval = el.attribute(attr);
	bool ok;
	float retval = attrval.toFloat(&ok);
	if (!ok) {
		retval = defaultVal;
	}
	return retval;
}

std::shared_ptr<ImageXYZC> FileReader::loadOMETiff_4D(const std::string& filepath)
{
	QElapsedTimer timer;
	timer.start();

// Loads tiff file
  TIFF* tiff = TIFFOpen(filepath.c_str(), "r");
  if (!tiff) {
    QString msg = "Failed to open TIFF: '" + QString(filepath.c_str()) + "'";
	LOG_ERROR << msg.toStdString();
    //throw new Exception(NULL, msg, this, __FUNCTION__, __LINE__);
	return std::shared_ptr<ImageXYZC>();
  }


  char* omexmlstr = nullptr;
  // ome-xml is in ImageDescription of first IFD in the file.
  if (TIFFGetField(tiff, TIFFTAG_IMAGEDESCRIPTION, &omexmlstr) != 1) {
    QString msg = "Failed to read imagedescription of TIFF: '" + QString(filepath.c_str()) + "'";
	LOG_ERROR << msg.toStdString();

    //throw new Exception(NULL, msg, this, __FUNCTION__, __LINE__);
  }
  // convert c to xml doc.  if this fails then we don't have an ome tif.
  QDomDocument omexml;
  bool ok = omexml.setContent(QString(omexmlstr));
  if (!ok) {
    QString msg = "Bad ome xml content";
	LOG_ERROR << msg.toStdString();
	//throw new Exception(NULL, msg, this, __FUNCTION__, __LINE__);
  }

  // extract some necessary info from the xml:
  QDomElement pixelsEl = omexml.elementsByTagName("Pixels").at(0).toElement();
  if (pixelsEl.isNull()) {
	  QString msg = "No <Pixels> element in ome xml";
	  LOG_ERROR << msg.toStdString();
  }

  // skipping "complex", "double-complex", and "bit".
  std::map<std::string, uint32_t> mapPixelTypeBPP = { 
	  { "uint8", 8 },
	  { "uint16", 16 },
	  { "uint32", 32 },
	  { "int8", 8 },
	  { "int16", 16 },
	  { "int32", 32 },
	  { "float", 32 },
	  { "double", 64 } 
  };

  QString pixelType = pixelsEl.attribute("PixelType", "uint16").toLower();
  uint32_t bpp = mapPixelTypeBPP[pixelType.toStdString()];
  uint32_t sizeX = requireUint32Attr(pixelsEl, "SizeX", 0);
  uint32_t sizeY = requireUint32Attr(pixelsEl, "SizeY", 0);
  uint32_t sizeZ = requireUint32Attr(pixelsEl, "SizeZ", 0);
  uint32_t sizeC = requireUint32Attr(pixelsEl, "SizeC", 0);
  uint32_t sizeT = requireUint32Attr(pixelsEl, "SizeT", 0);
  // one of : "XYZCT", "XYZTC","XYCTZ","XYCZT","XYTCZ","XYTZC"
  QString dimensionOrder = pixelsEl.attribute("DimensionOrder", "XYCZT");
  float physicalSizeX = requireFloatAttr(pixelsEl, "PhysicalSizeX", 1.0f);
  float physicalSizeY = requireFloatAttr(pixelsEl, "PhysicalSizeY", 1.0f);
  float physicalSizeZ = requireFloatAttr(pixelsEl, "PhysicalSizeZ", 1.0f);
  QString physicalSizeXunit = pixelsEl.attribute("PhysicalSizeXUnit", "");
  QString physicalSizeYunit = pixelsEl.attribute("PhysicalSizeYUnit", "");
  QString physicalSizeZunit = pixelsEl.attribute("PhysicalSizeZUnit", "");

  // find channel names
  QDomNodeList channels = omexml.elementsByTagName("Channel");
  std::vector<QString> channelNames;
  for (int i = 0; i < channels.length(); ++i) {
	  QDomNode dn = channels.at(i);
	  QDomElement chel = dn.toElement();
	  QString chid = chel.attribute("ID");
	  QString chname = chel.attribute("Name");
	  if (!chname.isEmpty()) {
		  channelNames.push_back(chname);
	  }
	  else if (!chid.isEmpty()) {
		  channelNames.push_back(chid);
	  }
	  else {
		  channelNames.push_back(QString("%1").arg(i));
	  }
  }


  // Temporary variables
  uint32 width, height;
//  tsize_t scanlength;

  // Read dimensions of image
  if (TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width) != 1) {
    QString msg = "Failed to read width of TIFF: '" + QString(filepath.c_str()) + "'";
	LOG_ERROR << msg.toStdString();
	//throw new Exception(NULL, msg, this, __FUNCTION__, __LINE__);
  }
  if (TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height) != 1) {
    QString msg = "Failed to read height of TIFF: '" + QString(filepath.c_str()) + "'";
	LOG_ERROR << msg.toStdString();
	//throw new Exception(NULL, msg, this, __FUNCTION__, __LINE__);
  }

  assert(sizeX == width);
  assert(sizeY == height);

	// allocate the destination buffer!!!!
  assert(sizeC >= 1);
  assert(sizeX >= 1);
  assert(sizeY >= 1);
  assert(sizeZ >= 1);
  size_t planesize = sizeX*sizeY*bpp / 8;
  uint8_t* data = new uint8_t[planesize*sizeZ*sizeC];
  memset(data, 0, planesize*sizeZ*sizeC);



  uint8_t* destptr = data;

  if (TIFFIsTiled(tiff)) {
	  tsize_t tilesize = TIFFTileSize(tiff);
	  uint32 ntiles = TIFFNumberOfTiles(tiff);
	  assert(ntiles == 1);
	  // assuming ntiles == 1 for all IFDs
	  tdata_t buf = _TIFFmalloc(tilesize);
	  for (uint32_t i = 0; i < sizeC; ++i) {
		  for (uint32_t j = 0; j < sizeZ; ++j) {
			  int setdirok = TIFFSetDirectory(tiff, j + i*sizeZ);
			  if (setdirok == 0) {
				  LOG_ERROR << "Bad tiff directory specified: " << (j + i*sizeZ);
			  }
			  int readtileok = TIFFReadEncodedTile(tiff, 0, buf, tilesize);
			  if (readtileok < 0) {
				  LOG_ERROR << "Error reading tiff tile";
			  }
			  // copy buf into data.
			  memcpy(destptr, buf, tilesize);
			  destptr += tilesize;
		  }
	  }
	  _TIFFfree(buf);
  }
  else {
	  // stripped.
	  // Number of bytes in a decoded scanline
	  tsize_t striplength = TIFFStripSize(tiff);
	  tdata_t buf = _TIFFmalloc(striplength);
	  for (uint32_t i = 0; i < sizeZ; ++i) {
		  for (uint32_t j = 0; j < sizeC; ++j) {
			  uint32_t planeindexintiff = j + i*sizeC;
			  int setdirok = TIFFSetDirectory(tiff, planeindexintiff);
			  if (setdirok == 0) {
				  LOG_ERROR << "Bad tiff directory specified: " << (j + i*sizeC);
			  }
			  // ensure channels are coalesced (transposing from xycz to xyzc)
			  uint32_t planeindexinbuffer = i + j*sizeZ;
			  destptr = data + (planesize*(planeindexinbuffer));
			  uint32 nstrips = TIFFNumberOfStrips(tiff);
			  for (tstrip_t strip = 0; strip < nstrips; strip++)
			  {
				  int readstripok = TIFFReadEncodedStrip(tiff, strip, buf, striplength);
				  if (readstripok < 0) {
					  LOG_ERROR << "Error reading tiff strip";
				  }

				  // copy buf into data.
				  memcpy(destptr, buf, striplength);
				  destptr += striplength;
			  }

		  }
	  }
	  _TIFFfree(buf);

  }

  TIFFClose(tiff);	

  LOG_DEBUG << "TIFF loaded in " << timer.elapsed() << "ms";

  timer.start();
  ImageXYZC* im = new ImageXYZC(sizeX, sizeY, sizeZ, sizeC, uint32_t(bpp), data, physicalSizeX, physicalSizeY, physicalSizeZ);
  LOG_DEBUG << "ImageXYZC prepared in " << timer.elapsed() << "ms";
  
  im->setChannelNames(channelNames);

  return std::shared_ptr<ImageXYZC>(im);
}

Assimp::Importer* FileReader::loadAsset(const char* path, CBoundingBox* bb)
{
	Assimp::Importer* importer = new Assimp::Importer;

	const aiScene* scene = importer->ReadFile(
		path,
		aiProcess_Triangulate
		//| aiProcess_JoinIdenticalVertices
		| aiProcess_SortByPType
		| aiProcess_ValidateDataStructure
		| aiProcess_SplitLargeMeshes
		| aiProcess_FixInfacingNormals
		| aiProcess_GenSmoothNormals
	);
	if (scene) {
		//getBoundingBox(&scene_min, &scene_max);
		//scene_center.x = (scene_min.x + scene_max.x) / 2.0f;
		//scene_center.y = (scene_min.y + scene_max.y) / 2.0f;
		//scene_center.z = (scene_min.z + scene_max.z) / 2.0f;

		//float3 optixMin = { scene_min.x, scene_min.y, scene_min.z };
		//float3 optixMax = { scene_max.x, scene_max.y, scene_max.z };
		//aabb.set(optixMin, optixMax);

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
				aiMesh *mesh = scene->mMeshes[m];
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

			printf("BBOX: X:(%f,%f)  Y:(%f,%f)  Z:(%f,%f)\n", bb->GetMinP().x, bb->GetMaxP().x, bb->GetMinP().y, bb->GetMaxP().y, bb->GetMinP().z, bb->GetMaxP().z);

		}

	}
	return importer;
}
