#pragma once

#include <memory>
#include <string>

#include "assimp/Importer.hpp"

class CBoundingBox;
class ImageXYZC;

class FileReader
{
public:
	FileReader();
	virtual ~FileReader();

	std::shared_ptr<ImageXYZC> loadOMETiff_4D(const std::string& filepath);
	
	Assimp::Importer* loadAsset(const char* path, CBoundingBox* bb);

};

