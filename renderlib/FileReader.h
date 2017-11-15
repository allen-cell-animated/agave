#pragma once

#include <ome/files/FormatReader.h>
#include <ome/files/Types.h>

#include <memory>
#include <string>

class ImageXYZC;

class FileReader
{
public:
	FileReader(const std::string& omeSchemaDir);
	virtual ~FileReader();

	std::shared_ptr<ome::files::FormatReader> open(const std::string& filepath);
	std::shared_ptr<ImageXYZC> openToImage(const std::string& filepath);

	std::shared_ptr<ImageXYZC> loadOMETiff_4D(const std::string& filepath);
	
};

