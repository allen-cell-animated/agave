#pragma once

#include <memory>
#include <string>

class ImageXYZC;

class FileReader
{
public:
	FileReader();
	virtual ~FileReader();

	std::shared_ptr<ImageXYZC> loadOMETiff_4D(const std::string& filepath);
	
};

