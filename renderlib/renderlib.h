#pragma once

#include <string>
#include <map>
#include <memory>

struct ImageCuda;

class renderlib {
public:
	static int initialize();
	static void cleanup();

private:
	static std::map<std::string, std::shared_ptr<ImageCuda>> sCudaImageCache;
};