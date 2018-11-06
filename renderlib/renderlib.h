#pragma once

#include <string>
#include <map>
#include <memory>

struct ImageCuda;
class ImageXYZC;

class renderlib {
public:
	static int initialize();
	static void clearCudaVolumeCache();
	static void cleanup();

	// usage of this cache:
	// websocketserver: 
	//   preloaded images will be cached at preload time and never deallocated
	//   images loaded by command will not be cached here.
	//   gpu cache will stay alive until shutdown
	// desktop gui:
	//   there is no preloading.
	//   gpu cache will be cleared every time a new volume is loaded.
	// if do_cache is false, caller is responsible for deallocating from gpu mem.
	// if do_cache is true, caller can be assured gpu mem will stay cached until an event that clears the cache
	// events that clear the cache: 
	// websocketserver: 
	//    shutdown
	// desktop:
	//    load new image or shutdown
	static std::shared_ptr<ImageCuda> imageAllocGPU_Cuda(std::shared_ptr<ImageXYZC> image, bool do_cache = true);
	static void imageDeallocGPU_Cuda(std::shared_ptr<ImageXYZC> image);
private:
	static std::map<std::shared_ptr<ImageXYZC>, std::shared_ptr<ImageCuda>> sCudaImageCache;
};