#pragma once

#include <vulkan/vulkan.h>
#include <memory>

class ImageXYZC;

// Base struct for Vulkan GPU image resources
struct ImageGpuVK
{
public:
  ImageGpuVK() = default;
  virtual ~ImageGpuVK() = default;

  // Pure virtual interface that must be implemented by concrete classes
  virtual void allocGpuInterleaved(VkDevice device,
                                   VkPhysicalDevice physicalDevice,
                                   VkCommandPool commandPool,
                                   VkQueue transferQueue,
                                   const ImageXYZC* image) = 0;
  virtual void deallocGpu() = 0;

  // Common properties
  virtual uint32_t getWidth() const = 0;
  virtual uint32_t getHeight() const = 0;
  virtual uint32_t getDepth() const = 0;
  virtual uint32_t getChannels() const = 0;

  // Vulkan resource access
  virtual VkImage getVolumeTexture() const = 0;
  virtual VkImageView getVolumeTextureView() const = 0;
  virtual VkSampler getVolumeSampler() const = 0;
  virtual VkImage getChannelTexture(uint32_t channel) const = 0;
  virtual VkImageView getChannelTextureView(uint32_t channel) const = 0;
};