#pragma once

#include <vulkan/vulkan.h>
#include <memory>

class ImageXYZC;

class ImageXyzcGpuVK
{
public:
  ImageXyzcGpuVK();
  ~ImageXyzcGpuVK();

  void allocGpu(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue transferQueue);
  void deallocGpu();

  void allocGpuInterleaved(VkDevice device,
                           VkPhysicalDevice physicalDevice,
                           VkCommandPool commandPool,
                           VkQueue transferQueue,
                           const ImageXYZC* image);

  // Get the 3D texture for volume rendering
  VkImage getVolumeTexture() const { return m_volumeTexture; }
  VkImageView getVolumeTextureView() const { return m_volumeTextureView; }
  VkSampler getVolumeSampler() const { return m_volumeSampler; }

  // Get individual channel textures
  VkImage getChannelTexture(uint32_t channel) const;
  VkImageView getChannelTextureView(uint32_t channel) const;

  bool isAllocated() const { return m_volumeTexture != VK_NULL_HANDLE; }

  // Update texture data
  void updateVolumeData(VkCommandBuffer commandBuffer, const void* data, size_t dataSize);

private:
  VkDevice m_device;
  VkPhysicalDevice m_physicalDevice;
  VkCommandPool m_commandPool;
  VkQueue m_transferQueue;

  // Volume texture (interleaved channels)
  VkImage m_volumeTexture;
  VkDeviceMemory m_volumeTextureMemory;
  VkImageView m_volumeTextureView;
  VkSampler m_volumeSampler;

  // Individual channel textures
  struct ChannelTexture
  {
    VkImage image;
    VkDeviceMemory memory;
    VkImageView view;
  };
  std::vector<ChannelTexture> m_channelTextures;

  // Staging buffer for texture uploads
  VkBuffer m_stagingBuffer;
  VkDeviceMemory m_stagingBufferMemory;

  // Image properties
  uint32_t m_width;
  uint32_t m_height;
  uint32_t m_depth;
  uint32_t m_channels;

  // Helper methods
  bool createVolumeTexture(uint32_t width, uint32_t height, uint32_t depth, uint32_t channels);
  bool createChannelTextures(uint32_t width, uint32_t height, uint32_t depth, uint32_t channels);
  bool createStagingBuffer(size_t size);
  void transitionImageLayout(VkCommandBuffer commandBuffer,
                             VkImage image,
                             VkImageLayout oldLayout,
                             VkImageLayout newLayout);
  void copyBufferToImage(VkCommandBuffer commandBuffer,
                         VkBuffer buffer,
                         VkImage image,
                         uint32_t width,
                         uint32_t height,
                         uint32_t depth);
  uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
};