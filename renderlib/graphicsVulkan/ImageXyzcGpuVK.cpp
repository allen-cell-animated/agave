#include "ImageXyzcGpuVK.h"
#include "ImageXYZC.h"
#include "Logging.h"
#include <cstring>

ImageXyzcGpuVK::ImageXyzcGpuVK()
  : m_device(VK_NULL_HANDLE)
  , m_physicalDevice(VK_NULL_HANDLE)
  , m_commandPool(VK_NULL_HANDLE)
  , m_transferQueue(VK_NULL_HANDLE)
  , m_volumeTexture(VK_NULL_HANDLE)
  , m_volumeTextureMemory(VK_NULL_HANDLE)
  , m_volumeTextureView(VK_NULL_HANDLE)
  , m_volumeSampler(VK_NULL_HANDLE)
  , m_stagingBuffer(VK_NULL_HANDLE)
  , m_stagingBufferMemory(VK_NULL_HANDLE)
  , m_width(0)
  , m_height(0)
  , m_depth(0)
  , m_channels(0)
{
}

ImageXyzcGpuVK::~ImageXyzcGpuVK()
{
  deallocGpu();
}

void
ImageXyzcGpuVK::allocGpu(VkDevice device,
                         VkPhysicalDevice physicalDevice,
                         VkCommandPool commandPool,
                         VkQueue transferQueue)
{
  m_device = device;
  m_physicalDevice = physicalDevice;
  m_commandPool = commandPool;
  m_transferQueue = transferQueue;
}

void
ImageXyzcGpuVK::deallocGpu()
{
  if (m_device != VK_NULL_HANDLE) {
    // Wait for device to be idle before cleanup
    vkDeviceWaitIdle(m_device);

    // Cleanup channel textures
    for (auto& channel : m_channelTextures) {
      if (channel.view != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device, channel.view, nullptr);
      }
      if (channel.image != VK_NULL_HANDLE) {
        vkDestroyImage(m_device, channel.image, nullptr);
      }
      if (channel.memory != VK_NULL_HANDLE) {
        vkFreeMemory(m_device, channel.memory, nullptr);
      }
    }
    m_channelTextures.clear();

    // Cleanup volume texture
    if (m_volumeSampler != VK_NULL_HANDLE) {
      vkDestroySampler(m_device, m_volumeSampler, nullptr);
      m_volumeSampler = VK_NULL_HANDLE;
    }
    if (m_volumeTextureView != VK_NULL_HANDLE) {
      vkDestroyImageView(m_device, m_volumeTextureView, nullptr);
      m_volumeTextureView = VK_NULL_HANDLE;
    }
    if (m_volumeTexture != VK_NULL_HANDLE) {
      vkDestroyImage(m_device, m_volumeTexture, nullptr);
      m_volumeTexture = VK_NULL_HANDLE;
    }
    if (m_volumeTextureMemory != VK_NULL_HANDLE) {
      vkFreeMemory(m_device, m_volumeTextureMemory, nullptr);
      m_volumeTextureMemory = VK_NULL_HANDLE;
    }

    // Cleanup staging buffer
    if (m_stagingBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(m_device, m_stagingBuffer, nullptr);
      m_stagingBuffer = VK_NULL_HANDLE;
    }
    if (m_stagingBufferMemory != VK_NULL_HANDLE) {
      vkFreeMemory(m_device, m_stagingBufferMemory, nullptr);
      m_stagingBufferMemory = VK_NULL_HANDLE;
    }
  }

  m_device = VK_NULL_HANDLE;
  m_physicalDevice = VK_NULL_HANDLE;
  m_commandPool = VK_NULL_HANDLE;
  m_transferQueue = VK_NULL_HANDLE;
}

void
ImageXyzcGpuVK::allocGpuInterleaved(VkDevice device,
                                    VkPhysicalDevice physicalDevice,
                                    VkCommandPool commandPool,
                                    VkQueue transferQueue,
                                    const ImageXYZC* image)
{
  allocGpu(device, physicalDevice, commandPool, transferQueue);

  if (!image) {
    LOG_ERROR << "Invalid image provided to allocGpuInterleaved";
    return;
  }

  m_width = image->sizeX();
  m_height = image->sizeY();
  m_depth = image->sizeZ();
  m_channels = image->sizeC();

  LOG_INFO << "Allocating Vulkan volume texture: " << m_width << "x" << m_height << "x" << m_depth
           << " channels: " << m_channels;

  // Create volume texture
  if (!createVolumeTexture(m_width, m_height, m_depth, m_channels)) {
    LOG_ERROR << "Failed to create volume texture";
    return;
  }

  // Create individual channel textures
  if (!createChannelTextures(m_width, m_height, m_depth, m_channels)) {
    LOG_ERROR << "Failed to create channel textures";
    return;
  }

  // Create staging buffer for data upload
  size_t dataSize = m_width * m_height * m_depth * m_channels * sizeof(uint16_t);
  if (!createStagingBuffer(dataSize)) {
    LOG_ERROR << "Failed to create staging buffer";
    return;
  }

  // Upload image data
  void* stagingData;
  VkResult result = vkMapMemory(m_device, m_stagingBufferMemory, 0, dataSize, 0, &stagingData);
  if (result != VK_SUCCESS) {
    LOG_ERROR << "Failed to map staging buffer: " << result;
    return;
  }

  // Copy interleaved data to staging buffer
  const uint16_t* imageData = image->ptr();
  memcpy(stagingData, imageData, dataSize);
  vkUnmapMemory(m_device, m_stagingBufferMemory);

  // Create command buffer for transfer operations
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = m_commandPool;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  vkAllocateCommandBuffers(m_device, &allocInfo, &commandBuffer);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(commandBuffer, &beginInfo);

  // Transition image layout and copy data
  transitionImageLayout(
    commandBuffer, m_volumeTexture, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

  copyBufferToImage(commandBuffer, m_stagingBuffer, m_volumeTexture, m_width, m_height, m_depth);

  transitionImageLayout(
    commandBuffer, m_volumeTexture, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  vkEndCommandBuffer(commandBuffer);

  // Submit command buffer
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  vkQueueSubmit(m_transferQueue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(m_transferQueue);

  vkFreeCommandBuffers(m_device, m_commandPool, 1, &commandBuffer);

  LOG_INFO << "Vulkan volume texture created and uploaded successfully";
}

bool
ImageXyzcGpuVK::createVolumeTexture(uint32_t width, uint32_t height, uint32_t depth, uint32_t channels)
{
  // Create 3D texture
  VkImageCreateInfo imageInfo{};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_3D;
  imageInfo.extent.width = width;
  imageInfo.extent.height = height;
  imageInfo.extent.depth = depth;
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = (channels == 1) ? VK_FORMAT_R16_UNORM : VK_FORMAT_R16G16B16A16_UNORM;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkResult result = vkCreateImage(m_device, &imageInfo, nullptr, &m_volumeTexture);
  if (result != VK_SUCCESS) {
    LOG_ERROR << "Failed to create volume texture: " << result;
    return false;
  }

  // Allocate memory for texture
  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(m_device, m_volumeTexture, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  result = vkAllocateMemory(m_device, &allocInfo, nullptr, &m_volumeTextureMemory);
  if (result != VK_SUCCESS) {
    LOG_ERROR << "Failed to allocate volume texture memory: " << result;
    return false;
  }

  vkBindImageMemory(m_device, m_volumeTexture, m_volumeTextureMemory, 0);

  // Create image view
  VkImageViewCreateInfo viewInfo{};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = m_volumeTexture;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
  viewInfo.format = imageInfo.format;
  viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = 1;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;

  result = vkCreateImageView(m_device, &viewInfo, nullptr, &m_volumeTextureView);
  if (result != VK_SUCCESS) {
    LOG_ERROR << "Failed to create volume texture view: " << result;
    return false;
  }

  // Create sampler
  VkSamplerCreateInfo samplerInfo{};
  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.anisotropyEnable = VK_FALSE;
  samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  samplerInfo.unnormalizedCoordinates = VK_FALSE;
  samplerInfo.compareEnable = VK_FALSE;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

  result = vkCreateSampler(m_device, &samplerInfo, nullptr, &m_volumeSampler);
  if (result != VK_SUCCESS) {
    LOG_ERROR << "Failed to create volume sampler: " << result;
    return false;
  }

  return true;
}

bool
ImageXyzcGpuVK::createChannelTextures(uint32_t width, uint32_t height, uint32_t depth, uint32_t channels)
{
  m_channelTextures.resize(channels);

  for (uint32_t c = 0; c < channels; ++c) {
    ChannelTexture& channel = m_channelTextures[c];

    // Create image
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_3D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = depth;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R16_UNORM;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult result = vkCreateImage(m_device, &imageInfo, nullptr, &channel.image);
    if (result != VK_SUCCESS) {
      LOG_ERROR << "Failed to create channel " << c << " texture: " << result;
      return false;
    }

    // Allocate memory
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(m_device, channel.image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    result = vkAllocateMemory(m_device, &allocInfo, nullptr, &channel.memory);
    if (result != VK_SUCCESS) {
      LOG_ERROR << "Failed to allocate channel " << c << " texture memory: " << result;
      return false;
    }

    vkBindImageMemory(m_device, channel.image, channel.memory, 0);

    // Create image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = channel.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
    viewInfo.format = VK_FORMAT_R16_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    result = vkCreateImageView(m_device, &viewInfo, nullptr, &channel.view);
    if (result != VK_SUCCESS) {
      LOG_ERROR << "Failed to create channel " << c << " texture view: " << result;
      return false;
    }
  }

  return true;
}

bool
ImageXyzcGpuVK::createStagingBuffer(size_t size)
{
  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkResult result = vkCreateBuffer(m_device, &bufferInfo, nullptr, &m_stagingBuffer);
  if (result != VK_SUCCESS) {
    LOG_ERROR << "Failed to create staging buffer: " << result;
    return false;
  }

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(m_device, m_stagingBuffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(
    memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  result = vkAllocateMemory(m_device, &allocInfo, nullptr, &m_stagingBufferMemory);
  if (result != VK_SUCCESS) {
    LOG_ERROR << "Failed to allocate staging buffer memory: " << result;
    return false;
  }

  vkBindBufferMemory(m_device, m_stagingBuffer, m_stagingBufferMemory, 0);
  return true;
}

void
ImageXyzcGpuVK::transitionImageLayout(VkCommandBuffer commandBuffer,
                                      VkImage image,
                                      VkImageLayout oldLayout,
                                      VkImageLayout newLayout)
{
  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = oldLayout;
  barrier.newLayout = newLayout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;

  VkPipelineStageFlags sourceStage;
  VkPipelineStageFlags destinationStage;

  if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  } else {
    LOG_ERROR << "Unsupported layout transition";
    return;
  }

  vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void
ImageXyzcGpuVK::copyBufferToImage(VkCommandBuffer commandBuffer,
                                  VkBuffer buffer,
                                  VkImage image,
                                  uint32_t width,
                                  uint32_t height,
                                  uint32_t depth)
{
  VkBufferImageCopy region{};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageOffset = { 0, 0, 0 };
  region.imageExtent = { width, height, depth };

  vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}

uint32_t
ImageXyzcGpuVK::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }

  LOG_ERROR << "Failed to find suitable memory type";
  return 0;
}

VkImage
ImageXyzcGpuVK::getChannelTexture(uint32_t channel) const
{
  if (channel < m_channelTextures.size()) {
    return m_channelTextures[channel].image;
  }
  return VK_NULL_HANDLE;
}

VkImageView
ImageXyzcGpuVK::getChannelTextureView(uint32_t channel) const
{
  if (channel < m_channelTextures.size()) {
    return m_channelTextures[channel].view;
  }
  return VK_NULL_HANDLE;
}