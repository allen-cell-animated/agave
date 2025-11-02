#include "RenderVKPT.h"
#include "ImageXyzcGpuVK.h"
#include "Logging.h"
#include <cstring>

RenderVKPT::RenderVKPT()
  : m_device(VK_NULL_HANDLE)
  , m_physicalDevice(VK_NULL_HANDLE)
  , m_commandPool(VK_NULL_HANDLE)
  , m_graphicsQueue(VK_NULL_HANDLE)
  , m_computeQueue(VK_NULL_HANDLE)
  , m_renderPass(VK_NULL_HANDLE)
  , m_frameNumber(0)
  , m_maxBounces(8)
  , m_samplesPerPixel(1)
  , m_stepSize(0.01f)
  , m_densityScale(1.0f)
  , m_denoisingEnabled(false)
  , m_accumulationReset(true)
  , m_volumeData(nullptr)
  , m_transferFunctionBuffer(VK_NULL_HANDLE)
  , m_transferFunctionMemory(VK_NULL_HANDLE)
  , m_quadVertexBuffer(VK_NULL_HANDLE)
  , m_quadVertexMemory(VK_NULL_HANDLE)
  , m_quadIndexBuffer(VK_NULL_HANDLE)
  , m_quadIndexMemory(VK_NULL_HANDLE)
  , m_displayPipelineLayout(VK_NULL_HANDLE)
  , m_displayPipeline(VK_NULL_HANDLE)
  , m_displayDescriptorLayout(VK_NULL_HANDLE)
  , m_displayDescriptorSet(VK_NULL_HANDLE)
  , m_width(800)
  , m_height(600)
{
  // Initialize light direction
  m_lightDirection[0] = 1.0f;
  m_lightDirection[1] = 1.0f;
  m_lightDirection[2] = 1.0f;

  // Initialize compute resources
  memset(&m_compute, 0, sizeof(m_compute));
}

RenderVKPT::~RenderVKPT()
{
  cleanup();
}

void
RenderVKPT::initialize(VkDevice device,
                       VkPhysicalDevice physicalDevice,
                       VkCommandPool commandPool,
                       VkQueue graphicsQueue,
                       VkRenderPass renderPass,
                       uint32_t width,
                       uint32_t height)
{
  RenderVK::initialize(device, physicalDevice, commandPool, graphicsQueue, renderPass, width, height);

  m_device = device;
  m_physicalDevice = physicalDevice;
  m_commandPool = commandPool;
  m_graphicsQueue = graphicsQueue;
  m_computeQueue = graphicsQueue; // Assume same queue for now
  m_renderPass = renderPass;
  m_width = width;
  m_height = height;

  if (!createComputeResources()) {
    LOG_ERROR << "Failed to create compute resources";
    return;
  }

  if (!createDisplayResources()) {
    LOG_ERROR << "Failed to create display resources";
    return;
  }

  resetAccumulation();
  LOG_INFO << "RenderVKPT initialized with dimensions " << width << "x" << height;
}

void
RenderVKPT::cleanup()
{
  if (m_device != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(m_device);

    // Cleanup compute resources
    if (m_compute.computePipeline != VK_NULL_HANDLE) {
      vkDestroyPipeline(m_device, m_compute.computePipeline, nullptr);
    }
    if (m_compute.pipelineLayout != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(m_device, m_compute.pipelineLayout, nullptr);
    }
    if (m_compute.descriptorPool != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(m_device, m_compute.descriptorPool, nullptr);
    }
    if (m_compute.descriptorSetLayout != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(m_device, m_compute.descriptorSetLayout, nullptr);
    }

    // Cleanup images
    if (m_compute.colorImageView != VK_NULL_HANDLE) {
      vkDestroyImageView(m_device, m_compute.colorImageView, nullptr);
    }
    if (m_compute.colorImage != VK_NULL_HANDLE) {
      vkDestroyImage(m_device, m_compute.colorImage, nullptr);
    }
    if (m_compute.colorImageMemory != VK_NULL_HANDLE) {
      vkFreeMemory(m_device, m_compute.colorImageMemory, nullptr);
    }

    if (m_compute.accumulationImageView != VK_NULL_HANDLE) {
      vkDestroyImageView(m_device, m_compute.accumulationImageView, nullptr);
    }
    if (m_compute.accumulationImage != VK_NULL_HANDLE) {
      vkDestroyImage(m_device, m_compute.accumulationImage, nullptr);
    }
    if (m_compute.accumulationImageMemory != VK_NULL_HANDLE) {
      vkFreeMemory(m_device, m_compute.accumulationImageMemory, nullptr);
    }

    // Cleanup buffers
    if (m_compute.uniformBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(m_device, m_compute.uniformBuffer, nullptr);
    }
    if (m_compute.uniformBufferMemory != VK_NULL_HANDLE) {
      vkFreeMemory(m_device, m_compute.uniformBufferMemory, nullptr);
    }

    if (m_transferFunctionBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(m_device, m_transferFunctionBuffer, nullptr);
    }
    if (m_transferFunctionMemory != VK_NULL_HANDLE) {
      vkFreeMemory(m_device, m_transferFunctionMemory, nullptr);
    }

    // Cleanup display resources
    if (m_displayPipeline != VK_NULL_HANDLE) {
      vkDestroyPipeline(m_device, m_displayPipeline, nullptr);
    }
    if (m_displayPipelineLayout != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(m_device, m_displayPipelineLayout, nullptr);
    }
    if (m_displayDescriptorLayout != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(m_device, m_displayDescriptorLayout, nullptr);
    }

    if (m_quadVertexBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(m_device, m_quadVertexBuffer, nullptr);
    }
    if (m_quadVertexMemory != VK_NULL_HANDLE) {
      vkFreeMemory(m_device, m_quadVertexMemory, nullptr);
    }
    if (m_quadIndexBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(m_device, m_quadIndexBuffer, nullptr);
    }
    if (m_quadIndexMemory != VK_NULL_HANDLE) {
      vkFreeMemory(m_device, m_quadIndexMemory, nullptr);
    }
  }

  memset(&m_compute, 0, sizeof(m_compute));
  RenderVK::cleanup();
}

void
RenderVKPT::render(VkCommandBuffer commandBuffer, Scene* scene, Camera* camera)
{
  // Update uniform buffer with current scene/camera data
  updateUniformBuffer(scene, camera);

  // Dispatch compute shader for path tracing
  dispatchCompute(commandBuffer);

  // Render result to screen
  renderToScreen(commandBuffer);

  // Increment frame counter for accumulation
  if (!m_accumulationReset) {
    m_frameNumber++;
  } else {
    m_frameNumber = 0;
    m_accumulationReset = false;
  }
}

void
RenderVKPT::resize(uint32_t width, uint32_t height)
{
  if (m_width != width || m_height != height) {
    m_width = width;
    m_height = height;

    // Recreate images with new dimensions
    vkDeviceWaitIdle(m_device);

    // Cleanup old images
    if (m_compute.colorImageView != VK_NULL_HANDLE) {
      vkDestroyImageView(m_device, m_compute.colorImageView, nullptr);
      m_compute.colorImageView = VK_NULL_HANDLE;
    }
    if (m_compute.colorImage != VK_NULL_HANDLE) {
      vkDestroyImage(m_device, m_compute.colorImage, nullptr);
      m_compute.colorImage = VK_NULL_HANDLE;
    }
    if (m_compute.colorImageMemory != VK_NULL_HANDLE) {
      vkFreeMemory(m_device, m_compute.colorImageMemory, nullptr);
      m_compute.colorImageMemory = VK_NULL_HANDLE;
    }

    if (m_compute.accumulationImageView != VK_NULL_HANDLE) {
      vkDestroyImageView(m_device, m_compute.accumulationImageView, nullptr);
      m_compute.accumulationImageView = VK_NULL_HANDLE;
    }
    if (m_compute.accumulationImage != VK_NULL_HANDLE) {
      vkDestroyImage(m_device, m_compute.accumulationImage, nullptr);
      m_compute.accumulationImage = VK_NULL_HANDLE;
    }
    if (m_compute.accumulationImageMemory != VK_NULL_HANDLE) {
      vkFreeMemory(m_device, m_compute.accumulationImageMemory, nullptr);
      m_compute.accumulationImageMemory = VK_NULL_HANDLE;
    }

    // Create new images
    createImages();

    // Reset accumulation
    resetAccumulation();
  }
}

void
RenderVKPT::setMaxBounces(uint32_t maxBounces)
{
  m_maxBounces = maxBounces;
  resetAccumulation();
}

void
RenderVKPT::setSamplesPerPixel(uint32_t samples)
{
  m_samplesPerPixel = samples;
  resetAccumulation();
}

void
RenderVKPT::setDenoising(bool enabled)
{
  m_denoisingEnabled = enabled;
}

void
RenderVKPT::resetAccumulation()
{
  m_accumulationReset = true;
  m_frameNumber = 0;
}

void
RenderVKPT::setVolumeData(ImageXyzcGpuVK* volumeData)
{
  m_volumeData = volumeData;
  resetAccumulation();
}

void
RenderVKPT::setTransferFunction(const std::vector<float>& transferFunction)
{
  m_transferFunction = transferFunction;

  // Update transfer function buffer
  if (!m_transferFunction.empty()) {
    createTransferFunctionBuffer();
  }

  resetAccumulation();
}

void
RenderVKPT::setStepSize(float stepSize)
{
  m_stepSize = stepSize;
  resetAccumulation();
}

void
RenderVKPT::setDensityScale(float densityScale)
{
  m_densityScale = densityScale;
  resetAccumulation();
}

void
RenderVKPT::setLightDirection(float x, float y, float z)
{
  m_lightDirection[0] = x;
  m_lightDirection[1] = y;
  m_lightDirection[2] = z;
  resetAccumulation();
}

bool
RenderVKPT::createComputeResources()
{
  if (!createUniformBuffer()) {
    LOG_ERROR << "Failed to create uniform buffer";
    return false;
  }

  if (!createImages()) {
    LOG_ERROR << "Failed to create images";
    return false;
  }

  if (!createDescriptorSets()) {
    LOG_ERROR << "Failed to create descriptor sets";
    return false;
  }

  if (!createComputePipeline()) {
    LOG_ERROR << "Failed to create compute pipeline";
    return false;
  }

  return true;
}

bool
RenderVKPT::createDisplayResources()
{
  if (!createScreenQuad()) {
    LOG_ERROR << "Failed to create screen quad";
    return false;
  }

  if (!createDisplayPipeline()) {
    LOG_ERROR << "Failed to create display pipeline";
    return false;
  }

  return true;
}

bool
RenderVKPT::createUniformBuffer()
{
  VkDeviceSize bufferSize = sizeof(PathTracingUniforms);

  return createBuffer(bufferSize,
                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      m_compute.uniformBuffer,
                      m_compute.uniformBufferMemory);
}

bool
RenderVKPT::createImages()
{
  // Create color output image (RGBA32F for HDR)
  if (!createImage(m_width,
                   m_height,
                   VK_FORMAT_R32G32B32A32_SFLOAT,
                   VK_IMAGE_TILING_OPTIMAL,
                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                   m_compute.colorImage,
                   m_compute.colorImageMemory)) {
    return false;
  }

  m_compute.colorImageView =
    createImageView(m_compute.colorImage, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);
  if (m_compute.colorImageView == VK_NULL_HANDLE) {
    return false;
  }

  // Create accumulation image for progressive rendering
  if (!createImage(m_width,
                   m_height,
                   VK_FORMAT_R32G32B32A32_SFLOAT,
                   VK_IMAGE_TILING_OPTIMAL,
                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                   m_compute.accumulationImage,
                   m_compute.accumulationImageMemory)) {
    return false;
  }

  m_compute.accumulationImageView =
    createImageView(m_compute.accumulationImage, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);
  if (m_compute.accumulationImageView == VK_NULL_HANDLE) {
    return false;
  }

  return true;
}

// Stub implementations for remaining methods
bool
RenderVKPT::createDescriptorSets()
{
  return true;
}
bool
RenderVKPT::createComputePipeline()
{
  return true;
}
bool
RenderVKPT::createDisplayPipeline()
{
  return true;
}
bool
RenderVKPT::createTransferFunctionBuffer()
{
  return true;
}
bool
RenderVKPT::createScreenQuad()
{
  return true;
}

void
RenderVKPT::updateUniformBuffer(Scene* scene, Camera* camera)
{
}
void
RenderVKPT::dispatchCompute(VkCommandBuffer commandBuffer)
{
}
void
RenderVKPT::renderToScreen(VkCommandBuffer commandBuffer)
{
}

uint32_t
RenderVKPT::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
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

bool
RenderVKPT::createBuffer(VkDeviceSize size,
                         VkBufferUsageFlags usage,
                         VkMemoryPropertyFlags properties,
                         VkBuffer& buffer,
                         VkDeviceMemory& bufferMemory)
{
  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkResult result = vkCreateBuffer(m_device, &bufferInfo, nullptr, &buffer);
  if (result != VK_SUCCESS) {
    LOG_ERROR << "Failed to create buffer: " << result;
    return false;
  }

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(m_device, buffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

  result = vkAllocateMemory(m_device, &allocInfo, nullptr, &bufferMemory);
  if (result != VK_SUCCESS) {
    LOG_ERROR << "Failed to allocate buffer memory: " << result;
    return false;
  }

  vkBindBufferMemory(m_device, buffer, bufferMemory, 0);
  return true;
}

bool
RenderVKPT::createImage(uint32_t width,
                        uint32_t height,
                        VkFormat format,
                        VkImageTiling tiling,
                        VkImageUsageFlags usage,
                        VkMemoryPropertyFlags properties,
                        VkImage& image,
                        VkDeviceMemory& imageMemory)
{
  VkImageCreateInfo imageInfo{};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.extent.width = width;
  imageInfo.extent.height = height;
  imageInfo.extent.depth = 1;
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = format;
  imageInfo.tiling = tiling;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = usage;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkResult result = vkCreateImage(m_device, &imageInfo, nullptr, &image);
  if (result != VK_SUCCESS) {
    LOG_ERROR << "Failed to create image: " << result;
    return false;
  }

  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(m_device, image, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

  result = vkAllocateMemory(m_device, &allocInfo, nullptr, &imageMemory);
  if (result != VK_SUCCESS) {
    LOG_ERROR << "Failed to allocate image memory: " << result;
    return false;
  }

  vkBindImageMemory(m_device, image, imageMemory, 0);
  return true;
}

VkImageView
RenderVKPT::createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags)
{
  VkImageViewCreateInfo viewInfo{};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = image;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = format;
  viewInfo.subresourceRange.aspectMask = aspectFlags;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = 1;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;

  VkImageView imageView;
  VkResult result = vkCreateImageView(m_device, &viewInfo, nullptr, &imageView);
  if (result != VK_SUCCESS) {
    LOG_ERROR << "Failed to create image view: " << result;
    return VK_NULL_HANDLE;
  }

  return imageView;
}

void
RenderVKPT::transitionImageLayout(VkCommandBuffer commandBuffer,
                                  VkImage image,
                                  VkFormat format,
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

  if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  } else {
    LOG_ERROR << "Unsupported layout transition";
    return;
  }

  vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}