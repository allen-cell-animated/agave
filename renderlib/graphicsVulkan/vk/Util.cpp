#include "Util.h"
#include "Logging.h"
#include <cstring>
#include <fstream>
#include <array>

void check_vk(VkResult result, const std::string& message)
{
  if (result != VK_SUCCESS) {
    LOG_ERROR << "Vulkan error in " << message << ": " << result;
  }
}

// VulkanBuffer implementation
VulkanBuffer::VulkanBuffer()
  : m_device(VK_NULL_HANDLE)
  , m_buffer(VK_NULL_HANDLE)
  , m_bufferMemory(VK_NULL_HANDLE)
  , m_size(0)
  , m_mapped(nullptr)
{
}

VulkanBuffer::~VulkanBuffer()
{
  destroy();
}

bool VulkanBuffer::create(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize size,
                         VkBufferUsageFlags usage, VkMemoryPropertyFlags properties)
{
  m_device = device;
  m_size = size;

  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkResult result = vkCreateBuffer(device, &bufferInfo, nullptr, &m_buffer);
  if (result != VK_SUCCESS) {
    LOG_ERROR << "Failed to create buffer: " << result;
    return false;
  }

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device, m_buffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

  result = vkAllocateMemory(device, &allocInfo, nullptr, &m_bufferMemory);
  if (result != VK_SUCCESS) {
    LOG_ERROR << "Failed to allocate buffer memory: " << result;
    return false;
  }

  vkBindBufferMemory(device, m_buffer, m_bufferMemory, 0);
  return true;
}

void VulkanBuffer::destroy()
{
  if (m_device != VK_NULL_HANDLE) {
    if (m_mapped) {
      unmap();
    }
    if (m_buffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(m_device, m_buffer, nullptr);
      m_buffer = VK_NULL_HANDLE;
    }
    if (m_bufferMemory != VK_NULL_HANDLE) {
      vkFreeMemory(m_device, m_bufferMemory, nullptr);
      m_bufferMemory = VK_NULL_HANDLE;
    }
  }
  m_device = VK_NULL_HANDLE;
}

void* VulkanBuffer::map()
{
  if (m_mapped) {
    return m_mapped;
  }
  
  VkResult result = vkMapMemory(m_device, m_bufferMemory, 0, m_size, 0, &m_mapped);
  if (result != VK_SUCCESS) {
    LOG_ERROR << "Failed to map buffer memory: " << result;
    return nullptr;
  }
  return m_mapped;
}

void VulkanBuffer::unmap()
{
  if (m_mapped) {
    vkUnmapMemory(m_device, m_bufferMemory);
    m_mapped = nullptr;
  }
}

void VulkanBuffer::copyData(const void* data, VkDeviceSize size)
{
  void* mappedData = map();
  if (mappedData) {
    memcpy(mappedData, data, size);
    unmap();
  }
}

uint32_t VulkanBuffer::findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter,
                                     VkMemoryPropertyFlags properties)
{
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }

  LOG_ERROR << "Failed to find suitable memory type";
  return 0;
}

// VulkanImage implementation
VulkanImage::VulkanImage()
  : m_device(VK_NULL_HANDLE)
  , m_physicalDevice(VK_NULL_HANDLE)
  , m_image(VK_NULL_HANDLE)
  , m_imageView(VK_NULL_HANDLE)
  , m_imageMemory(VK_NULL_HANDLE)
  , m_format(VK_FORMAT_UNDEFINED)
  , m_width(0)
  , m_height(0)
  , m_depth(0)
{
}

VulkanImage::~VulkanImage()
{
  destroy();
}

bool VulkanImage::create(VkDevice device, VkPhysicalDevice physicalDevice,
                        uint32_t width, uint32_t height, uint32_t depth,
                        VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
                        VkMemoryPropertyFlags properties)
{
  m_device = device;
  m_physicalDevice = physicalDevice;
  m_width = width;
  m_height = height;
  m_depth = depth;
  m_format = format;

  VkImageCreateInfo imageInfo{};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = (depth > 1) ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
  imageInfo.extent.width = width;
  imageInfo.extent.height = height;
  imageInfo.extent.depth = depth;
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = format;
  imageInfo.tiling = tiling;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = usage;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkResult result = vkCreateImage(device, &imageInfo, nullptr, &m_image);
  if (result != VK_SUCCESS) {
    LOG_ERROR << "Failed to create image: " << result;
    return false;
  }

  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(device, m_image, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

  result = vkAllocateMemory(device, &allocInfo, nullptr, &m_imageMemory);
  if (result != VK_SUCCESS) {
    LOG_ERROR << "Failed to allocate image memory: " << result;
    return false;
  }

  vkBindImageMemory(device, m_image, m_imageMemory, 0);
  return true;
}

void VulkanImage::destroy()
{
  if (m_device != VK_NULL_HANDLE) {
    if (m_imageView != VK_NULL_HANDLE) {
      vkDestroyImageView(m_device, m_imageView, nullptr);
      m_imageView = VK_NULL_HANDLE;
    }
    if (m_image != VK_NULL_HANDLE) {
      vkDestroyImage(m_device, m_image, nullptr);
      m_image = VK_NULL_HANDLE;
    }
    if (m_imageMemory != VK_NULL_HANDLE) {
      vkFreeMemory(m_device, m_imageMemory, nullptr);
      m_imageMemory = VK_NULL_HANDLE;
    }
  }
  m_device = VK_NULL_HANDLE;
  m_physicalDevice = VK_NULL_HANDLE;
}

bool VulkanImage::createImageView(VkImageViewType viewType, VkImageAspectFlags aspectFlags)
{
  VkImageViewCreateInfo viewInfo{};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = m_image;
  viewInfo.viewType = viewType;
  viewInfo.format = m_format;
  viewInfo.subresourceRange.aspectMask = aspectFlags;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = 1;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;

  VkResult result = vkCreateImageView(m_device, &viewInfo, nullptr, &m_imageView);
  if (result != VK_SUCCESS) {
    LOG_ERROR << "Failed to create image view: " << result;
    return false;
  }
  return true;
}

uint32_t VulkanImage::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
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

// RectImage2DVK implementation
RectImage2DVK::RectImage2DVK()
  : m_device(VK_NULL_HANDLE)
  , m_vertexBuffer(VK_NULL_HANDLE)
  , m_vertexBufferMemory(VK_NULL_HANDLE)
  , m_indexBuffer(VK_NULL_HANDLE)
  , m_indexBufferMemory(VK_NULL_HANDLE)
  , m_graphicsPipeline(VK_NULL_HANDLE)
  , m_pipelineLayout(VK_NULL_HANDLE)
  , m_descriptorSetLayout(VK_NULL_HANDLE)
  , m_descriptorSet(VK_NULL_HANDLE)
  , m_textureSampler(VK_NULL_HANDLE)
{
}

RectImage2DVK::~RectImage2DVK()
{
  cleanup();
}

bool RectImage2DVK::initialize(VkDevice device, VkPhysicalDevice physicalDevice,
                              VkRenderPass renderPass, VkDescriptorPool descriptorPool)
{
  m_device = device;

  if (!createVertexBuffer(physicalDevice) ||
      !createIndexBuffer(physicalDevice) ||
      !createDescriptorSetLayout() ||
      !createGraphicsPipeline(renderPass) ||
      !createTextureSampler()) {
    return false;
  }

  LOG_INFO << "RectImage2DVK initialized";
  return true;
}

void RectImage2DVK::draw(VkCommandBuffer commandBuffer, VkImageView texture)
{
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);

  VkBuffer vertexBuffers[] = {m_vertexBuffer};
  VkDeviceSize offsets[] = {0};
  vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
  vkCmdBindIndexBuffer(commandBuffer, m_indexBuffer, 0, VK_INDEX_TYPE_UINT16);

  if (m_descriptorSet != VK_NULL_HANDLE) {
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);
  }

  vkCmdDrawIndexed(commandBuffer, 6, 1, 0, 0, 0);
}

void RectImage2DVK::cleanup()
{
  if (m_device != VK_NULL_HANDLE) {
    if (m_textureSampler != VK_NULL_HANDLE) {
      vkDestroySampler(m_device, m_textureSampler, nullptr);
      m_textureSampler = VK_NULL_HANDLE;
    }
    if (m_graphicsPipeline != VK_NULL_HANDLE) {
      vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
      m_graphicsPipeline = VK_NULL_HANDLE;
    }
    if (m_pipelineLayout != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
      m_pipelineLayout = VK_NULL_HANDLE;
    }
    if (m_descriptorSetLayout != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
      m_descriptorSetLayout = VK_NULL_HANDLE;
    }
    if (m_vertexBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(m_device, m_vertexBuffer, nullptr);
      vkFreeMemory(m_device, m_vertexBufferMemory, nullptr);
    }
    if (m_indexBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(m_device, m_indexBuffer, nullptr);
      vkFreeMemory(m_device, m_indexBufferMemory, nullptr);
    }
  }
}

bool RectImage2DVK::createVertexBuffer(VkPhysicalDevice physicalDevice)
{
  struct Vertex {
    float pos[2];
    float texCoord[2];
  };

  std::array<Vertex, 4> vertices = {{
    {{-1.0f, -1.0f}, {0.0f, 0.0f}},
    {{ 1.0f, -1.0f}, {1.0f, 0.0f}},
    {{ 1.0f,  1.0f}, {1.0f, 1.0f}},
    {{-1.0f,  1.0f}, {0.0f, 1.0f}}
  }};

  VkDeviceSize bufferSize = sizeof(vertices);

  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = bufferSize;
  bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(m_device, &bufferInfo, nullptr, &m_vertexBuffer) != VK_SUCCESS) {
    LOG_ERROR << "Failed to create vertex buffer";
    return false;
  }

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(m_device, m_vertexBuffer, &memRequirements);

  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

  uint32_t memoryTypeIndex = 0;
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((memRequirements.memoryTypeBits & (1 << i)) &&
        (memProperties.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
      memoryTypeIndex = i;
      break;
    }
  }

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = memoryTypeIndex;

  if (vkAllocateMemory(m_device, &allocInfo, nullptr, &m_vertexBufferMemory) != VK_SUCCESS) {
    LOG_ERROR << "Failed to allocate vertex buffer memory";
    return false;
  }

  vkBindBufferMemory(m_device, m_vertexBuffer, m_vertexBufferMemory, 0);

  void* data;
  vkMapMemory(m_device, m_vertexBufferMemory, 0, bufferSize, 0, &data);
  memcpy(data, vertices.data(), bufferSize);
  vkUnmapMemory(m_device, m_vertexBufferMemory);

  return true;
}

bool RectImage2DVK::createIndexBuffer(VkPhysicalDevice physicalDevice)
{
  std::array<uint16_t, 6> indices = {0, 1, 2, 2, 3, 0};
  VkDeviceSize bufferSize = sizeof(indices);

  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = bufferSize;
  bufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(m_device, &bufferInfo, nullptr, &m_indexBuffer) != VK_SUCCESS) {
    LOG_ERROR << "Failed to create index buffer";
    return false;
  }

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(m_device, m_indexBuffer, &memRequirements);

  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

  uint32_t memoryTypeIndex = 0;
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((memRequirements.memoryTypeBits & (1 << i)) &&
        (memProperties.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
      memoryTypeIndex = i;
      break;
    }
  }

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = memoryTypeIndex;

  if (vkAllocateMemory(m_device, &allocInfo, nullptr, &m_indexBufferMemory) != VK_SUCCESS) {
    LOG_ERROR << "Failed to allocate index buffer memory";
    return false;
  }

  vkBindBufferMemory(m_device, m_indexBuffer, m_indexBufferMemory, 0);

  void* data;
  vkMapMemory(m_device, m_indexBufferMemory, 0, bufferSize, 0, &data);
  memcpy(data, indices.data(), bufferSize);
  vkUnmapMemory(m_device, m_indexBufferMemory);

  return true;
}

bool RectImage2DVK::createDescriptorSetLayout()
{
  VkDescriptorSetLayoutBinding samplerLayoutBinding{};
  samplerLayoutBinding.binding = 0;
  samplerLayoutBinding.descriptorCount = 1;
  samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  samplerLayoutBinding.pImmutableSamplers = nullptr;
  samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = 1;
  layoutInfo.pBindings = &samplerLayoutBinding;

  if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descriptorSetLayout) != VK_SUCCESS) {
    LOG_ERROR << "Failed to create descriptor set layout";
    return false;
  }

  return true;
}

bool RectImage2DVK::createGraphicsPipeline(VkRenderPass renderPass)
{
  // TODO: Load compiled SPIR-V shaders
  // For now, create a minimal pipeline
  
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &m_descriptorSetLayout;

  if (vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
    LOG_ERROR << "Failed to create pipeline layout";
    return false;
  }

  // TODO: Complete pipeline creation with shaders
  LOG_WARNING << "RectImage2DVK pipeline creation incomplete - needs SPIR-V shaders";
  return true;
}

bool RectImage2DVK::createTextureSampler()
{
  VkSamplerCreateInfo samplerInfo{};
  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerInfo.anisotropyEnable = VK_FALSE;
  samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  samplerInfo.unnormalizedCoordinates = VK_FALSE;
  samplerInfo.compareEnable = VK_FALSE;
  samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

  if (vkCreateSampler(m_device, &samplerInfo, nullptr, &m_textureSampler) != VK_SUCCESS) {
    LOG_ERROR << "Failed to create texture sampler";
    return false;
  }

  return true;
}