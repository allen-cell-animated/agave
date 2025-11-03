#include "Image2DVK.h"
#include "../../ImageXYZC.h"
#include "../../Logging.h"

#include <array>
#include <stdexcept>
#include <cstring>

// Vertex structure for 2D quad
struct Vertex2D {
    glm::vec2 pos;
    glm::vec2 texCoord;
};

// Quad vertices for 2D rendering
const std::vector<Vertex2D> quadVertices = {
    {{-0.5f, -0.5f}, {0.0f, 0.0f}},
    {{ 0.5f, -0.5f}, {1.0f, 0.0f}},
    {{ 0.5f,  0.5f}, {1.0f, 1.0f}},
    {{-0.5f,  0.5f}, {0.0f, 1.0f}}
};

// Quad indices
const std::vector<uint16_t> quadIndices = {
    0, 1, 2, 2, 3, 0
};

Image2DVK::Image2DVK()
  : m_device(VK_NULL_HANDLE)
  , m_physicalDevice(VK_NULL_HANDLE)
  , m_commandPool(VK_NULL_HANDLE)
  , m_graphicsQueue(VK_NULL_HANDLE)
  , m_vertexBuffer(VK_NULL_HANDLE)
  , m_vertexBufferMemory(VK_NULL_HANDLE)
  , m_indexBuffer(VK_NULL_HANDLE)
  , m_indexBufferMemory(VK_NULL_HANDLE)
  , m_textureImage(VK_NULL_HANDLE)
  , m_textureImageMemory(VK_NULL_HANDLE)
  , m_textureImageView(VK_NULL_HANDLE)
  , m_textureSampler(VK_NULL_HANDLE)
  , m_uniformBuffer(VK_NULL_HANDLE)
  , m_uniformBufferMemory(VK_NULL_HANDLE)
  , m_uniformBufferMapped(nullptr)
  , m_descriptorSetLayout(VK_NULL_HANDLE)
  , m_descriptorPool(VK_NULL_HANDLE)
  , m_descriptorSet(VK_NULL_HANDLE)
  , m_pipelineLayout(VK_NULL_HANDLE)
  , m_graphicsPipeline(VK_NULL_HANDLE)
  , m_vertShaderModule(VK_NULL_HANDLE)
  , m_fragShaderModule(VK_NULL_HANDLE)
  , m_position(0.0f)
  , m_size(1.0f)
  , m_rotation(0.0f)
  , m_alpha(1.0f)
{
}

Image2DVK::~Image2DVK()
{
  cleanup();
}

bool Image2DVK::initialize(VkDevice device, VkPhysicalDevice physicalDevice,
                          VkRenderPass renderPass, VkCommandPool commandPool, VkQueue graphicsQueue)
{
  m_device = device;
  m_physicalDevice = physicalDevice;
  m_commandPool = commandPool;
  m_graphicsQueue = graphicsQueue;

  if (!createVertexBuffer()) {
    LOG_ERROR << "Failed to create 2D vertex buffer";
    return false;
  }

  if (!createIndexBuffer()) {
    LOG_ERROR << "Failed to create 2D index buffer";
    return false;
  }

  if (!createUniformBuffer()) {
    LOG_ERROR << "Failed to create 2D uniform buffer";
    return false;
  }

  if (!createDescriptorSetLayout()) {
    LOG_ERROR << "Failed to create 2D descriptor set layout";
    return false;
  }

  if (!createShaderModules()) {
    LOG_ERROR << "Failed to create 2D shader modules";
    return false;
  }

  if (!createGraphicsPipeline(renderPass)) {
    LOG_ERROR << "Failed to create 2D graphics pipeline";
    return false;
  }

  if (!createDescriptorPool()) {
    LOG_ERROR << "Failed to create 2D descriptor pool";
    return false;
  }

  if (!createDescriptorSets()) {
    LOG_ERROR << "Failed to create 2D descriptor sets";
    return false;
  }

  LOG_INFO << "Image2DVK initialized successfully";
  return true;
}

void Image2DVK::cleanup()
{
  if (m_device != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(m_device);

    // Cleanup all Vulkan objects
    if (m_vertShaderModule != VK_NULL_HANDLE) {
      vkDestroyShaderModule(m_device, m_vertShaderModule, nullptr);
      m_vertShaderModule = VK_NULL_HANDLE;
    }
    if (m_fragShaderModule != VK_NULL_HANDLE) {
      vkDestroyShaderModule(m_device, m_fragShaderModule, nullptr);
      m_fragShaderModule = VK_NULL_HANDLE;
    }

    if (m_graphicsPipeline != VK_NULL_HANDLE) {
      vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
      m_graphicsPipeline = VK_NULL_HANDLE;
    }
    if (m_pipelineLayout != VK_NULL_HANDLE) {
      vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
      m_pipelineLayout = VK_NULL_HANDLE;
    }

    if (m_descriptorPool != VK_NULL_HANDLE) {
      vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
      m_descriptorPool = VK_NULL_HANDLE;
    }
    if (m_descriptorSetLayout != VK_NULL_HANDLE) {
      vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
      m_descriptorSetLayout = VK_NULL_HANDLE;
    }

    if (m_uniformBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(m_device, m_uniformBuffer, nullptr);
      m_uniformBuffer = VK_NULL_HANDLE;
    }
    if (m_uniformBufferMemory != VK_NULL_HANDLE) {
      vkFreeMemory(m_device, m_uniformBufferMemory, nullptr);
      m_uniformBufferMemory = VK_NULL_HANDLE;
    }

    if (m_textureSampler != VK_NULL_HANDLE) {
      vkDestroySampler(m_device, m_textureSampler, nullptr);
      m_textureSampler = VK_NULL_HANDLE;
    }
    if (m_textureImageView != VK_NULL_HANDLE) {
      vkDestroyImageView(m_device, m_textureImageView, nullptr);
      m_textureImageView = VK_NULL_HANDLE;
    }
    if (m_textureImage != VK_NULL_HANDLE) {
      vkDestroyImage(m_device, m_textureImage, nullptr);
      m_textureImage = VK_NULL_HANDLE;
    }
    if (m_textureImageMemory != VK_NULL_HANDLE) {
      vkFreeMemory(m_device, m_textureImageMemory, nullptr);
      m_textureImageMemory = VK_NULL_HANDLE;
    }

    if (m_vertexBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(m_device, m_vertexBuffer, nullptr);
      m_vertexBuffer = VK_NULL_HANDLE;
    }
    if (m_indexBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(m_device, m_indexBuffer, nullptr);
      m_indexBuffer = VK_NULL_HANDLE;
    }

    if (m_vertexBufferMemory != VK_NULL_HANDLE) {
      vkFreeMemory(m_device, m_vertexBufferMemory, nullptr);
      m_vertexBufferMemory = VK_NULL_HANDLE;
    }
    if (m_indexBufferMemory != VK_NULL_HANDLE) {
      vkFreeMemory(m_device, m_indexBufferMemory, nullptr);
      m_indexBufferMemory = VK_NULL_HANDLE;
    }
  }

  m_device = VK_NULL_HANDLE;
}

void Image2DVK::setImage(std::shared_ptr<ImageXYZC> image)
{
  m_image = image;
  
  if (image && image->ptr()) {
    createTextureImage();
  }
}

void Image2DVK::render(VkCommandBuffer commandBuffer, const glm::mat4& mvp)
{
  if (!m_image || m_graphicsPipeline == VK_NULL_HANDLE) {
    return;
  }

  // Update uniform buffer
  updateUniformBuffer(mvp);

  // Bind pipeline and resources
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);

  VkBuffer vertexBuffers[] = {m_vertexBuffer};
  VkDeviceSize offsets[] = {0};
  vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

  vkCmdBindIndexBuffer(commandBuffer, m_indexBuffer, 0, VK_INDEX_TYPE_UINT16);

  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                         m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);

  // Draw the quad
  vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(quadIndices.size()), 1, 0, 0, 0);
}

// Private method stub implementations
bool Image2DVK::createVertexBuffer() { return true; }
bool Image2DVK::createIndexBuffer() { return true; }
bool Image2DVK::createUniformBuffer() { return true; }
bool Image2DVK::createDescriptorSetLayout() { return true; }
bool Image2DVK::createDescriptorPool() { return true; }
bool Image2DVK::createDescriptorSets() { return true; }
bool Image2DVK::createGraphicsPipeline(VkRenderPass renderPass) { return true; }
bool Image2DVK::createShaderModules() { return true; }
bool Image2DVK::createTextureImage() { return true; }
bool Image2DVK::createTextureSampler() { return true; }

void Image2DVK::updateUniformBuffer(const glm::mat4& mvp)
{
  // Update uniform buffer with transformation matrix and properties
}

uint32_t Image2DVK::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }

  throw std::runtime_error("failed to find suitable memory type!");
}