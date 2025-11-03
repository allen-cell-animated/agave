#include "FontVK.h"
#include "../../Logging.h"

#include <stdexcept>
#include <cstring>
#include <map>

FontVK::FontVK()
  : m_device(VK_NULL_HANDLE)
  , m_physicalDevice(VK_NULL_HANDLE)
  , m_commandPool(VK_NULL_HANDLE)
  , m_graphicsQueue(VK_NULL_HANDLE)
  , m_fontTexture(VK_NULL_HANDLE)
  , m_fontTextureMemory(VK_NULL_HANDLE)
  , m_fontTextureView(VK_NULL_HANDLE)
  , m_fontSampler(VK_NULL_HANDLE)
  , m_vertexBuffer(VK_NULL_HANDLE)
  , m_vertexBufferMemory(VK_NULL_HANDLE)
  , m_vertexBufferMapped(nullptr)
  , m_indexBuffer(VK_NULL_HANDLE)
  , m_indexBufferMemory(VK_NULL_HANDLE)
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
  , m_fontSize(24.0f)
{
}

FontVK::~FontVK()
{
  cleanup();
}

bool FontVK::initialize(VkDevice device, VkPhysicalDevice physicalDevice,
                       VkRenderPass renderPass, VkCommandPool commandPool, VkQueue graphicsQueue)
{
  m_device = device;
  m_physicalDevice = physicalDevice;
  m_commandPool = commandPool;
  m_graphicsQueue = graphicsQueue;

  if (!createVertexBuffer()) {
    LOG_ERROR << "Failed to create font vertex buffer";
    return false;
  }

  if (!createIndexBuffer()) {
    LOG_ERROR << "Failed to create font index buffer";
    return false;
  }

  if (!createUniformBuffer()) {
    LOG_ERROR << "Failed to create font uniform buffer";
    return false;
  }

  if (!createDescriptorSetLayout()) {
    LOG_ERROR << "Failed to create font descriptor set layout";
    return false;
  }

  if (!createShaderModules()) {
    LOG_ERROR << "Failed to create font shader modules";
    return false;
  }

  if (!createGraphicsPipeline(renderPass)) {
    LOG_ERROR << "Failed to create font graphics pipeline";
    return false;
  }

  if (!createDescriptorPool()) {
    LOG_ERROR << "Failed to create font descriptor pool";
    return false;
  }

  if (!createDescriptorSets()) {
    LOG_ERROR << "Failed to create font descriptor sets";
    return false;
  }

  LOG_INFO << "FontVK initialized successfully";
  return true;
}

void FontVK::cleanup()
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

    if (m_fontSampler != VK_NULL_HANDLE) {
      vkDestroySampler(m_device, m_fontSampler, nullptr);
      m_fontSampler = VK_NULL_HANDLE;
    }
    if (m_fontTextureView != VK_NULL_HANDLE) {
      vkDestroyImageView(m_device, m_fontTextureView, nullptr);
      m_fontTextureView = VK_NULL_HANDLE;
    }
    if (m_fontTexture != VK_NULL_HANDLE) {
      vkDestroyImage(m_device, m_fontTexture, nullptr);
      m_fontTexture = VK_NULL_HANDLE;
    }
    if (m_fontTextureMemory != VK_NULL_HANDLE) {
      vkFreeMemory(m_device, m_fontTextureMemory, nullptr);
      m_fontTextureMemory = VK_NULL_HANDLE;
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

bool FontVK::loadFont(const std::string& fontPath, float fontSize)
{
  m_fontPath = fontPath;
  m_fontSize = fontSize;

  // Load font using FreeType and create texture atlas
  // This would typically involve:
  // 1. Initialize FreeType
  // 2. Load font file
  // 3. Generate glyph bitmaps
  // 4. Pack into texture atlas
  // 5. Store character metrics

  if (!createFontTexture()) {
    LOG_ERROR << "Failed to create font texture";
    return false;
  }

  if (!createFontSampler()) {
    LOG_ERROR << "Failed to create font sampler";
    return false;
  }

  LOG_INFO << "Font loaded: " << fontPath << " (size: " << fontSize << ")";
  return true;
}

void FontVK::renderText(VkCommandBuffer commandBuffer, const std::string& text,
                       const glm::vec2& position, const glm::vec3& color, const glm::mat4& mvp)
{
  if (text.empty() || m_graphicsPipeline == VK_NULL_HANDLE) {
    return;
  }

  // Update vertex buffer with text quads
  updateVertexBuffer(text, position);

  // Update uniform buffer with color and transformation
  updateUniformBuffer(color, mvp);

  // Bind pipeline and resources
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);

  VkBuffer vertexBuffers[] = {m_vertexBuffer};
  VkDeviceSize offsets[] = {0};
  vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

  vkCmdBindIndexBuffer(commandBuffer, m_indexBuffer, 0, VK_INDEX_TYPE_UINT16);

  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                         m_pipelineLayout, 0, 1, &m_descriptorSet, 0, nullptr);

  // Draw text (6 indices per character: 2 triangles per quad)
  uint32_t indexCount = static_cast<uint32_t>(text.length() * 6);
  vkCmdDrawIndexed(commandBuffer, indexCount, 1, 0, 0, 0);
}

glm::vec2 FontVK::getTextSize(const std::string& text) const
{
  glm::vec2 size(0.0f);
  
  for (char c : text) {
    auto it = m_characters.find(c);
    if (it != m_characters.end()) {
      size.x += (it->second.advance >> 6); // Convert from 64ths of pixels
      size.y = std::max(size.y, static_cast<float>(it->second.size.y));
    }
  }
  
  return size;
}

// Private method stub implementations
bool FontVK::createVertexBuffer() { return true; }
bool FontVK::createIndexBuffer() { return true; }
bool FontVK::createUniformBuffer() { return true; }
bool FontVK::createDescriptorSetLayout() { return true; }
bool FontVK::createDescriptorPool() { return true; }
bool FontVK::createDescriptorSets() { return true; }
bool FontVK::createGraphicsPipeline(VkRenderPass renderPass) { return true; }
bool FontVK::createShaderModules() { return true; }
bool FontVK::createFontTexture() { return true; }
bool FontVK::createFontSampler() { return true; }

void FontVK::updateVertexBuffer(const std::string& text, const glm::vec2& position)
{
  // Generate vertex data for text quads and update vertex buffer
}

void FontVK::updateUniformBuffer(const glm::vec3& color, const glm::mat4& mvp)
{
  TextUniforms ubo{};
  ubo.projection = mvp;
  ubo.textColor = color;

  if (m_uniformBufferMapped) {
    memcpy(m_uniformBufferMapped, &ubo, sizeof(ubo));
  }
}

uint32_t FontVK::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
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