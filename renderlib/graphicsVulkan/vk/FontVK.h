#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <string>
#include <memory>
#include <map>

/**
 * Vulkan font rendering class.
 * 
 * Renders text using Vulkan with bitmap fonts or signed distance fields.
 */
class FontVK
{
public:
  FontVK();
  ~FontVK();

  bool initialize(VkDevice device, VkPhysicalDevice physicalDevice,
                 VkRenderPass renderPass, VkCommandPool commandPool, VkQueue graphicsQueue);
  
  void cleanup();

  // Font loading
  bool loadFont(const std::string& fontPath, float fontSize);
  
  // Text rendering
  void renderText(VkCommandBuffer commandBuffer, const std::string& text, 
                 const glm::vec2& position, const glm::vec3& color, const glm::mat4& mvp);
  
  // Font properties
  void setFontSize(float size) { m_fontSize = size; }
  float getFontSize() const { return m_fontSize; }
  
  glm::vec2 getTextSize(const std::string& text) const;

private:
  VkDevice m_device;
  VkPhysicalDevice m_physicalDevice;
  VkCommandPool m_commandPool;
  VkQueue m_graphicsQueue;

  // Font texture atlas
  VkImage m_fontTexture;
  VkDeviceMemory m_fontTextureMemory;
  VkImageView m_fontTextureView;
  VkSampler m_fontSampler;

  // Dynamic vertex buffer for text quads
  VkBuffer m_vertexBuffer;
  VkDeviceMemory m_vertexBufferMemory;
  void* m_vertexBufferMapped;
  
  // Index buffer for quads
  VkBuffer m_indexBuffer;
  VkDeviceMemory m_indexBufferMemory;

  // Uniform buffer for text properties
  VkBuffer m_uniformBuffer;
  VkDeviceMemory m_uniformBufferMemory;
  void* m_uniformBufferMapped;

  // Pipeline objects
  VkDescriptorSetLayout m_descriptorSetLayout;
  VkDescriptorPool m_descriptorPool;
  VkDescriptorSet m_descriptorSet;
  VkPipelineLayout m_pipelineLayout;
  VkPipeline m_graphicsPipeline;

  // Shader modules
  VkShaderModule m_vertShaderModule;
  VkShaderModule m_fragShaderModule;

  // Font properties
  float m_fontSize;
  std::string m_fontPath;

  // Character info structure
  struct Character {
    glm::ivec2 size;       // Size of glyph
    glm::ivec2 bearing;    // Offset from baseline to left/top of glyph
    unsigned int advance;   // Offset to advance to next glyph
    glm::vec4 texCoords;   // Texture coordinates in atlas (x, y, width, height)
  };

  std::map<char, Character> m_characters;

  // Vertex structure for text rendering
  struct TextVertex {
    glm::vec2 position;
    glm::vec2 texCoords;
  };

  // Uniform buffer structure
  struct TextUniforms {
    glm::mat4 projection;
    glm::vec3 textColor;
    float padding;
  };

  // Private methods
  bool createVertexBuffer();
  bool createIndexBuffer();
  bool createUniformBuffer();
  bool createDescriptorSetLayout();
  bool createDescriptorPool();
  bool createDescriptorSets();
  bool createGraphicsPipeline(VkRenderPass renderPass);
  bool createShaderModules();
  bool createFontTexture();
  bool createFontSampler();

  void updateVertexBuffer(const std::string& text, const glm::vec2& position);
  void updateUniformBuffer(const glm::vec3& color, const glm::mat4& mvp);
  uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
};