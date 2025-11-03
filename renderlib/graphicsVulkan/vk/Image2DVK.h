#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <memory>
#include <vector>

class ImageXYZC;

/**
 * Vulkan 2D image renderer.
 * 
 * Renders 2D images and textures using Vulkan.
 */
class Image2DVK
{
public:
  Image2DVK();
  ~Image2DVK();

  bool initialize(VkDevice device, VkPhysicalDevice physicalDevice, 
                 VkRenderPass renderPass, VkCommandPool commandPool, VkQueue graphicsQueue);
  
  void cleanup();
  
  void setImage(std::shared_ptr<ImageXYZC> image);
  void render(VkCommandBuffer commandBuffer, const glm::mat4& mvp);

  // Image properties
  void setPosition(const glm::vec2& position) { m_position = position; }
  void setSize(const glm::vec2& size) { m_size = size; }
  void setRotation(float rotation) { m_rotation = rotation; }
  void setAlpha(float alpha) { m_alpha = alpha; }

private:
  VkDevice m_device;
  VkPhysicalDevice m_physicalDevice;
  VkCommandPool m_commandPool;
  VkQueue m_graphicsQueue;

  // Quad geometry for 2D rendering
  VkBuffer m_vertexBuffer;
  VkDeviceMemory m_vertexBufferMemory;
  VkBuffer m_indexBuffer;
  VkDeviceMemory m_indexBufferMemory;

  // Texture resources
  VkImage m_textureImage;
  VkDeviceMemory m_textureImageMemory;
  VkImageView m_textureImageView;
  VkSampler m_textureSampler;

  // Uniform buffer
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

  // Transform properties
  glm::vec2 m_position;
  glm::vec2 m_size;
  float m_rotation;
  float m_alpha;

  std::shared_ptr<ImageXYZC> m_image;

  // Private methods
  bool createVertexBuffer();
  bool createIndexBuffer();
  bool createUniformBuffer();
  bool createDescriptorSetLayout();
  bool createDescriptorPool();
  bool createDescriptorSets();
  bool createGraphicsPipeline(VkRenderPass renderPass);
  bool createShaderModules();
  bool createTextureImage();
  bool createTextureSampler();

  void updateUniformBuffer(const glm::mat4& mvp);
  uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

  // Uniform buffer structure
  struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec4 color;
  };
};