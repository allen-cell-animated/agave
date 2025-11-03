#pragma once

#include "../../AppScene.h"
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <memory>
#include <vector>

class ImageXYZC;
class RenderSettings;

/**
 * Vulkan 3D image renderer.
 *
 * Vulkan implementation of the volume renderer equivalent to Image3D.
 */
class Image3DVK
{
public:
  /**
   * Create a Vulkan 3D image renderer.
   */
  explicit Image3DVK();

  /// Destructor.
  virtual ~Image3DVK();

  bool initialize(VkDevice device,
                  VkPhysicalDevice physicalDevice,
                  VkRenderPass renderPass,
                  VkCommandPool commandPool,
                  VkQueue graphicsQueue);

  void create(std::shared_ptr<ImageXYZC> img);
  void render(VkCommandBuffer commandBuffer,
              const CCamera& camera,
              const Scene* scene,
              const RenderSettings* renderSettings);
  void prepareTexture(Scene& s, bool useLinearInterpolation);
  void cleanup();

protected:
  /**
   * Set the size of the x and y dimensions.
   *
   * @param xlim the x axis limits (range).
   * @param ylim the y axis limits (range).
   */
  virtual void setSize(const glm::vec2& xlim, const glm::vec2& ylim);

private:
  VkDevice m_device;
  VkPhysicalDevice m_physicalDevice;
  VkCommandPool m_commandPool;
  VkQueue m_graphicsQueue;

  // Vulkan resources for volume rendering
  VkBuffer m_vertexBuffer;
  VkDeviceMemory m_vertexBufferMemory;
  VkBuffer m_indexBuffer;
  VkDeviceMemory m_indexBufferMemory;

  // 3D texture for volume data
  VkImage m_volumeTexture;
  VkDeviceMemory m_volumeTextureMemory;
  VkImageView m_volumeTextureView;
  VkSampler m_volumeSampler;

  // Transfer function texture
  VkImage m_transferTexture;
  VkDeviceMemory m_transferTextureMemory;
  VkImageView m_transferTextureView;
  VkSampler m_transferSampler;

  // Uniform buffers
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

  std::shared_ptr<ImageXYZC> m_image;

  // Private methods
  bool createVertexBuffer();
  bool createIndexBuffer();
  bool createUniformBuffers();
  bool createDescriptorSetLayout();
  bool createDescriptorPool();
  bool createDescriptorSets();
  bool createGraphicsPipeline(VkRenderPass renderPass);
  bool createShaderModules();
  bool createVolumeTexture(const ImageXYZC* image);
  bool createTransferTexture();
  bool createTextureSampler();

  void updateUniformBuffer(const CCamera& camera, const Scene* scene, const RenderSettings* renderSettings);
  void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
  void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, uint32_t depth);

  uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

  // Uniform buffer object structure
  struct UniformBufferObject
  {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec4 volumeScale;
    glm::vec4 rayParams;
  };
};