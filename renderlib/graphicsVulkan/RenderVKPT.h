#pragma once

#include "RenderVK.h"
#include <vulkan/vulkan.h>
#include <vector>
#include <memory>

class Scene;
class Camera;
class ImageXyzcGpuVK;

class RenderVKPT : public RenderVK
{
public:
  RenderVKPT();
  virtual ~RenderVKPT();

  // Override RenderVK methods
  virtual void initialize(VkDevice device,
                          VkPhysicalDevice physicalDevice,
                          VkCommandPool commandPool,
                          VkQueue graphicsQueue,
                          VkRenderPass renderPass,
                          uint32_t width,
                          uint32_t height) override;
  virtual void cleanup() override;
  virtual void render(VkCommandBuffer commandBuffer, Scene* scene, Camera* camera) override;
  virtual void resize(uint32_t width, uint32_t height) override;

  // Path tracing specific methods
  void setMaxBounces(uint32_t maxBounces);
  void setSamplesPerPixel(uint32_t samples);
  void setDenoising(bool enabled);
  void resetAccumulation();

  // Volume rendering
  void setVolumeData(ImageXyzcGpuVK* volumeData);
  void setTransferFunction(const std::vector<float>& transferFunction);

  // Ray tracing parameters
  void setStepSize(float stepSize);
  void setDensityScale(float densityScale);
  void setLightDirection(float x, float y, float z);

private:
  struct PathTracingUniforms
  {
    float viewMatrix[16];
    float projMatrix[16];
    float invViewMatrix[16];
    float invProjMatrix[16];
    float cameraPos[4];
    float lightDir[4];
    uint32_t frameNumber;
    uint32_t maxBounces;
    uint32_t samplesPerPixel;
    float stepSize;
    float densityScale;
    float time;
    uint32_t width;
    uint32_t height;
  };

  struct ComputeResources
  {
    VkBuffer uniformBuffer;
    VkDeviceMemory uniformBufferMemory;
    VkImage colorImage;
    VkDeviceMemory colorImageMemory;
    VkImageView colorImageView;
    VkImage accumulationImage;
    VkDeviceMemory accumulationImageMemory;
    VkImageView accumulationImageView;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    VkPipelineLayout pipelineLayout;
    VkPipeline computePipeline;
  };

  // Vulkan objects
  VkDevice m_device;
  VkPhysicalDevice m_physicalDevice;
  VkCommandPool m_commandPool;
  VkQueue m_graphicsQueue;
  VkQueue m_computeQueue;
  VkRenderPass m_renderPass;

  // Compute resources
  ComputeResources m_compute;

  // Path tracing state
  uint32_t m_frameNumber;
  uint32_t m_maxBounces;
  uint32_t m_samplesPerPixel;
  float m_stepSize;
  float m_densityScale;
  float m_lightDirection[3];
  bool m_denoisingEnabled;
  bool m_accumulationReset;

  // Volume data
  ImageXyzcGpuVK* m_volumeData;
  std::vector<float> m_transferFunction;
  VkBuffer m_transferFunctionBuffer;
  VkDeviceMemory m_transferFunctionMemory;

  // Screen quad for final display
  VkBuffer m_quadVertexBuffer;
  VkDeviceMemory m_quadVertexMemory;
  VkBuffer m_quadIndexBuffer;
  VkDeviceMemory m_quadIndexMemory;
  VkPipelineLayout m_displayPipelineLayout;
  VkPipeline m_displayPipeline;
  VkDescriptorSetLayout m_displayDescriptorLayout;
  VkDescriptorSet m_displayDescriptorSet;

  // Dimensions
  uint32_t m_width;
  uint32_t m_height;

  // Helper methods
  bool createComputeResources();
  bool createDisplayResources();
  bool createUniformBuffer();
  bool createImages();
  bool createDescriptorSets();
  bool createComputePipeline();
  bool createDisplayPipeline();
  bool createTransferFunctionBuffer();
  bool createScreenQuad();

  void updateUniformBuffer(Scene* scene, Camera* camera);
  void dispatchCompute(VkCommandBuffer commandBuffer);
  void renderToScreen(VkCommandBuffer commandBuffer);

  // Utility methods
  uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
  bool createBuffer(VkDeviceSize size,
                    VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags properties,
                    VkBuffer& buffer,
                    VkDeviceMemory& bufferMemory);
  bool createImage(uint32_t width,
                   uint32_t height,
                   VkFormat format,
                   VkImageTiling tiling,
                   VkImageUsageFlags usage,
                   VkMemoryPropertyFlags properties,
                   VkImage& image,
                   VkDeviceMemory& imageMemory);
  VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);

  void transitionImageLayout(VkCommandBuffer commandBuffer,
                             VkImage image,
                             VkFormat format,
                             VkImageLayout oldLayout,
                             VkImageLayout newLayout);
};