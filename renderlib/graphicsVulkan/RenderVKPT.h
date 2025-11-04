#pragma once

#include "IVulkanRenderWindow.h"
#include "AppScene.h"
#include "Status.h"
#include "Timing.h"

#include <vulkan/vulkan.h>
#include <vector>
#include <memory>
#include <chrono>

class ImageXyzcGpuVK;
class RenderSettings;
class VulkanDevice;
class VulkanSwapchain;
class VulkanCommandBuffer;

class RenderVKPT : public IVulkanRenderWindow
{
public:
  static const std::string TYPE_NAME;

  RenderVKPT(RenderSettings* rs);
  virtual ~RenderVKPT();

  // Override IVulkanRenderWindow methods
  virtual void initialize(uint32_t w, uint32_t h) override;
  virtual void render(const CCamera& camera) override;
  virtual void renderTo(const CCamera& camera, VulkanFramebufferObject* fbo) override;
  virtual void resize(uint32_t w, uint32_t h) override;
  virtual void getSize(uint32_t& w, uint32_t& h) override
  {
    w = m_w;
    h = m_h;
  }
  virtual void cleanUpResources() override;

  virtual std::shared_ptr<CStatus> getStatusInterface() override { return m_status; }
  virtual RenderSettings& renderSettings() override;
  virtual Scene* scene() override;
  virtual void setScene(Scene* s) override;

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

  ImageXyzcGpuVK* getImage() const { return m_image3d; }

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

  // Core Vulkan objects (like RenderVK)
  VkInstance m_instance;
  VkPhysicalDevice m_physicalDevice;
  VkDevice m_device;
  VkQueue m_graphicsQueue;
  VkQueue m_presentQueue;
  VkQueue m_computeQueue;
  VkSurfaceKHR m_surface;
  VkRenderPass m_renderPass;
  VkCommandPool m_commandPool;

  // Core member variables (like RenderVK)
  ImageXyzcGpuVK* m_image3d;
  RenderSettings* m_renderSettings;
  Scene* m_scene;
  std::shared_ptr<CStatus> m_status;
  Timing m_timingRender;
  std::chrono::time_point<std::chrono::high_resolution_clock> mStartTime;
  uint32_t m_w, m_h;
  uint32_t m_currentFrame;

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

  void updateUniformBuffer(const CCamera& camera);
  void dispatchCompute(VkCommandBuffer commandBuffer);
  void renderToScreen(VkCommandBuffer commandBuffer);

  // Vulkan initialization methods (like RenderVK)
  bool initVulkan();
  bool createLogicalDevice();
  VkCommandBuffer beginSingleTimeCommands();
  void endSingleTimeCommands(VkCommandBuffer commandBuffer);

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