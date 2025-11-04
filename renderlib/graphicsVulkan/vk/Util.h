#pragma once

#include <vulkan/vulkan.h>
#include "glm.h"
#include "Logging.h"
#include <string>
#include <vector>

class Scene;

/**
 * Check Vulkan result.
 *
 * @param result the Vulkan result code
 * @param message the message to log on error.
 */
extern void
check_vk(VkResult result, const std::string& message);

/**
 * Vulkan buffer wrapper
 */
class VulkanBuffer
{
public:
  VulkanBuffer();
  ~VulkanBuffer();

  bool create(VkDevice device,
              VkPhysicalDevice physicalDevice,
              VkDeviceSize size,
              VkBufferUsageFlags usage,
              VkMemoryPropertyFlags properties);
  void destroy();

  void* map();
  void unmap();
  void copyData(const void* data, VkDeviceSize size);

  VkBuffer getBuffer() const { return m_buffer; }
  VkDeviceMemory getMemory() const { return m_bufferMemory; }
  VkDeviceSize getSize() const { return m_size; }

private:
  VkDevice m_device;
  VkBuffer m_buffer;
  VkDeviceMemory m_bufferMemory;
  VkDeviceSize m_size;
  void* m_mapped;

  uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);
};

/**
 * Vulkan image wrapper
 */
class VulkanImage
{
public:
  VulkanImage();
  ~VulkanImage();

  bool create(VkDevice device,
              VkPhysicalDevice physicalDevice,
              uint32_t width,
              uint32_t height,
              uint32_t depth,
              VkFormat format,
              VkImageTiling tiling,
              VkImageUsageFlags usage,
              VkMemoryPropertyFlags properties);
  void destroy();

  bool createImageView(VkImageViewType viewType, VkImageAspectFlags aspectFlags);

  VkImage getImage() const { return m_image; }
  VkImageView getImageView() const { return m_imageView; }
  VkDeviceMemory getMemory() const { return m_imageMemory; }

private:
  VkDevice m_device;
  VkPhysicalDevice m_physicalDevice;
  VkImage m_image;
  VkImageView m_imageView;
  VkDeviceMemory m_imageMemory;
  VkFormat m_format;

  uint32_t m_width;
  uint32_t m_height;
  uint32_t m_depth;
};

/**
 * Vulkan equivalent of RectImage2D
 */
class RectImage2DVK
{
public:
  RectImage2DVK();
  ~RectImage2DVK();

  bool initialize(VkDevice device,
                  VkPhysicalDevice physicalDevice,
                  VkRenderPass renderPass,
                  VkDescriptorPool descriptorPool);
  void draw(VkCommandBuffer commandBuffer, VkImageView texture);
  void cleanup();

private:
  VkDevice m_device;
  VkBuffer m_vertexBuffer;
  VkDeviceMemory m_vertexBufferMemory;
  VkBuffer m_indexBuffer;
  VkDeviceMemory m_indexBufferMemory;
  VkPipeline m_graphicsPipeline;
  VkPipelineLayout m_pipelineLayout;
  VkDescriptorSetLayout m_descriptorSetLayout;
  VkDescriptorSet m_descriptorSet;
  VkSampler m_textureSampler;

  bool createVertexBuffer(VkPhysicalDevice physicalDevice);
  bool createIndexBuffer(VkPhysicalDevice physicalDevice);
  bool createDescriptorSetLayout();
  bool createGraphicsPipeline(VkRenderPass renderPass);
  bool createTextureSampler();
};

/**
 * Vulkan equivalent of BoundingBoxDrawable
 */
class BoundingBoxDrawableVK
{
public:
  BoundingBoxDrawableVK();
  ~BoundingBoxDrawableVK();

  bool initialize(VkDevice device, VkPhysicalDevice physicalDevice, VkRenderPass renderPass);
  void drawLines(VkCommandBuffer commandBuffer, const glm::mat4& transform, const glm::vec4& color);
  void drawTickMarks(VkCommandBuffer commandBuffer, const glm::mat4& transform, const glm::vec4& color);
  void drawFaces(VkCommandBuffer commandBuffer, const glm::mat4& transform, const glm::vec4& color);
  void updateTickMarks(const glm::vec3& scale, float maxPhysicalDim);
  void cleanup();

private:
  VkDevice m_device;
  VulkanBuffer m_vertexBuffer;
  VulkanBuffer m_tickMarkBuffer;
  VulkanBuffer m_lineIndexBuffer;
  VulkanBuffer m_faceIndexBuffer;

  VkPipeline m_graphicsPipeline;
  VkPipelineLayout m_pipelineLayout;
  VkDescriptorSetLayout m_descriptorSetLayout;

  size_t m_numTickMarkFloats;
  size_t m_numLineElements;
  size_t m_numFaceElements;

  bool createBuffers(VkPhysicalDevice physicalDevice);
  bool createGraphicsPipeline(VkRenderPass renderPass);
};

/**
 * Vulkan timer utility
 */
class VulkanTimer
{
public:
  VulkanTimer();
  ~VulkanTimer();

  bool initialize(VkDevice device, VkPhysicalDevice physicalDevice);
  void startTimer(VkCommandBuffer commandBuffer);
  float stopTimer(VkCommandBuffer commandBuffer);
  float getElapsedTime();
  void cleanup();

private:
  VkDevice m_device;
  VkQueryPool m_queryPool;
  bool m_started;
  float m_timestampPeriod;
};

/**
 * Vulkan framebuffer object
 */
class VulkanFramebufferObject
{
public:
  VulkanFramebufferObject(VkDevice device,
                          VkPhysicalDevice physicalDevice,
                          int width,
                          int height,
                          VkFormat colorFormat);
  ~VulkanFramebufferObject();

  void bind(VkCommandBuffer commandBuffer);
  void release();
  int width() const { return m_width; }
  int height() const { return m_height; }

  // Copy framebuffer to CPU memory
  void toImage(void* pixels);

private:
  VkDevice m_device;
  VkFramebuffer m_framebuffer;
  VulkanImage m_colorImage;
  VulkanImage m_depthImage;
  VkRenderPass m_renderPass;
  int m_width;
  int m_height;
};

/**
 * Vulkan shader module wrapper
 */
class VulkanShader
{
public:
  VulkanShader(VkDevice device);
  ~VulkanShader();

  bool loadFromFile(const std::string& filename);
  bool loadFromSource(const std::string& source, const std::string& entryPoint = "main");

  VkShaderModule getShaderModule() const { return m_shaderModule; }
  const std::string& getEntryPoint() const { return m_entryPoint; }

private:
  VkDevice m_device;
  VkShaderModule m_shaderModule;
  std::string m_entryPoint;

  bool createShaderModule(const std::vector<char>& code);
};

/**
 * Vulkan pipeline builder utility
 */
class VulkanPipelineBuilder
{
public:
  VulkanPipelineBuilder(VkDevice device);

  VulkanPipelineBuilder& setVertexShader(VkShaderModule shaderModule, const std::string& entryPoint = "main");
  VulkanPipelineBuilder& setFragmentShader(VkShaderModule shaderModule, const std::string& entryPoint = "main");
  VulkanPipelineBuilder& setVertexInputBinding(uint32_t binding,
                                               uint32_t stride,
                                               VkVertexInputRate inputRate = VK_VERTEX_INPUT_RATE_VERTEX);
  VulkanPipelineBuilder& addVertexAttribute(uint32_t location, uint32_t binding, VkFormat format, uint32_t offset);
  VulkanPipelineBuilder& setPrimitiveTopology(VkPrimitiveTopology topology);
  VulkanPipelineBuilder& setViewport(float x, float y, float width, float height);
  VulkanPipelineBuilder& setScissor(int32_t x, int32_t y, uint32_t width, uint32_t height);
  VulkanPipelineBuilder& setCullMode(VkCullModeFlags cullMode);
  VulkanPipelineBuilder& setDepthTest(bool enable, bool write = true, VkCompareOp compareOp = VK_COMPARE_OP_LESS);
  VulkanPipelineBuilder& setBlending(bool enable);

  VkPipeline build(VkRenderPass renderPass, VkPipelineLayout pipelineLayout);

private:
  VkDevice m_device;
  std::vector<VkPipelineShaderStageCreateInfo> m_shaderStages;
  VkPipelineVertexInputStateCreateInfo m_vertexInputInfo;
  VkPipelineInputAssemblyStateCreateInfo m_inputAssembly;
  VkPipelineViewportStateCreateInfo m_viewportState;
  VkPipelineRasterizationStateCreateInfo m_rasterizer;
  VkPipelineMultisampleStateCreateInfo m_multisampling;
  VkPipelineDepthStencilStateCreateInfo m_depthStencil;
  VkPipelineColorBlendStateCreateInfo m_colorBlending;

  std::vector<VkVertexInputBindingDescription> m_bindingDescriptions;
  std::vector<VkVertexInputAttributeDescription> m_attributeDescriptions;
  VkViewport m_viewport;
  VkRect2D m_scissor;
  VkPipelineColorBlendAttachmentState m_colorBlendAttachment;
};