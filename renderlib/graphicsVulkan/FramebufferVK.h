#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <memory>

class VulkanImage;

class FramebufferVK
{
public:
  FramebufferVK();
  ~FramebufferVK();

  bool create(VkDevice device, VkPhysicalDevice physicalDevice, VkRenderPass renderPass,
              uint32_t width, uint32_t height, uint32_t layers = 1);
  void destroy();

  void bind(VkCommandBuffer commandBuffer);
  void clear(VkCommandBuffer commandBuffer, float r = 0.0f, float g = 0.0f, float b = 0.0f, float a = 1.0f);

  VkFramebuffer getFramebuffer() const { return m_framebuffer; }
  VkImageView getColorImageView() const;
  VkImageView getDepthImageView() const;
  
  uint32_t getWidth() const { return m_width; }
  uint32_t getHeight() const { return m_height; }

  // Copy framebuffer contents to CPU memory
  bool copyToMemory(void* data, VkFormat format = VK_FORMAT_R8G8B8A8_UNORM);

private:
  VkDevice m_device;
  VkPhysicalDevice m_physicalDevice;
  VkFramebuffer m_framebuffer;
  VkRenderPass m_renderPass;
  
  std::unique_ptr<VulkanImage> m_colorImage;
  std::unique_ptr<VulkanImage> m_depthImage;
  
  uint32_t m_width;
  uint32_t m_height;
  uint32_t m_layers;

  uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
};