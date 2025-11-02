#include "FramebufferVK.h"
#include "vk/Util.h"
#include "Logging.h"

FramebufferVK::FramebufferVK()
  : m_device(VK_NULL_HANDLE)
  , m_physicalDevice(VK_NULL_HANDLE)
  , m_framebuffer(VK_NULL_HANDLE)
  , m_renderPass(VK_NULL_HANDLE)
  , m_width(0)
  , m_height(0)
  , m_layers(1)
{
}

FramebufferVK::~FramebufferVK()
{
  destroy();
}

bool FramebufferVK::create(VkDevice device, VkPhysicalDevice physicalDevice, VkRenderPass renderPass,
                          uint32_t width, uint32_t height, uint32_t layers)
{
  m_device = device;
  m_physicalDevice = physicalDevice;
  m_renderPass = renderPass;
  m_width = width;
  m_height = height;
  m_layers = layers;

  // Create color attachment
  m_colorImage = std::make_unique<VulkanImage>();
  if (!m_colorImage->create(device, physicalDevice, width, height, 1,
                           VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL,
                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
    LOG_ERROR << "Failed to create color attachment";
    return false;
  }

  if (!m_colorImage->createImageView(VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT)) {
    LOG_ERROR << "Failed to create color attachment image view";
    return false;
  }

  // Create depth attachment
  m_depthImage = std::make_unique<VulkanImage>();
  if (!m_depthImage->create(device, physicalDevice, width, height, 1,
                           VK_FORMAT_D32_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
                           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
    LOG_ERROR << "Failed to create depth attachment";
    return false;
  }

  if (!m_depthImage->createImageView(VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT)) {
    LOG_ERROR << "Failed to create depth attachment image view";
    return false;
  }

  // Create framebuffer
  std::vector<VkImageView> attachments = {
    m_colorImage->getImageView(),
    m_depthImage->getImageView()
  };

  VkFramebufferCreateInfo framebufferInfo{};
  framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebufferInfo.renderPass = renderPass;
  framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
  framebufferInfo.pAttachments = attachments.data();
  framebufferInfo.width = width;
  framebufferInfo.height = height;
  framebufferInfo.layers = layers;

  VkResult result = vkCreateFramebuffer(device, &framebufferInfo, nullptr, &m_framebuffer);
  if (result != VK_SUCCESS) {
    LOG_ERROR << "Failed to create framebuffer: " << result;
    return false;
  }

  LOG_INFO << "Created Vulkan framebuffer " << width << "x" << height;
  return true;
}

void FramebufferVK::destroy()
{
  if (m_device != VK_NULL_HANDLE) {
    if (m_framebuffer != VK_NULL_HANDLE) {
      vkDestroyFramebuffer(m_device, m_framebuffer, nullptr);
      m_framebuffer = VK_NULL_HANDLE;
    }
  }

  m_colorImage.reset();
  m_depthImage.reset();
  
  m_device = VK_NULL_HANDLE;
  m_physicalDevice = VK_NULL_HANDLE;
}

void FramebufferVK::bind(VkCommandBuffer commandBuffer)
{
  VkRenderPassBeginInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfo.renderPass = m_renderPass;
  renderPassInfo.framebuffer = m_framebuffer;
  renderPassInfo.renderArea.offset = {0, 0};
  renderPassInfo.renderArea.extent = {m_width, m_height};

  std::vector<VkClearValue> clearValues(2);
  clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
  clearValues[1].depthStencil = {1.0f, 0};

  renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
  renderPassInfo.pClearValues = clearValues.data();

  vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
}

void FramebufferVK::clear(VkCommandBuffer commandBuffer, float r, float g, float b, float a)
{
  VkClearColorValue clearColor = {{r, g, b, a}};
  VkImageSubresourceRange range = {};
  range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  range.baseMipLevel = 0;
  range.levelCount = 1;
  range.baseArrayLayer = 0;
  range.layerCount = 1;

  vkCmdClearColorImage(commandBuffer, m_colorImage->getImage(), 
                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, &clearColor, 1, &range);
}

VkImageView FramebufferVK::getColorImageView() const
{
  return m_colorImage ? m_colorImage->getImageView() : VK_NULL_HANDLE;
}

VkImageView FramebufferVK::getDepthImageView() const
{
  return m_depthImage ? m_depthImage->getImageView() : VK_NULL_HANDLE;
}

bool FramebufferVK::copyToMemory(void* data, VkFormat format)
{
  // TODO: Implement framebuffer readback
  LOG_WARNING << "FramebufferVK::copyToMemory not yet implemented";
  return false;
}

uint32_t FramebufferVK::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
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