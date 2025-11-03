#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

/**
 * Vulkan Full Screen Quad utility.
 * 
 * Provides a simple full-screen quad for post-processing effects,
 * screen-space rendering, and final display operations.
 */
class FSQ
{
public:
  FSQ();
  ~FSQ();

  bool initialize(VkDevice device, VkPhysicalDevice physicalDevice,
                 VkCommandPool commandPool, VkQueue graphicsQueue);
  
  void cleanup();

  // Render the full-screen quad
  void render(VkCommandBuffer commandBuffer);

  // Get vertex buffer for binding
  VkBuffer getVertexBuffer() const { return m_vertexBuffer; }
  VkBuffer getIndexBuffer() const { return m_indexBuffer; }
  uint32_t getIndexCount() const { return 6; } // 2 triangles

private:
  VkDevice m_device;
  VkPhysicalDevice m_physicalDevice;

  // Quad geometry
  VkBuffer m_vertexBuffer;
  VkDeviceMemory m_vertexBufferMemory;
  VkBuffer m_indexBuffer;
  VkDeviceMemory m_indexBufferMemory;

  // Vertex structure for full-screen quad
  struct FSQVertex {
    glm::vec2 position;
    glm::vec2 texCoord;
  };

  // Static quad data
  static const FSQVertex vertices[4];
  static const uint16_t indices[6];

  // Private methods
  bool createVertexBuffer();
  bool createIndexBuffer();
  uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
};

// Static vertex data for full-screen quad
const FSQ::FSQVertex FSQ::vertices[4] = {
  {{-1.0f, -1.0f}, {0.0f, 0.0f}}, // Bottom-left
  {{ 1.0f, -1.0f}, {1.0f, 0.0f}}, // Bottom-right
  {{ 1.0f,  1.0f}, {1.0f, 1.0f}}, // Top-right
  {{-1.0f,  1.0f}, {0.0f, 1.0f}}  // Top-left
};

const uint16_t FSQ::indices[6] = {
  0, 1, 2, 2, 3, 0
};