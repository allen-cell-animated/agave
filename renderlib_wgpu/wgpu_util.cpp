#include "wgpu_util.h"

#include "../renderlib/Logging.h"

void
request_adapter_callback(WGPURequestAdapterStatus status, WGPUAdapter received, const char* message, void* userdata)
{
  if (status == WGPURequestAdapterStatus_Success) {
    LOG_INFO << "Got WebGPU adapter";
  } else {
    LOG_INFO << "Could not get WebGPU adapter";
  }
  if (message) {
    LOG_INFO << message;
  }
  *(WGPUAdapter*)userdata = received;
}

void
request_device_callback(WGPURequestDeviceStatus status, WGPUDevice received, const char* message, void* userdata)
{
  if (status == WGPURequestDeviceStatus_Success) {
    LOG_INFO << "Got WebGPU device";
  } else {
    LOG_INFO << "Could not get WebGPU adapter";
  }
  if (message) {
    LOG_INFO << message;
  }
  *(WGPUDevice*)userdata = received;
}

void
handle_uncaptured_error(WGPUErrorType type, char const* message, void* userdata)
{
  std::string s;
  switch (type) {
    case WGPUErrorType_NoError:
      s = "NoError";
      break;
    case WGPUErrorType_Validation:
      s = "Validation";
      break;
    case WGPUErrorType_OutOfMemory:
      s = "OutOfMemory";
      break;
    case WGPUErrorType_Internal:
      s = "Internal";
      break;
    case WGPUErrorType_Unknown:
      s = "Unknown";
      break;
    case WGPUErrorType_DeviceLost:
      s = "DeviceLost";
      break;
    default:
      s = "Unknown";
      break;
  }
  // UNUSED(userdata);

  LOG_INFO << "UNCAPTURED ERROR " << s << " (" << type << "): " << message;
}

void
printAdapterFeatures(WGPUAdapter adapter)
{
  std::vector<WGPUFeatureName> features;

  // Call the function a first time with a null return address, just to get
  // the entry count.
  size_t count = wgpuAdapterEnumerateFeatures(adapter, nullptr);

  // Allocate memory (could be a new, or a malloc() if this were a C program)
  features.resize(count);

  // Call the function a second time, with a non-null return address
  wgpuAdapterEnumerateFeatures(adapter, features.data());

  LOG_INFO << "Adapter features:";
  for (uint32_t f : features) {
    std::string s;
    switch (f) {
      case WGPUFeatureName_Undefined:
        s = "Undefined";
        break;
      case WGPUFeatureName_DepthClipControl:
        s = "DepthClipControl";
        break;
      case WGPUFeatureName_Depth32FloatStencil8:
        s = "Depth32FloatStencil8";
        break;
      case WGPUFeatureName_TimestampQuery:
        s = "TimestampQuery";
        break;
      case WGPUFeatureName_TextureCompressionBC:
        s = "TextureCompressionBC";
        break;
      case WGPUFeatureName_TextureCompressionETC2:
        s = "TextureCompressionETC2";
        break;
      case WGPUFeatureName_TextureCompressionASTC:
        s = "TextureCompressionASTC";
        break;
      case WGPUFeatureName_IndirectFirstInstance:
        s = "IndirectFirstInstance";
        break;
      case WGPUFeatureName_ShaderF16:
        s = "ShaderF16";
        break;
      case WGPUFeatureName_RG11B10UfloatRenderable:
        s = "RG11B10UfloatRenderable";
        break;
      case WGPUFeatureName_BGRA8UnormStorage:
        s = "BGRA8UnormStorage";
        break;
      case WGPUFeatureName_Float32Filterable:
        s = "Float32Filterable";
        break;
      case WGPUNativeFeature_PushConstants:
        s = "PushConstants";
        break;
      case WGPUNativeFeature_TextureAdapterSpecificFormatFeatures:
        s = "TextureAdapterSpecificFormatFeatures";
        break;
      case WGPUNativeFeature_MultiDrawIndirect:
        s = "MultiDrawIndirect";
        break;
      case WGPUNativeFeature_MultiDrawIndirectCount:
        s = "MultiDrawIndirectCount";
        break;
      case WGPUNativeFeature_VertexWritableStorage:
        s = "VertexWritableStorage";
        break;
      case WGPUNativeFeature_TextureBindingArray:
        s = "TextureBindingArray";
        break;
      case WGPUNativeFeature_SampledTextureAndStorageBufferArrayNonUniformIndexing:
        s = "SampledTextureAndStorageBufferArrayNonUniformIndexing";
        break;
      case WGPUNativeFeature_PipelineStatisticsQuery:
        s = "PipelineStatisticsQuery";
        break;
      default:
        s = "Unknown";
        break;
    }
    LOG_INFO << " + " << s << " (" << f << ")";
  }
}

void
handle_device_lost(WGPUDeviceLostReason reason, char const* message, void* userdata)
{
  LOG_INFO << "DEVICE LOST (" << reason << "): " << message;
  // UNUSED(userdata);
}
