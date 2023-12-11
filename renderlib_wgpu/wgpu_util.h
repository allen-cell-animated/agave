#pragma once

#include "webgpu-headers/webgpu.h"
#include "wgpu.h"

void
request_adapter_callback(WGPURequestAdapterStatus status, WGPUAdapter received, const char* message, void* userdata);

void
request_device_callback(WGPURequestDeviceStatus status, WGPUDevice received, const char* message, void* userdata);

void
handle_uncaptured_error(WGPUErrorType type, char const* message, void* userdata);

void
printAdapterFeatures(WGPUAdapter adapter);

void
handle_device_lost(WGPUDeviceLostReason reason, char const* message, void* userdata);
