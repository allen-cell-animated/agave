#include "commandBuffer.h"

#include "command.h"

#include <algorithm>
#include <assert.h>
#include <vector>

#if HAVE_BYTESWAP_H
#include <byteswap.h>
#else
#define bswap_16(value) ((((value)&0xff) << 8) | ((value) >> 8))

#define bswap_32(value)                                                                                                \
  (((uint32_t)bswap_16((uint16_t)((value)&0xffff)) << 16) | (uint32_t)bswap_16((uint16_t)((value) >> 16)))

#define bswap_64(value)                                                                                                \
  (((uint64_t)bswap_32((uint32_t)((value)&0xffffffff)) << 32) | (uint64_t)bswap_32((uint32_t)((value) >> 32)))
#endif

commandBuffer::commandBuffer(size_t len, const uint8_t* buf)
  : _length(len)
  , _headPos(buf)
{}

commandBuffer::~commandBuffer() {}

CommandBufferIterator::CommandBufferIterator(commandBuffer* buf)
  : _commandBuffer(buf)
  , _currentPos(const_cast<uint8_t*>(buf->head()))
{}

//////////////////
// forward declare.
#define FWDDECL_PARSE(CMDCLASS) Command* parse##CMDCLASS(CommandBufferIterator* c);

FWDDECL_PARSE(SessionCommand);
FWDDECL_PARSE(SessionCommand);
FWDDECL_PARSE(AssetPathCommand);
FWDDECL_PARSE(LoadOmeTifCommand);
FWDDECL_PARSE(SetCameraPosCommand);
FWDDECL_PARSE(SetCameraUpCommand);
FWDDECL_PARSE(SetCameraTargetCommand);
FWDDECL_PARSE(SetCameraTargetCommand);
FWDDECL_PARSE(SetCameraApertureCommand);
FWDDECL_PARSE(SetCameraFovYCommand);
FWDDECL_PARSE(SetCameraFocalDistanceCommand);
FWDDECL_PARSE(SetCameraExposureCommand);
FWDDECL_PARSE(SetDiffuseColorCommand);
FWDDECL_PARSE(SetSpecularColorCommand);
FWDDECL_PARSE(SetEmissiveColorCommand);
FWDDECL_PARSE(SetRenderIterationsCommand);
FWDDECL_PARSE(SetStreamModeCommand);
FWDDECL_PARSE(RequestRedrawCommand);
FWDDECL_PARSE(SetResolutionCommand);
FWDDECL_PARSE(SetDensityCommand);
FWDDECL_PARSE(FrameSceneCommand);
FWDDECL_PARSE(SetGlossinessCommand);
FWDDECL_PARSE(EnableChannelCommand);
FWDDECL_PARSE(SetWindowLevelCommand);
FWDDECL_PARSE(OrbitCameraCommand);
FWDDECL_PARSE(SetSkylightTopColorCommand);
FWDDECL_PARSE(SetSkylightMiddleColorCommand);
FWDDECL_PARSE(SetSkylightBottomColorCommand);
FWDDECL_PARSE(SetLightPosCommand);
FWDDECL_PARSE(SetLightColorCommand);
FWDDECL_PARSE(SetLightSizeCommand);
FWDDECL_PARSE(SetClipRegionCommand);
FWDDECL_PARSE(SetVoxelScaleCommand);
FWDDECL_PARSE(AutoThresholdCommand);

#define CMD_CASE(CMDCLASS)                                                                                             \
  case (CMDCLASS::m_ID):                                                                                               \
    return parse##CMDCLASS(&iterator);                                                                                 \
    break;

void
commandBuffer::processBuffer()
{
  CommandBufferIterator iterator(this);
  while (!iterator.end()) {
    // new command.
    // read its int32 enum value.
    int32_t cmd = iterator.parseInt32();

    // lambda that takes our iterator and the cmd id to initialize the command object.
    Command* c = [cmd, &iterator]() -> Command* {
      try {
        switch (cmd) {
          CMD_CASE(SessionCommand);
          CMD_CASE(AssetPathCommand);
          CMD_CASE(LoadOmeTifCommand);
          CMD_CASE(SetCameraPosCommand);
          CMD_CASE(SetCameraTargetCommand);
          CMD_CASE(SetCameraUpCommand);
          CMD_CASE(SetCameraApertureCommand);
          CMD_CASE(SetCameraFovYCommand);
          CMD_CASE(SetCameraFocalDistanceCommand);
          CMD_CASE(SetCameraExposureCommand);
          CMD_CASE(SetDiffuseColorCommand);
          CMD_CASE(SetSpecularColorCommand);
          CMD_CASE(SetEmissiveColorCommand);
          CMD_CASE(SetRenderIterationsCommand);
          CMD_CASE(SetStreamModeCommand);
          CMD_CASE(RequestRedrawCommand);
          CMD_CASE(SetResolutionCommand);
          CMD_CASE(SetDensityCommand);
          CMD_CASE(FrameSceneCommand);
          CMD_CASE(SetGlossinessCommand);
          CMD_CASE(EnableChannelCommand);
          CMD_CASE(SetWindowLevelCommand);
          CMD_CASE(OrbitCameraCommand);
          CMD_CASE(SetSkylightTopColorCommand);
          CMD_CASE(SetSkylightMiddleColorCommand);
          CMD_CASE(SetSkylightBottomColorCommand);
          CMD_CASE(SetLightPosCommand);
          CMD_CASE(SetLightColorCommand);
          CMD_CASE(SetLightSizeCommand);
          CMD_CASE(SetClipRegionCommand);
          CMD_CASE(SetVoxelScaleCommand);
          CMD_CASE(AutoThresholdCommand);
          default:
            // ERROR UNRECOGNIZED COMMAND SIGNATURE.
            // PRINT OUT PREVIOUS! BAIL OUT! OR DO SOMETHING CLEVER AND CORRECT!
            return nullptr;
            break;
        }
      } catch (...) {
        // buffer error?
        return nullptr;
      }
      return nullptr;
    }();

    if (c) {
      // good! add to list of commands
      _commands.push_back(c);
    } else {
      // error! do something.
    }
  }
}

void
commandBuffer::execute(ExecutionContext* c)
{
  // let's run all the commands now
  for (auto i = _commands.begin(); i != _commands.end(); ++i) {
    (*i)->execute(c);
  }
}

bool
CommandBufferIterator::end()
{
  return (_currentPos >= _commandBuffer->head() + _commandBuffer->length());
}

int32_t
CommandBufferIterator::parseInt32()
{
  int32_t value = bswap_32(*((int32_t*)(_currentPos)));
  _currentPos += sizeof(int32_t);
  return value;
}

float
CommandBufferIterator::parseFloat32()
{
  // assuming sizeof float == sizeof int32 == 4
  // float value = (float)bswap_32(*((int32_t*)(_currentPos)));
  float value = *((float*)(_currentPos));
  _currentPos += sizeof(float);
  return value;
}

std::string
CommandBufferIterator::parseString()
{
  int32_t len = parseInt32();
  std::string s(reinterpret_cast<char const*>(_currentPos), (size_t)len);
  _currentPos += len;
  return s;
}

////////////////////////////////////////////
Command*
parseSessionCommand(CommandBufferIterator* c)
{
  SessionCommandD data;
  data.m_name = c->parseString();
  return new SessionCommand(data);
}
Command*
parseAssetPathCommand(CommandBufferIterator* c)
{
  AssetPathCommandD data;
  data.m_name = c->parseString();
  return new AssetPathCommand(data);
}
Command*
parseLoadOmeTifCommand(CommandBufferIterator* c)
{
  LoadOmeTifCommandD data;
  data.m_name = c->parseString();
  return new LoadOmeTifCommand(data);
}
Command*
parseSetCameraPosCommand(CommandBufferIterator* c)
{
  SetCameraPosCommandD data;
  data.m_x = c->parseFloat32();
  data.m_y = c->parseFloat32();
  data.m_z = c->parseFloat32();
  return new SetCameraPosCommand(data);
}
Command*
parseSetCameraUpCommand(CommandBufferIterator* c)
{
  SetCameraUpCommandD data;
  data.m_x = c->parseFloat32();
  data.m_y = c->parseFloat32();
  data.m_z = c->parseFloat32();
  return new SetCameraUpCommand(data);
}
Command*
parseSetCameraTargetCommand(CommandBufferIterator* c)
{
  SetCameraTargetCommandD data;
  data.m_x = c->parseFloat32();
  data.m_y = c->parseFloat32();
  data.m_z = c->parseFloat32();
  return new SetCameraTargetCommand(data);
}
Command*
parseSetCameraApertureCommand(CommandBufferIterator* c)
{
  SetCameraApertureCommandD data;
  data.m_x = c->parseFloat32();
  return new SetCameraApertureCommand(data);
}
Command*
parseSetCameraFovYCommand(CommandBufferIterator* c)
{
  SetCameraFovYCommandD data;
  data.m_x = c->parseFloat32();
  return new SetCameraFovYCommand(data);
}
Command*
parseSetCameraFocalDistanceCommand(CommandBufferIterator* c)
{
  SetCameraFocalDistanceCommandD data;
  data.m_x = c->parseFloat32();
  return new SetCameraFocalDistanceCommand(data);
}
Command*
parseSetCameraExposureCommand(CommandBufferIterator* c)
{
  SetCameraExposureCommandD data;
  data.m_x = c->parseFloat32();
  return new SetCameraExposureCommand(data);
}
Command*
parseSetDiffuseColorCommand(CommandBufferIterator* c)
{
  SetDiffuseColorCommandD data;
  data.m_channel = c->parseInt32();
  data.m_r = c->parseFloat32();
  data.m_g = c->parseFloat32();
  data.m_b = c->parseFloat32();
  data.m_a = c->parseFloat32();
  return new SetDiffuseColorCommand(data);
}
Command*
parseSetSpecularColorCommand(CommandBufferIterator* c)
{
  SetSpecularColorCommandD data;
  data.m_channel = c->parseInt32();
  data.m_r = c->parseFloat32();
  data.m_g = c->parseFloat32();
  data.m_b = c->parseFloat32();
  data.m_a = c->parseFloat32();
  return new SetSpecularColorCommand(data);
}
Command*
parseSetEmissiveColorCommand(CommandBufferIterator* c)
{
  SetEmissiveColorCommandD data;
  data.m_channel = c->parseInt32();
  data.m_r = c->parseFloat32();
  data.m_g = c->parseFloat32();
  data.m_b = c->parseFloat32();
  data.m_a = c->parseFloat32();
  return new SetEmissiveColorCommand(data);
}
Command*
parseSetRenderIterationsCommand(CommandBufferIterator* c)
{
  SetRenderIterationsCommandD data;
  data.m_x = c->parseInt32();
  return new SetRenderIterationsCommand(data);
}
Command*
parseSetStreamModeCommand(CommandBufferIterator* c)
{
  SetStreamModeCommandD data;
  data.m_x = c->parseInt32();
  return new SetStreamModeCommand(data);
}
Command*
parseRequestRedrawCommand(CommandBufferIterator* c)
{
  RequestRedrawCommandD data;
  return new RequestRedrawCommand(data);
}
Command*
parseSetResolutionCommand(CommandBufferIterator* c)
{
  SetResolutionCommandD data;
  data.m_x = c->parseInt32();
  data.m_y = c->parseInt32();
  return new SetResolutionCommand(data);
}
Command*
parseSetDensityCommand(CommandBufferIterator* c)
{
  SetDensityCommandD data;
  data.m_x = c->parseFloat32();
  return new SetDensityCommand(data);
}
Command*
parseFrameSceneCommand(CommandBufferIterator* c)
{
  FrameSceneCommandD data;
  return new FrameSceneCommand(data);
}
Command*
parseSetGlossinessCommand(CommandBufferIterator* c)
{
  SetGlossinessCommandD data;
  data.m_channel = c->parseInt32();
  data.m_glossiness = c->parseFloat32();
  return new SetGlossinessCommand(data);
}
Command*
parseEnableChannelCommand(CommandBufferIterator* c)
{
  EnableChannelCommandD data;
  data.m_channel = c->parseInt32();
  data.m_enabled = c->parseInt32();
  return new EnableChannelCommand(data);
}
Command*
parseSetWindowLevelCommand(CommandBufferIterator* c)
{
  SetWindowLevelCommandD data;
  data.m_channel = c->parseInt32();
  data.m_window = c->parseFloat32();
  data.m_level = c->parseFloat32();
  return new SetWindowLevelCommand(data);
}
Command*
parseOrbitCameraCommand(CommandBufferIterator* c)
{
  OrbitCameraCommandD data;
  data.m_theta = c->parseFloat32();
  data.m_phi = c->parseFloat32();
  return new OrbitCameraCommand(data);
}
Command*
parseSetSkylightTopColorCommand(CommandBufferIterator* c)
{
  SetSkylightTopColorCommandD data;
  data.m_r = c->parseFloat32();
  data.m_g = c->parseFloat32();
  data.m_b = c->parseFloat32();
  return new SetSkylightTopColorCommand(data);
}
Command*
parseSetSkylightMiddleColorCommand(CommandBufferIterator* c)
{
  SetSkylightMiddleColorCommandD data;
  data.m_r = c->parseFloat32();
  data.m_g = c->parseFloat32();
  data.m_b = c->parseFloat32();
  return new SetSkylightMiddleColorCommand(data);
}
Command*
parseSetSkylightBottomColorCommand(CommandBufferIterator* c)
{
  SetSkylightBottomColorCommandD data;
  data.m_r = c->parseFloat32();
  data.m_g = c->parseFloat32();
  data.m_b = c->parseFloat32();
  return new SetSkylightBottomColorCommand(data);
}

Command*
parseSetLightPosCommand(CommandBufferIterator* c)
{
  SetLightPosCommandD data;
  data.m_index = c->parseInt32();
  data.m_r = c->parseFloat32();
  data.m_theta = c->parseFloat32();
  data.m_phi = c->parseFloat32();
  return new SetLightPosCommand(data);
}
Command*
parseSetLightColorCommand(CommandBufferIterator* c)
{
  SetLightColorCommandD data;
  data.m_index = c->parseInt32();
  data.m_r = c->parseFloat32();
  data.m_g = c->parseFloat32();
  data.m_b = c->parseFloat32();
  return new SetLightColorCommand(data);
}
Command*
parseSetLightSizeCommand(CommandBufferIterator* c)
{
  SetLightSizeCommandD data;
  data.m_index = c->parseInt32();
  data.m_x = c->parseFloat32();
  data.m_y = c->parseFloat32();
  return new SetLightSizeCommand(data);
}

float
clamp(float x, float bottom, float top)
{
  return std::min(top, std::max(bottom, x));
}

Command*
parseSetClipRegionCommand(CommandBufferIterator* c)
{
  SetClipRegionCommandD data;
  data.m_minx = c->parseFloat32();
  data.m_minx = clamp(data.m_minx, 0.0, 1.0);
  data.m_maxx = c->parseFloat32();
  data.m_maxx = clamp(data.m_maxx, 0.0, 1.0);
  data.m_miny = c->parseFloat32();
  data.m_miny = clamp(data.m_miny, 0.0, 1.0);
  data.m_maxy = c->parseFloat32();
  data.m_maxy = clamp(data.m_maxy, 0.0, 1.0);
  data.m_minz = c->parseFloat32();
  data.m_minz = clamp(data.m_minz, 0.0, 1.0);
  data.m_maxz = c->parseFloat32();
  data.m_maxz = clamp(data.m_maxz, 0.0, 1.0);
  return new SetClipRegionCommand(data);
}

Command*
parseSetVoxelScaleCommand(CommandBufferIterator* c)
{
  SetVoxelScaleCommandD data;
  data.m_x = c->parseFloat32();
  data.m_y = c->parseFloat32();
  data.m_z = c->parseFloat32();
  return new SetVoxelScaleCommand(data);
}
Command*
parseAutoThresholdCommand(CommandBufferIterator* c)
{
  AutoThresholdCommandD data;
  data.m_channel = c->parseInt32();
  data.m_method = c->parseInt32();
  return new AutoThresholdCommand(data);
}
