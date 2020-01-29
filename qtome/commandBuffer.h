#pragma once

#include "command.h"

#include <stdint.h>
#include <string>
#include <vector>

struct ExecutionContext;
class Renderer;
class CScene;
class RenderParameters;

class commandBuffer
{
public:
  commandBuffer(size_t len, const uint8_t* buf);
  virtual ~commandBuffer();

  void processBuffer();
  void execute(ExecutionContext* c);
  const uint8_t* head() { return _headPos; }
  size_t length() { return _length; }

  std::vector<Command*> getQueue() { return _commands; }

private:
  size_t _length;
  const uint8_t* _headPos;

  // queue really.
  std::vector<Command*> _commands;
};

class CommandBufferIterator : public ParseableStream
{
public:
  CommandBufferIterator(commandBuffer* buf);

  bool end();

  virtual int32_t parseInt32();
  virtual float parseFloat32();
  virtual std::vector<float> parseFloat32Array();
  virtual std::string parseString();

  commandBuffer* _commandBuffer;
  uint8_t* _currentPos;
};
