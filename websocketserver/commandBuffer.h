#pragma once

#include <stdint.h>
#include <string>
#include <vector>

class Command;
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

class CommandBufferIterator
{
public:
  CommandBufferIterator(commandBuffer* buf);

  bool end();

  int32_t parseInt32();
  float parseFloat32();
  std::string parseString();

  commandBuffer* _commandBuffer;
  uint8_t* _currentPos;
};
