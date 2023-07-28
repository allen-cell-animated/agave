#pragma once

class Manipulator
{
  virtual void drawGL() = 0;
  virtual void pickGL(int x, int y) = 0;
};

class RotateManipulator : public Manipulator
{
  RotateManipulator();
  void drawGL() override;
  void pickGL(int x, int y) override;
};
