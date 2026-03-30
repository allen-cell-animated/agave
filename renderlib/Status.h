#pragma once

#include <algorithm>
#include <string>
#include <vector>

class Scene;

class IStatusObserver
{
public:
  virtual void RenderBegin() = 0;
  virtual void RenderEnd() = 0;
  virtual void PreRenderFrame() = 0;
  virtual void PostRenderFrame() = 0;
  virtual void RenderPause(const bool& Paused) = 0;
  virtual void Resize() = 0;
  virtual void LoadPreset(const std::string& PresetName) = 0;
  virtual void StatisticChanged(const std::string& Group,
                                const std::string& Name,
                                const std::string& Value,
                                const std::string& Unit = "",
                                const std::string& Icon = "") = 0;
};

class CStatus
{

public:
  void EnableUpdates(bool enabled) { mUpdatesEnabled = enabled; }

  void SetRenderBegin();
  void SetRenderEnd();
  void SetPreRenderFrame();
  void SetPostRenderFrame();
  void SetRenderPause(const bool& Pause);
  void SetResize();
  void SetLoadPreset(const std::string& PresetName);
  void SetStatisticChanged(const std::string& Group,
                           const std::string& Name,
                           const std::string& Value,
                           const std::string& Unit = "",
                           const std::string& Icon = "");

  void onNewImage(const std::string& name, Scene* scene);

  void addObserver(IStatusObserver* ob) { mObservers.push_back(ob); }
  void removeObserver(IStatusObserver* ob)
  {
    auto iter = std::find(mObservers.begin(), mObservers.end(), ob);
    if (iter != mObservers.end()) {
      mObservers.erase(iter);
    }
  }

  std::vector<IStatusObserver*> mObservers;
  bool mUpdatesEnabled = true;
};
