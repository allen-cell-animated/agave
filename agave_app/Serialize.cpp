#include "Serialize.h"

namespace Serialize {
ViewerState
fromV1(const ViewerState_V1& v1)
{
  ViewerState v2;

  LoadSettings ls;
  ls.url = v1.name;
  ls.subpath = "";
  ls.scene = v1.scene;
  ls.time = v1.timeline.currentTime;
  // v1 assumed load all channels
  // v1 assumed load full clipRegion
  v2.datasets.push_back(ls);

  v2.version = v1.version;
  v2.pathTracer = v1.pathTracer;
  v2.timeline = v1.timeline;
  v2.clipRegion = v1.clipRegion;
  v2.scale = v1.scale;
  v2.camera = v1.camera;
  v2.backgroundColor = v1.backgroundColor;
  v2.boundingBoxColor = v1.boundingBoxColor;
  v2.showBoundingBox = v1.showBoundingBox;
  v2.channels = v1.channels;
  v2.density = v1.density;
  v2.lights = v1.lights;

  v2.capture.durationType = DurationType_PID::SAMPLES;
  v2.capture.samples = v1.renderIterations;
  v2.capture.width = v1.resolution[0];
  v2.capture.height = v1.resolution[1];
  v2.capture.startTime = v1.timeline.currentTime;
  v2.capture.endTime = v1.timeline.currentTime;

  return v2;
}
} // namespace Serialize