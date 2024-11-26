
#ifdef __APPLE__

#include "getsurface_wgpu_mac.h"

#include <Cocoa/Cocoa.h>
#include <Foundation/Foundation.h>
#include <QuartzCore/CAMetalLayer.h>

void*
getMetalLayerFromWindow(void* win_id)
{
  // #     id metal_layer = NULL;
  // #     NSWindow *ns_window = glfwGetCocoaWindow(window);
  // #     [ns_window.contentView setWantsLayer:YES];
  // #     metal_layer = [CAMetalLayer layer];
  // #     [ns_window.contentView setLayer:metal_layer];
  // #     surface = wgpu_create_surface_from_metal_layer(metal_layer);
  // # }

  NSView* view = (NSView*)(win_id);
  if (![view isKindOfClass:[NSView class]]) {
    return NULL;
  }

  if (![view.layer isKindOfClass:[CAMetalLayer class]]) {
    // orilayer = [view layer];
    [view setLayer:[CAMetalLayer layer]];
    [view setWantsLayer:YES];
  }

  // TODO cleanup later?
  return [view layer];
}
#endif
