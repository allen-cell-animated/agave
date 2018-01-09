#pragma once

#include <inttypes.h>

class Camera;
class RenderParams;
class Scene;

class IRenderWindow
{
public:
	IRenderWindow();
	virtual ~IRenderWindow();

	virtual void initialize(uint32_t w, uint32_t h) = 0;
	virtual void render(const Camera& camera) = 0;
	virtual void resize(uint32_t w, uint32_t h) = 0;
	virtual void cleanUpResources() {}

	// I own these.
	virtual RenderParams& renderParams() = 0;
	virtual Scene& scene() = 0;

};

