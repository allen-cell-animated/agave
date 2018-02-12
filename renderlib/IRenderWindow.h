#pragma once

#include <inttypes.h>

class CCamera;
class CStatus;
class RenderParams;
class Scene;

class IRenderWindow
{
public:
	IRenderWindow();
	virtual ~IRenderWindow();

	virtual void initialize(uint32_t w, uint32_t h) = 0;
	virtual void render(const CCamera& camera) = 0;
	virtual void resize(uint32_t w, uint32_t h) = 0;
	virtual void cleanUpResources() {}

	// an interface for reporting statistics and other data updates
	virtual CStatus* getStatusInterface() { return nullptr; }

	// I own these.
	virtual RenderParams& renderParams() = 0;

	virtual Scene* scene() = 0;
	virtual void setScene(Scene* s) = 0;

};

