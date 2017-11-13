#pragma once
#include "IRenderWindow.h"

#include <ome/files/FormatReader.h>

#include <memory>

class Image2D;
class Image3Dv33;
class ImageXYZC;

class RenderGL :
	public IRenderWindow
{
public:
	RenderGL(std::shared_ptr<ome::files::FormatReader>  reader,
		std::shared_ptr<ImageXYZC>  img,
		ome::files::dimension_size_type                    series);
	virtual ~RenderGL();

	virtual void initialize(uint32_t w, uint32_t h);
	virtual void render(const Camera& camera);
	virtual void resize(uint32_t w, uint32_t h);

	Image3Dv33* getImage() const { return image3d; };
private:
	Image3Dv33 *image3d;
	std::shared_ptr<ome::files::FormatReader>  _reader;
	std::shared_ptr<ImageXYZC>  _img;
	ome::files::dimension_size_type                    _series;
};

