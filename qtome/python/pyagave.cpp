#include "pybind11/embed.h"

#include "pyrenderer.h"

namespace py = pybind11;

PYBIND11_EMBEDDED_MODULE(agave, m)
{
  m.doc() = "agave plugin"; // optional module docstring

  py::class_<OffscreenRenderer>(m, "renderer")
    .def(py::init<std::shared_ptr<libCZI::IStream>>())
    .def("is_mosaic", &pylibczi::Reader::isMosaic)
    .def("read_dims", &pylibczi::Reader::readDims)
    .def("read_dims_string", &pylibczi::Reader::dimsString)
    .def("read_dims_sizes", &pylibczi::Reader::dimSizes)
    .def("read_scene_wh", &pylibczi::Reader::getSceneYXSize)
    .def("read_meta", &pylibczi::Reader::readMeta)
    .def("read_selected", &pylibczi::Reader::readSelected)
    .def("mosaic_shape", &pylibczi::Reader::mosaicShape)
    .def("read_meta_from_subblock", &pylibczi::Reader::readSubblockMeta)
    .def("read_mosaic", &pylibczi::Reader::readMosaic);
}
