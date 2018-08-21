#include <QApplication>
#include <QCommandLineParser>
#include <QDir>

#include "renderlib/FileReader.h"
#include "renderlib/ImageXYZC.h"

#undef foreach
#include <openvdb/openvdb.h>

#include <iostream>

int main(int argc, char *argv[])
{
	QCoreApplication a(argc, argv);

	QCommandLineParser parser;
	parser.setApplicationDescription("OME-TIFF to VDB converter");
	parser.addHelpOption();
	parser.addVersionOption();
	parser.addPositionalArgument("in", QCoreApplication::translate("in", "input ome tiff file."));
	parser.addPositionalArgument("out", QCoreApplication::translate("out", "output vdb file."));

	// Process the actual command line arguments given by the user
	parser.process(a);

	const QStringList args = parser.positionalArguments();
    // in is args.at(0)
    // out is args.at(1)
	if (args.size() < 2) { 
        // missing arg.
        //cout << "need 2 args, in file and out file"
        return 1;
    }

    QString infile = args.at(0);
    QString outfile = args.at(1);

    // load the in file.
    FileReader fileReader;
    std::shared_ptr<ImageXYZC> image = fileReader.loadOMETiff_4D(infile.toStdString());

    // prepare an openvdb grid
    openvdb::initialize();
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(0);

	typename openvdb::FloatGrid::Accessor accessor = grid->getAccessor();

	openvdb::Coord ijk;
	int &i = ijk[0], &j = ijk[1], &k = ijk[2];
	for (i = 0; i < image->sizeZ(); ++i) {
		for (j = 0; j < image->sizeY(); ++j) {
			for (k = 0; k < image->sizeX(); ++k) {
				// Convert the floating-point distance to the grid's value type.
				uint16_t val = image->channel(0)->_ptr[k + j*image->sizeX() + i*image->sizeX()*image->sizeY()];
				float valf = (float)val / (float)image->channel(0)->_max;
				// Set the distance for voxel (i,j,k).
				accessor.setValue(ijk, valf);
			}
		}
	}

	// set a scaling transform with the physical pixel dimensions
    grid->setTransform(openvdb::math::Transform::createLinearTransform(1.0));

    grid->setGridClass(openvdb::GRID_FOG_VOLUME);
    grid->setName("TEST");

    openvdb::GridPtrVec grids;
    grids.push_back(grid);

    openvdb::io::File file(outfile.toStdString());
    file.write(grids);
    file.close();

    return 0;
}
