#include "Texture3D.h"

Texture3D::Texture3D() {
}

Texture3D::Texture3D(std::string filePath){
	openvdb::initialize();
	openvdb::io::File file(RESOURCES_DIR + filePath);

	openvdb::GridBase::Ptr baseGrid;
	file.open();
	for (openvdb::io::File::NameIterator nameIter = file.beginName(); nameIter != file.endName(); ++nameIter) {
		if (nameIter.gridName() == "density") {
			baseGrid = file.readGrid(nameIter.gridName());
			break;
		}
	}
	file.close();

	openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);

	openvdb::CoordBBox bb = grid->evalActiveVoxelBoundingBox();
	this->width = abs(bb.min().x()) + abs(bb.max().x());
	this->height = abs(bb.min().y()) + abs(bb.max().y());
	this->depth = abs(bb.min().z()) + abs(bb.max().z());

	openvdb::Coord dim(this->width, this->height, this->depth);
	openvdb::Coord originvdb(-abs(bb.min().x()), -abs(bb.min().y()), -abs(bb.min().z()));
	openvdb::tools::Dense<float> dense(dim, originvdb);

	openvdb::tools::copyToDense<openvdb::tools::Dense<float>, openvdb::FloatGrid>(*grid, dense);

	this->floatData = new float[this->width * this->height * this->depth];

	std::copy(dense.data(), dense.data() + this->width * this->height * this->depth, this->floatData);
}

Texture3D::~Texture3D(){
	delete []floatData;
}
