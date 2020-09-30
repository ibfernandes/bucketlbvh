#pragma once
#define RESOURCES_DIR  "D:/Grids/"
#define __TBB_NO_IMPLICIT_LINKAGE 1
#define __TBBMALLOC_NO_IMPLICIT_LINKAGE 1
#include <openvdb/openvdb.h>
#include <openvdb/tools/Dense.h>
#include <iostream>
#include <glm/glm.hpp>

class Texture3D{
public:
	int width, height, depth;
	//3D grid Stored in ZYX order
	float *floatData;
	
	Texture3D();
	Texture3D(std::string filePath);

	inline glm::vec3 getResolution() {
		return glm::vec3(width, height, depth);
	}

	~Texture3D();
};

