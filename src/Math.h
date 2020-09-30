#pragma once
#include "Ray.h"
#include <limits>
#include <vector>
#include <iostream>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#define EPSILON 0.0000000001
#define PI 3.14159265358979323846264338327950288
#define TWO_PI 6.283185307179586476925286766559

/*
	Returns the mod of all elements of "v" by "m"
*/
inline glm::ivec3 intmod(glm::ivec3 v, int m) {
	return glm::ivec3(v.x % m, v.y % m, v.z % m);
};

/*
	Converts 3D point to 1D index in ZYX order.
*/
inline int to1D(int width, int height, int x, int y, int z) {
	return height * width * z + width * y + x;
}

/*
	Converts 3D point to 1D index in ZYX order.
*/
inline int to1D(int width, int height, glm::vec3 vec) {
	return to1D(width, height, vec.x, vec.y, vec.z);
}

/*
	Converts 1D index in ZYX order to 3D point.
*/
inline void to3D(int index, int width, int height, int &x, int &y, int &z) {
	z = index / (width * height);
	index -= (z * width * height);
	y = index / width;
	x = index % width;
}

/*
	Converts 1D index in ZYX order to 3D point.
*/
inline void to3D(int index, int width, int height, glm::ivec3 &vec) {
	return to3D(index, width, height, vec.x, vec.y, vec.z);
}

/*
	Receives an integer with a max value of 10 bits (1024) and interleaves it by 3.
*/
inline int interleaveBits(int x) {
	if (x == (1 << 10)) --x;
	x = (x | (x << 16)) & 0b00000011000000000000000011111111;
	x = (x | (x << 8)) & 0b00000011000000001111000000001111;
	x = (x | (x << 4)) & 0b00000011000011000011000011000011;
	x = (x | (x << 2)) & 0b00001001001001001001001001001001;

	return x;
}

/*
	Encodes 3D point to morton code using 10bits for each axis in ZYX order.
*/
inline int encodeMorton3D(int x, int y, int z) {
	return (interleaveBits(z) << 2) | (interleaveBits(y) << 1) | interleaveBits(x);
}

/*
	Encodes 3D point to morton code using 10bits for each axis in ZYX order.
*/
inline int encodeMorton3D(glm::ivec3 v) {
	return encodeMorton3D(v.x, v.y, v.z);
}

/*
	Decodes morton code to 3D point. Assumes ZYX order.
*/
inline void decodeMorton3D(int value, int &x, int&y, int &z) {
	z = value >> 2;
	z = z & 0b00001001001001001001001001001001;
	z = (z | (z >> 2)) & 0b00000011000011000011000011000011;
	z = (z | (z >> 4)) & 0b00000011000000001111000000001111;
	z = (z | (z >> 8)) & 0b00000011000000000000000011111111;
	z = (z | (z >> 16)) & 0b00000000000000000000001111111111;

	y = value >> 1;
	y = y & 0b00001001001001001001001001001001;
	y = (y | (y >> 2)) & 0b00000011000011000011000011000011;
	y = (y | (y >> 4)) & 0b00000011000000001111000000001111;
	y = (y | (y >> 8)) & 0b00000011000000000000000011111111;
	y = (y | (y >> 16)) & 0b00000000000000000000001111111111;

	x = value;
	x = x & 0b00001001001001001001001001001001;
	x = (x | (x >> 2)) & 0b00000011000011000011000011000011;
	x = (x | (x >> 4)) & 0b00000011000000001111000000001111;
	x = (x | (x >> 8)) & 0b00000011000000000000000011111111;
	x = (x | (x >> 16)) & 0b00000000000000000000001111111111;
}

/*
	Decodes morton code to 3D point. Assumes ZYX order.
*/
inline glm::ivec3 decodeMorton3D(int value) {
	glm::ivec3 res;
	decodeMorton3D(value, res.x, res.y, res.z);
	return res;
}

/*
	Encodes given coordiantes into a single integer where each axis is shifted 10 bits to the left in the ZYX order.
*/
inline int encodeSimple3D(int x, int y, int z) {
	int res = 0;
	res = z << 20;
	res = res | y << 10;
	res = res | x;
	return res;
}

/*
	Encodes given coordiantes into a single integer where each axis is shifted 10 bits to the left in the ZYX order.
*/
inline int encodeSimple3D(glm::ivec3 v) {
	return encodeSimple3D(v.x, v.y, v.z);
}

/*
   Decodes coordinates that were shifted 10 bits and fit into an integer value.
*/
inline void decodeSimple3D(int value, int &x, int &y, int &z) {
	//If not unsigned right shift was filling with 1's
	value = value & 0b00111111111111111111111111111111;
	unsigned int v = value;
	z = v >> 20;
	y = (v << 12) >> 22;
	x = (v << 22) >> 22;
}

/*
	Decodes coordinates that were shifted 10 bits and fit into an integer value.
*/
 inline glm::ivec3 decodeSimple3D(int value) {
	glm::ivec3 res;
	decodeSimple3D(value, res.x, res.y, res.z);
	return res;
}

/*
   Decodes coordinates that were shifted 10 bits and fit into an integer value.
*/
inline void decodeSimple3D(int value, glm::ivec3 &vec) {
	vec = decodeSimple3D(value);
}

/*
	Check if the ray with origin "origin" and direction "dir" intersects the Axis-Aligned Bounding Box (AABB) restrained by bmin and bmax.
*/
inline glm::vec2 intersectBox(glm::vec3 origin, glm::vec3 dir, glm::vec3 bmin, glm::vec3 bmax) {
	//Line's/Ray's equation
	// o + t*d = y
	// t = (y - o)/d
	//when t is negative, the box is behind the ray origin
	glm::vec3 tMinTemp = (bmin - origin) / dir;
	glm::vec3 tmaxTemp = (bmax - origin) / dir;

	glm::vec3 tMin = glm::min(tMinTemp, tmaxTemp);
	glm::vec3 tMax = glm::max(tMinTemp, tmaxTemp);

	float t0 = glm::max(tMin.x, glm::max(tMin.y, tMin.z));
	float t1 = glm::min(tMax.x, glm::min(tMax.y, tMax.z));

	//if t0 > t1 = miss
	return glm::vec2(t0, t1);
}

/*
	Check if the ray with origin "origin" and direction "dir" intersects the Axis-Aligned Bounding Box (AABB) restrained by bmin and bmax.
*/
inline glm::vec2 intersectBox(Ray r, glm::vec3 bmin, glm::vec3 bmax) {
	return intersectBox(r.origin, r.direction, bmin, bmax);
}

/*
	Sets the first bit of "mortonCode" to 1
*/
inline int setEmptyBit(int mortonCode) {
	return (mortonCode | 0b10000000000000000000000000000000) & 0b10000000000000000000000000000000;
}

/*
	Checks if the first bit of "mortonCode" is set to 1, if so returns true, else returns false.
*/
inline bool isEmpty(int mortonCode) {
	return mortonCode & 0b10000000000000000000000000000000;
}

inline void radixSort(int *grid, int size) {
	std::vector<int> gridVec(grid, grid + size);
	std::vector<int> tempArr(size);
	const int bitsPerPass = 6;
	int nBits = 30;
	int nPasses = nBits / bitsPerPass;

	for (int i = 0; i < nPasses; i++) {
		int lowBit = i * bitsPerPass;
		std::vector<int> &toBeSorted = (i & 1) ? tempArr : gridVec;
		std::vector<int> &sortedValues = (i & 1) ? gridVec : tempArr;

		const int nBuckets = 1 << bitsPerPass;
		int bucketCount[nBuckets] = { 0 };
		int bitMask = (1 << bitsPerPass) - 1;
		for (int mc : toBeSorted) {
			int bucket = (mc >> lowBit) & bitMask;
			++bucketCount[bucket];
		}

		int outIndex[nBuckets];
		outIndex[0] = 0;
		for (int k = 1; k < nBuckets; ++k)
			outIndex[k] = outIndex[k - 1] + bucketCount[k - 1];

		for (int mc : toBeSorted) {
			int bucket = (mc >> lowBit) & bitMask;
			sortedValues[outIndex[bucket]++] = mc;
		}
	}

	if (nPasses & 1)
		std::swap(gridVec, tempArr);

	std::copy(gridVec.begin(), gridVec.end(), grid);
}

inline int sign(float x) {
	return x >= 0 ? 1 : -1;
}

/*
	Returns the number of leading zeros of "x".
*/
inline int clz(unsigned int x) {
	// Keep shifting x by one until leftmost bit does not become 1. 
	int totalBits = sizeof(x) * 8;
	int res = 0;
	while (!(x & (1 << (totalBits - 1)))) {
		x = (x << 1);
		res++;
	}

	return res;
}

inline bool isAllZero(glm::vec3 v) {
	return (v.x == 0 && v.y == 0 && v.z == 0) ? true : false;
}

inline void printVec3(glm::vec3 v) {
	std::cout << "[" << v.x << ", " << v.y << ", " << v.z << "]" << std::endl;
}

inline bool greaterThan(glm::vec3 v1, glm::vec3 v2) {
	if (v1.x > v2.x || v1.y > v2.y || v1.z > v2.z)
		return true;
	return false;
}

inline bool greaterEqualThan(glm::vec3 v1, glm::vec3 v2) {
	if (v1.x >= v2.x || v1.y >= v2.y || v1.z >= v2.z)
		return true;
	return false;
}

inline float log4(float x) {
	return log(x) / log(4);
}

inline float log8(float x) {
	return log(x) / log(8);
}