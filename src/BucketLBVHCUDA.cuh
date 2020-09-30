#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MathCUDA.cuh"
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#include <iostream>
#include "Timer.h"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/remove.h>

struct is_zero {
	__host__ __device__
		auto operator()(int x) const -> bool {
		return x == 0;
	}
};

/*
	Transforms coordinates x, y, z into World Coordinates System.
	For performance tests this factor was taken out on both Reference and Bucket algorithms.
*/
__host__ __device__
glm::vec3 getWCS(float x, float y, float z) {
	return glm::vec3(x, y, z);
	//return ((glm::vec3(x, y, z) / gridSize)) * scale + translate;
}
__host__ __device__
glm::vec3 getWCS(glm::vec3 v) {
	return getWCS(v.x, v.y, v.z);
}
__host__ __device__
glm::vec3 getWCS(int mortonCode) {
	int x, y, z;
	decodeSimple3D(mortonCode, x, y, z);
	return glm::vec3(x, y, z);
}

/*
	Returns 2 ^ "exponent" using a simple bit shift. 
*/
__host__ __device__
int powBase2(int exponent) {
	return 1 << (exponent);
}

/*
	Returns the sum of all powers of two up to "exponent".
*/
__host__ __device__
int sumOfBase2(int exponent) {
	int sum = 1;
	for (int i = 1; i <= exponent; i++)
		sum += powBase2(i);
	return sum;
}

/*
	Returns the parent index of "node".
*/
__device__
int getParent(int node) {
	float div = node / 2.0f;
	float res = std::floor(div);
	return res;
}

/*
	Returns the leftmost child of "node" given the leftmost and rightmost node indexes of this node's tree level
*/
__device__
int getLeftmostChild(int node, int leftmost, int rightmost) {
	return node + (rightmost - node) + 2 * (node - leftmost) + 1;
}

/*
	Returns the rightmost child of "node" given the leftmost and rightmost node indexes of this node's tree level
*/
__device__
int getRightmostChild(int node, int leftmost, int rightmost) {
	return getLeftmostChild(node, leftmost, rightmost) + 1;
}

/*
	Returns the leftmost child of "node" given this node's level
*/
__device__
int getLeftmosChild(int node, int lvl) {
	int leftmost = sumOfBase2(lvl) - powBase2(lvl) + 1;
	int rightmost = sumOfBase2(lvl);
	return getLeftmostChild(node, leftmost, rightmost);
}

/*
	Returns the rightmost child of "node" given this node's level
*/
__device__
int getRightmostChild(int node, int lvl) {
	return getLeftmosChild(node, lvl) + 1;
}

__global__
void traverseTreeUntil(Ray r, glm::vec3 axis, float depth, int *node, int numberOfNodes, int levels, int *offsets, int *mortonCodes, float *grid, glm::ivec3 gridSize, glm::vec3 *res, glm::ivec2 screenRes, int batch, int numberOfBatches) {
	int totalPixels = screenRes.x * screenRes.y;
	int rangeMin = (totalPixels / numberOfBatches) * batch;
	int rangeMax = rangeMin + (totalPixels / numberOfBatches);
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	index += rangeMin;
	//Checks whether this thread id is out of screen bounds
	if (!(index >= rangeMin && index < rangeMax))
		return;

	int x, y;
	to2D(index, screenRes.x, screenRes.y, x, y);
	glm::ivec2 currentPixel = glm::ivec2(x, y);

	//Determine ray origin given the axis of each AABB's face and ray direction
	r.origin = glm::vec3(0);
	int f = 0;
	for (int i = 0; i < 3; i++) {
		if (axis[i] != 0) {
			r.origin[i] = currentPixel[f++] + 0.5f;
		}else if (axis[i] == 0) {
			if (r.direction[i] < 0)
				r.origin[i] = gridSize[i] + 0.5f;
			else
				r.origin[i] = 0;
		}
	}

	int currentNode = 1;
	int currentLevel = 0;
	const int offsetMinMax = 2;

	glm::vec2 t = intersectBox(r, getWCS(node[currentNode * offsetMinMax - 1]), getWCS(node[currentNode * offsetMinMax]));
	glm::vec2 finalt = glm::vec2(999999, -999999);

	if (t.x > t.y) {
		res->x = finalt.x;
		res->y = finalt.y;
		atomicAdd(&res->z, 0);
		return;
	}

	float accumulatedDensity = 0;
	int firstNodeAtDeepestLevel = numberOfNodes - powBase2(levels - 1) + 1;

	while (currentNode != numberOfNodes + 1) {
		bool miss = false;

		//If current node is empty, automatically misses
		if (isEmpty(node[currentNode * offsetMinMax]))
			miss = true;
		else {
			t = intersectBox(r, getWCS(node[currentNode * offsetMinMax - 1]), getWCS(node[currentNode * offsetMinMax]));
			if (t.x > t.y)
				miss = true;
		}

		//If miss
		if (miss) {
			//If it's the rightmost node on current level, end
			if (currentNode == sumOfBase2(currentLevel))
				break;

			//if this node is the rightmost child of its parent
			if (getRightmostChild(getParent(currentNode), currentLevel - 1) == currentNode) {
				currentNode = getParent(currentNode) + 1;
				currentLevel--;
			}else if (getRightmostChild(currentNode, currentLevel) == currentNode) {
				currentNode = getParent(currentNode) + 1;
				currentLevel--;
			}else {
				currentNode = currentNode + 1;
			}

			continue;
		}

		//If we are checking a leaf node
		if (currentNode >= firstNodeAtDeepestLevel) {

			int offsetsPosition = currentNode - firstNodeAtDeepestLevel;
			int startingIndex;
			int elementsOnThisBucket = 0;

			if (offsetsPosition == 0) {
				startingIndex = 0;
				elementsOnThisBucket = offsets[offsetsPosition];
			}else {
				startingIndex = offsets[offsetsPosition - 1];
				elementsOnThisBucket = offsets[offsetsPosition] - offsets[offsetsPosition - 1];
			}

			//For each voxel on this bucket (leaf node), check which ones intersect this ray.
			//Remember that here we only check mortonCodes that represent non-empty voxels
			for (int i = 0; i < elementsOnThisBucket; i++) {
				int morton = mortonCodes[startingIndex + i];
				glm::vec3 min, max;
				min = decodeMorton3D(morton);
				max = min + 1.0f;

				glm::vec2 t2 = intersectBox(r, getWCS(min), getWCS(max));

				//If intersects this voxel at current bucket, accumulate density and update intersection t's
				if (t2.x <= t2.y) {
					float densitySampled = grid[to1D(gridSize.x, gridSize.y, min.x, min.y, min.z)];
					accumulatedDensity += grid[to1D(gridSize.x, gridSize.y, min.x, min.y, min.z)];

					if (t2.x < finalt.x)
						finalt.x = t2.x;
					if (t2.y >= finalt.y)
						finalt.y = t2.y;

					float distance = finalt.y - glm::max(0.0f, finalt.x);
					if (distance > depth) {
						res->x = finalt.x;
						res->y = finalt.y;
						atomicAdd(&res->z, accumulatedDensity);
						return;
					}
				}
			}

			//If this node was the tree's last, end.
			if (currentNode == numberOfNodes)
				break;

			if (getRightmostChild(getParent(currentNode), currentLevel - 1) == currentNode) {
				currentNode = getParent(currentNode) + 1;
				currentLevel--;
			}else {
				currentNode = currentNode + 1;
			}
		}else {
			currentNode = getLeftmosChild(currentNode, currentLevel);
			currentLevel++;
		}
	}

	res->x = finalt.x;
	res->y = finalt.y;
	atomicAdd(&res->z, accumulatedDensity);
	return;
}

__global__
void generateLevel(int rightmost, int leftmost, int *node, int *emptyNodes) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int currentNode = rightmost - index;
	if (currentNode < leftmost)
		return;

	const int offsetMinMax = 2;
	int children[2];
	children[0] = getLeftmostChild(currentNode, leftmost, rightmost);
	children[1] = children[0] + 1;

	//If all children have the empty bit set then currentNode is also empty.
	if (isEmpty(node[children[0] * offsetMinMax]) && isEmpty(node[children[1] * offsetMinMax])) {
		atomicAdd(emptyNodes, 1);
		node[currentNode * offsetMinMax - 1] = setEmptyBit(0);
		node[currentNode * offsetMinMax] = setEmptyBit(0);
		currentNode--;
		return;
	}

	glm::ivec3 min = glm::vec3(99999, 99999, 99999);
	glm::ivec3 max = glm::vec3(-99999, -99999, -99999);
	//For each child calculate the AABB that encapsulates all of them and set it as currentNode AABB
	for (int c = 0; c < 2; c++) {
		if (isEmpty(node[children[c] * 2]))
			continue;

		min = glm::min(min, decodeSimple3D(node[children[c] * 2 - 1]));
		max = glm::max(max, decodeSimple3D(node[children[c] * 2]));
	}

	node[currentNode * 2 - 1] = encodeSimple3D(min);
	node[currentNode * 2] = encodeSimple3D(max);
}

__global__
void generateMortonCodes(float *grid, int *mortonCodes, int gridSize, glm::ivec3 gridRes) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < gridSize; i += stride) {
		if (grid[i] != 0) {
			glm::ivec3 v;
			to3D(i, gridRes.x, gridRes.y, v);
			mortonCodes[i] = encodeMorton3D(v);
		}
	}
}

__global__
void generateLeafs(int *mortonCodes, int *offsets, int *node, int firstNodeAtDeepestLevel, int numberOfNodes, int *emptyNodes) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int currentNode = firstNodeAtDeepestLevel + index;
	int offsetMinMax = 2;

	//If over node array boundaries
	if (currentNode > numberOfNodes)
		return;

	//If there is 0 elements in this bucket (is empty) set leaf as empty
	if (index != 0 && offsets[index] - offsets[index - 1] == 0) {
		node[currentNode * offsetMinMax - 1] = setEmptyBit(0);
		node[currentNode * offsetMinMax] = setEmptyBit(0);
		atomicAdd(emptyNodes, 1);
	}else {
	//If not, for each voxel contained within this bucket, calculate its AABB and compute current Leaf AABB
		int offsetsPosition = currentNode - firstNodeAtDeepestLevel;
		int startingIndex;
		int elementsOnThisBucket = 0;

		if (offsetsPosition == 0) {
			startingIndex = 0;
			elementsOnThisBucket = offsets[offsetsPosition];
		}
		else {
			startingIndex = offsets[offsetsPosition - 1];
			elementsOnThisBucket = offsets[offsetsPosition] - offsets[offsetsPosition - 1];
		}

		glm::vec3 min = glm::vec3(99999, 99999, 99999);
		glm::vec3 max = glm::vec3(-99999, -99999, -99999);

		for (int i = 0; i < elementsOnThisBucket; i++) {
			int morton = mortonCodes[startingIndex + i];
			glm::vec3 coord = decodeMorton3D(morton);
			min = glm::min(min, coord);
			max = glm::max(max, coord + 1.0f);
		}

		node[currentNode * offsetMinMax - 1] = encodeSimple3D(min.x, min.y, min.z);
		node[currentNode * offsetMinMax] = encodeSimple3D(max.x, max.y, max.z);
	}
}

__global__
void fillbuckets(int *mortonCodes, int mortonCodesSize, int bucketSize, int *offsets) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < mortonCodesSize; i += stride) {
		int bucketIndex = mortonCodes[i] / bucketSize;
		atomicAdd(&offsets[bucketIndex], 1);
	}
}

void bucketLBVHMeasurePerformance(float *data, glm::ivec3 gridRes, int bucketSize, std::stringstream &ss, float *results) {
	Timer *t = new Timer();
	int gridSize = gridRes.x * gridRes.y * gridRes.z;
	float *grid;
	int *mortonCodes;
	int mortonCodesSize;
	int *nonEmptyBuckets;
	int levels = 0;
	const int offsetMinMax = 2;
	int *numberOfEmptyNodes = 0;
	float totalElapsedTime = 0;
	float milliseconds = 0;
	int blockSize;
	int numBlocks;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Pre-allocate memory
	cudaMalloc(&grid, gridSize * sizeof(float));
	HANDLE_ERROR(cudaPeekAtLastError());

	cudaMemcpy(grid, data, gridSize * sizeof(float), cudaMemcpyHostToDevice);
	HANDLE_ERROR(cudaPeekAtLastError());

	cudaMallocManaged(&mortonCodes, gridSize * sizeof(int));
	HANDLE_ERROR(cudaPeekAtLastError());

	cudaMallocManaged(&nonEmptyBuckets, 1 * sizeof(int));
	HANDLE_ERROR(cudaPeekAtLastError());

	cudaMallocManaged(&numberOfEmptyNodes, 1 * sizeof(int));
	HANDLE_ERROR(cudaPeekAtLastError());
	*numberOfEmptyNodes = 0;

	ss << "Bucket bin. tree" << std::endl;
	ss << std::endl;
	ss << "Generate tree (milliseconds)" << std::endl;
	ss << std::endl;

	//=====================
	//1. Generate Morton codes
	cudaEventRecord(start);
	blockSize = 256;
	numBlocks = (gridSize + blockSize - 1) / blockSize;
	generateMortonCodes << <numBlocks, blockSize >> > (grid, mortonCodes, gridSize, gridRes);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	ss << "generate morton codes: \t" << milliseconds << std::endl;
	totalElapsedTime += milliseconds;
	HANDLE_ERROR(cudaPeekAtLastError());

	//=====================
	//2. Sorts all morton codes in ascending order
	cudaEventRecord(start);
	thrust::sort(thrust::device, mortonCodes, mortonCodes + gridSize);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	ss << "thrust sort: \t\t\t" << milliseconds << std::endl;
	totalElapsedTime += milliseconds;
	HANDLE_ERROR(cudaPeekAtLastError());

	//=====================
	//3. Remove invalid morton codes (empty voxels were marked as zeroes on step 1)
	int *tempArr;
	cudaEventRecord(start);
	int *newEnd = thrust::remove_if(thrust::device, mortonCodes, mortonCodes + gridSize, is_zero());
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	ss << "thrust remove_if: \t\t" << milliseconds << std::endl;
	totalElapsedTime += milliseconds;
	HANDLE_ERROR(cudaPeekAtLastError());

	mortonCodesSize = newEnd - mortonCodes;

	//=====================
	//4. Copy non-empty voxel morton codes and generate offsets by doing parallel prefix sum (scan)
	int *offsets;
	int *tempOffsets;
	//The minimun amount of buckets is the highest morton code value divided by the bucket size
	int offsetsSize = std::ceil(std::log2(std::ceil(mortonCodes[mortonCodesSize - 1] / float(bucketSize))));
	offsetsSize = powBase2(offsetsSize);
	cudaMallocManaged(&offsets, offsetsSize * sizeof(int));
	cudaMallocManaged(&tempOffsets, offsetsSize * sizeof(int));

	cudaEventRecord(start);
	//Fills all empty bucket offsets to the right with 0's
	thrust::fill(thrust::device, tempOffsets, tempOffsets + offsetsSize, 0);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totalElapsedTime += milliseconds;
	HANDLE_ERROR(cudaPeekAtLastError());

	cudaEventRecord(start);
	blockSize = 256;
	numBlocks = (offsetsSize + blockSize - 1) / blockSize;
	//Counts how many morton codes are within each bucket and store its count on tempOffsets[]
	fillbuckets << <numBlocks, blockSize >> > (mortonCodes, mortonCodesSize, bucketSize, tempOffsets);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	ss << "fill buckets: \t\t\t" << milliseconds << std::endl;
	totalElapsedTime += milliseconds;
	cudaDeviceSynchronize();
	HANDLE_ERROR(cudaPeekAtLastError());

	cudaEventRecord(start);
	int *newEndOffsets = thrust::remove_if(thrust::device, tempOffsets, tempOffsets + offsetsSize, is_zero());
	*nonEmptyBuckets = (newEndOffsets - tempOffsets);
	thrust::fill(thrust::device, tempOffsets + *nonEmptyBuckets, tempOffsets + offsetsSize, 0);
	thrust::inclusive_scan(thrust::device, tempOffsets, tempOffsets + offsetsSize, offsets);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	ss << "compact buckets: \t\t" << milliseconds << std::endl;
	totalElapsedTime += milliseconds;
	cudaDeviceSynchronize();
	HANDLE_ERROR(cudaPeekAtLastError());

	levels = std::ceil(std::log2(*nonEmptyBuckets)) + 1;
	int numberOfNodes = sumOfBase2(levels - 1);
	int amountOfBuckets = powBase2(levels - 1);
	int *node;
	int firstNodeAtDeepestLevel = numberOfNodes - powBase2(levels - 1) + 1;
	cudaMallocManaged(&node, (numberOfNodes * offsetMinMax + 1) * sizeof(int));

	//=====================
	//5. Generate Leafs(Buckets)
	cudaEventRecord(start);
	blockSize = 256;
	numBlocks = (amountOfBuckets + blockSize - 1) / blockSize;
	generateLeafs << <numBlocks, blockSize >> > (mortonCodes, offsets, node, firstNodeAtDeepestLevel, numberOfNodes, numberOfEmptyNodes);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	ss << "generate leafs: \t\t" << milliseconds << std::endl;
	totalElapsedTime += milliseconds;
	HANDLE_ERROR(cudaPeekAtLastError());

	//=====================
	//6. Generate rest of the tree in a bottom-up manner
	//currentNode starts at the rightmost node of its level
	int currentNode = firstNodeAtDeepestLevel - 1;
	cudaEventRecord(start);
	//Since the deepest level is already created (Leafs bucket), start one level above and walk the tree BFS backwards
	for (int l = levels - 2; l >= 0; l--) {
		int nodesInThisLevel = powBase2(l);
		blockSize = 256;
		numBlocks = (nodesInThisLevel + blockSize - 1) / blockSize;
		int leftmost = currentNode - nodesInThisLevel + 1;
		int rightmost = currentNode;
		generateLevel << <numBlocks, blockSize >> > (rightmost, leftmost, node, numberOfEmptyNodes);
		//cudaDeviceSynchronize();
		currentNode = leftmost - 1;
	}
	HANDLE_ERROR(cudaPeekAtLastError());
	cudaEventRecord(stop);
	cudaDeviceSynchronize();
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	ss << "generate tree: \t\t\t" << milliseconds << std::endl;
	totalElapsedTime += milliseconds;

	//DEEBUG
	/*for (int i = numberOfNodes - 64 + 1; i <= numberOfNodes; i++) {
		std::cout << "node " << i << "{" << std::endl;
		std::cout << node[i * 2] << std::endl;
		if (!isEmpty(node[i * 2])) {
			std::cout << "min: ";
			printVec3(decodeSimple3D(node[i * 2 - 1]));
			std::cout << "max: ";
			printVec3(decodeSimple3D(node[i * 2]));
		}
		else {
			std::cout << "empty" << std::endl;
		}
		std::cout << "}" << std::endl;
	}

	std::cout << "offsets: ";
	for (int i = 0; i < amountOfBuckets; i++)
		std::cout << offsets[i] << ", ";

	std::cout << std::endl;
	for (int i = 0; i < mortonCodesSize; i++)
		std::cout << mortonCodes[i] << std::endl;
	*/

	glm::vec3 *res;
	cudaMallocManaged(&res, 1 * sizeof(glm::vec3));
	HANDLE_ERROR(cudaPeekAtLastError());
	res->x = 0;
	res->y = 0;
	res->z = 0;

	Ray ray;
	float avgTraverseTime = 0;
	float accumulatedDensity = 0;

	const int numberOfFaces = 6;
	glm::vec3 faceAxis[numberOfFaces] = {glm::vec3(1, 1, 0), glm::vec3(0, 1, 1), glm::vec3(1, 1, 0), glm::vec3(0, 1, 1), glm::vec3(1, 0, 1), glm::vec3(1, 0, 1)};
	glm::vec3 rayDirection[numberOfFaces] = {glm::vec3(0, 0, 1), glm::vec3(-1, 0, 0), glm::vec3(0, 0, -1), glm::vec3(1, 0, 0), glm::vec3(0, -1, 0), glm::vec3(0, 1, 0)};
	
	//For each face of the AABB compute the average traversal time of all pixels
	for (int v = 0; v < numberOfFaces; v++){
		glm::vec2 screenRes;
		int ind = 0;
		for (int i = 0; i < 3; i++) {
			if (faceAxis[v][i] != 0)
				screenRes[ind++] = gridRes[i];
		}
		int totalPixels = screenRes.x * screenRes.y;
		ray.direction = rayDirection[v];

		float currentFaceAvgTraverseTime = 0;

		int batches = 4;
		for (int i = 0; i < batches; i++) {
			cudaEventRecord(start);
			blockSize = 1024;
			numBlocks = (totalPixels / batches + blockSize - 1) / blockSize;
			traverseTreeUntil <<<numBlocks, blockSize >>> (ray, faceAxis[v], 99999, node, numberOfNodes, levels, offsets, mortonCodes, grid, gridRes, res, screenRes, i, batches);
			HANDLE_ERROR(cudaPeekAtLastError());
			cudaEventRecord(stop);
			cudaDeviceSynchronize();
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliseconds, start, stop);
			currentFaceAvgTraverseTime += milliseconds;
			HANDLE_ERROR(cudaPeekAtLastError());
		}

		currentFaceAvgTraverseTime /= screenRes.x * screenRes.y;
		avgTraverseTime += currentFaceAvgTraverseTime;
	}

	accumulatedDensity += res->z;
	avgTraverseTime = avgTraverseTime / numberOfFaces;

	ss << "Total generation time: \t" << totalElapsedTime << std::endl;
	ss << std::endl;
	ss << "Accumulated density: " << accumulatedDensity << std::endl;
	ss << "Avg. traverse time: " << avgTraverseTime << std::endl;
	ss << std::endl;

	ss << "Depth: " << levels << std::endl;
	ss << "Number of nodes: " << numberOfNodes << std::endl;
	ss << "Number of empty nodes: " << *numberOfEmptyNodes << std::endl;
	ss << "% of non-empty nodes: " << (1.0f - (*numberOfEmptyNodes / (float)numberOfNodes)) * 100 << std::endl;
	ss << "Bucket size: " << bucketSize <<std::endl;

	//"Number of nodes", "Number of empty nodes", 
	//"Nodes Mem. consumption (KB)", "Offset Mem. consumption (KB)", "MortonCode Mem. consumption (KB)", "Total Mem. consumption (MB)", 
	// "Tree construction time (ms)", "Tree avg. Traversal time (ms)", "Avg. accumulated density"
	results[0] = results[0] + numberOfNodes;
	results[1] = results[1] +  (*numberOfEmptyNodes);
	float nodesMemKB = sizeof(int) * 2 * numberOfNodes / 1024.0f;
	float offsetMemKB = sizeof(int) * amountOfBuckets / 1024.0f;
	float mortonMemKB = sizeof(int) * mortonCodesSize / 1024.0f;
	results[2] = results[2] + nodesMemKB;
	results[3] = results[3] + offsetMemKB;
	results[4] = results[4] + mortonMemKB;
	results[5] = results[5] + (nodesMemKB + offsetMemKB + mortonMemKB) / 1024.0f;
	results[6] = results[6] + totalElapsedTime;
	results[7] = results[7] + avgTraverseTime;
	results[8] = results[8] + accumulatedDensity;
	
	cudaFree(grid);
	HANDLE_ERROR(cudaPeekAtLastError());
	cudaFree(mortonCodes);
	HANDLE_ERROR(cudaPeekAtLastError());
	cudaFree(nonEmptyBuckets);
	HANDLE_ERROR(cudaPeekAtLastError());
	cudaFree(numberOfEmptyNodes);
	HANDLE_ERROR(cudaPeekAtLastError());
	cudaFree(offsets);
	HANDLE_ERROR(cudaPeekAtLastError());
	cudaFree(tempOffsets);
	HANDLE_ERROR(cudaPeekAtLastError());
	cudaFree(node);
	HANDLE_ERROR(cudaPeekAtLastError());
}