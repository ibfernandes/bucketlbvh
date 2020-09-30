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
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__host__ __device__
struct AABB {
	glm::ivec3 min, max;
};

__host__ __device__
struct Brick {
	int mortonCode; 
	bool isEmpty = true;
	float grid[512];
};

struct BrickCmp {
	__host__ __device__
		bool operator()(const Brick& o1, const Brick& o2) {
		return o1.mortonCode < o2.mortonCode;
	}
};

struct isBrickEmpty {
	__host__ __device__
		auto operator()(Brick b) const -> bool {
		return b.isEmpty;
	}
};

__host__ __device__
struct Node {
	struct Node *leftChild = nullptr;
	struct Node *rightChild = nullptr;
	struct Node *parent = nullptr;
	Brick *brick;
	AABB aabb;
	int id;
	bool isLeaf = false;
};

__host__ __device__
AABB generateLeafAABB(Brick brick, int brickDimensionSize) {
	glm::ivec3 min = decodeMorton3D(brick.mortonCode);
	glm::ivec3 max = min + brickDimensionSize;

	return AABB{ min, max };
}

__host__ __device__
inline bool greaterEqualThan(glm::vec3 v1, glm::vec3 v2) {
	if (v1.x >= v2.x || v1.y >= v2.y || v1.z >= v2.z)
		return true;
	return false;
}

__host__ __device__
float calculateInnerBrick(Ray r, glm::vec3 mortonMin, Node *n, int &intersectionCount, int bricksSize, int brickDimensionSize, glm::ivec3 gridResolution) {
	float acc = 0;
	for (int i = 0; i < bricksSize; i++) {
		glm::ivec3 vec;
		to3D(i, brickDimensionSize, brickDimensionSize, vec);
		vec += mortonMin;
		if (greaterEqualThan(vec, gridResolution))
			continue;

		glm::vec2 hit = intersectBox(r, vec, vec + 1);
		intersectionCount++;
		if (hit.x <= hit.y) {
			acc += n->brick->grid[i];
		}
	}

	return acc;
}

__host__ __device__
void printNode(Node node) {
	printf("{\n");
	printf("id: %d\n", node.id);
	if (!node.isLeaf) {
		printf("childA: %d\n", node.leftChild->id);
		printf("childB: %d\n", node.rightChild->id);
	}
	else {
		printf("parent: %d\n", node.parent->id);
	}
	printVec3CUDA(node.aabb.min);
	printVec3CUDA(node.aabb.max);
	printf("}\n");
}

__global__
void traversePaper(Ray r, glm::vec3 axis, Node *nodes, int bricksSize, int brickDimensionSize, glm::ivec3 gridResolution, glm::vec3 *res, glm::ivec3 screenRes) {
	float acc = 0;
	int intersectionsCount = 0;

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= screenRes.z)
		return;

	int x, y;
	to2D(index, screenRes.x, screenRes.y, x, y);
	glm::ivec2 currentPixel = glm::ivec2(x, y);

	r.origin = glm::vec3(0);
	int f = 0;
	for (int i = 0; i < 3; i++) {
		if (axis[i] != 0) {
			r.origin[i] = currentPixel[f++] + 0.5f;
		}
		else if (axis[i] == 0) {
			if (r.direction[i] < 0)
				r.origin[i] = (*res)[i];
			else
				r.origin[i] = 0;
		}
	}

	Node dummyNode;
	dummyNode.id = -1;

	Node stack[64];
	Node *stackPtr = stack;
	*stackPtr++ = dummyNode; //"null node"
	*stackPtr++ = nodes[0]; // push null node to mark end of the stack
	
	Node *closestLeaf;
	Node *currentNode;

	//While stack is not empty
	while ((currentNode = &(*--stackPtr))->id != -1) {

		while (!currentNode->isLeaf) {
			Node *leftChild = currentNode->leftChild;
			Node *rightChild = currentNode->rightChild;
			
			glm::vec2 hitl = intersectBox(r, leftChild->aabb.min, leftChild->aabb.max);
			glm::vec2 hitr = intersectBox(r, rightChild->aabb.min, rightChild->aabb.max);
			intersectionsCount++;
			intersectionsCount++;

			bool didHitL = (hitl.x > hitl.y) ? false : true;
			bool didHitR = (hitr.x > hitr.y) ? false : true;

			if (didHitL && didHitR) {
				Node *near, *far;
				if (hitl.x > hitr.x) {
					near = rightChild;
					far = leftChild;
				}
				else {
					near = leftChild;
					far = rightChild;
				}
				currentNode = near;
				(*stackPtr++) = *far;
			}
			else if (didHitL) {
				currentNode = leftChild;
			}
			else if (didHitR) {
				currentNode = rightChild;
			}
			else {
				break;
			}
		}

		closestLeaf = currentNode;
		if (closestLeaf->isLeaf)
			acc += calculateInnerBrick(r, decodeMorton3D(closestLeaf->brick->mortonCode), closestLeaf, intersectionsCount, bricksSize, brickDimensionSize, gridResolution);
	}

	res->x = 0;
	res->y = 0;
	atomicAdd(&res->z, acc);
}

__global__
void genLeafs(int numOfLeafs, int numOfNodes, int brickDimensionSize, Node *nodes, Brick *bricks) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numOfNodes)
		return;

	int currentNodeIndex = numOfLeafs - 1 + index;

	nodes[currentNodeIndex].id = currentNodeIndex;
	nodes[currentNodeIndex].brick = &bricks[index];
	nodes[currentNodeIndex].isLeaf = true;
	nodes[currentNodeIndex].aabb = generateLeafAABB(*nodes[currentNodeIndex].brick, brickDimensionSize);
}

__host__ __device__
int delta(int x, int y, int numObjects, Brick *bricks) {
	if (x >= 0 && x <= numObjects - 1 && y >= 0 && y <= numObjects - 1)
		return clz(bricks[x].mortonCode ^ bricks[y].mortonCode);

	return -1;
}

__host__ __device__
inline int sign(float x) {
	return x >= 0 ? 1 : -1;
}

__host__ __device__
glm::ivec2 determineRange(int numObjects, int idx, Brick *bricks) {
	int d = sign(delta(idx, idx + 1, numObjects, bricks) - delta(idx, idx - 1, numObjects, bricks));
	int dmin = delta(idx, idx - d, numObjects, bricks);
	int lmax = 2;

	while (delta(idx, idx + lmax * d, numObjects, bricks) > dmin)
		lmax = lmax * 2;

	int l = 0;
	for (int t = lmax / 2; t >= 1; t /= 2)
		if (delta(idx, idx + (l + t)*d, numObjects, bricks) > dmin)
			l += t;

	int j = idx + l * d;
	glm::ivec2 range;
	range.x = glm::min(idx, j);
	range.y = glm::max(idx, j);

	return range;
}

//Based on Tero Karras "Thinking Parallel, Part III: Tree Construction on the GPU"
__host__ __device__
int generateSplitPositions(int first, int last, Brick *bricks) {
	// Identical Morton codes => split the range in the middle.
	int firstCode = bricks[first].mortonCode;
	int lastCode = bricks[last].mortonCode;

	if (firstCode == lastCode)
		return (first + last) >> 1;

	// Calculate the number of highest bits that are the same
	// for all objects, using the count-leading-zeros intrinsic.
	int commonPrefix = clz(firstCode ^ lastCode);

	// Use binary search to find where the next bit differs.
	// Specifically, we are looking for the highest object that
	// shares more than commonPrefix bits with the first one.
	int split = first; // initial guess
	int step = last - first;

	do {
		step = (step + 1) >> 1; // exponential decrease
		int newSplit = split + step; // proposed new position

		if (newSplit < last) {
			int splitCode = bricks[newSplit].mortonCode;
			int splitPrefix = clz(firstCode ^ splitCode);
			if (splitPrefix > commonPrefix)
				split = newSplit; // accept proposal
		}
	} while (step > 1);

	return split;
}

//Based on Karras "Thinking Parallel, Part III: Tree Construction on the GPU"
__global__
void genInternalNodes(Node *nodes, int numOfLeafs, Brick *bricks) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numOfLeafs - 1)
		return;

	// Find out which range of objects the node corresponds to.
	// (This is where the magic happens!)
	glm::ivec2 range = determineRange(numOfLeafs, index, bricks);
	int first = range.x;
	int last = range.y;

	// Determine where to split the range.
	int split = generateSplitPositions(first, last, bricks);

	// Select childA.
	Node* childA;
	if (split == first) { //split + numOfLeafs 
		childA = &nodes[(numOfLeafs - 1) + split];
		childA->isLeaf = true;
	}
	else
		childA = &nodes[split];

	// Select childB.
	Node* childB;
	if (split + 1 == last) {
		childB = &nodes[(numOfLeafs - 1) + split + 1];
		childB->isLeaf = true;
	}
	else
		childB = &nodes[split + 1];

	// Record parent-child relationships.
	childA->parent = &nodes[index];
	childB->parent = &nodes[index];
	nodes[index].id = index;
	nodes[index].leftChild = childA;
	nodes[index].rightChild = childB;
	nodes[index].aabb = AABB{ glm::vec3(9999999), glm::vec3(-9999999) };
}

/*
	Returns the AABB containing both a1 and a2.
*/
__host__ __device__
AABB encapsuleAABB(AABB a1, AABB a2) {
	glm::vec3 min = glm::min(a1.min, a2.min);
	glm::vec3 max = glm::max(a1.max, a2.max);

	return AABB{ min, max };
}

/*
	Returns the AABB containing both a1 and a2.
*/
__device__
void encapsuleAABBCUDA(AABB *parent, AABB *child) {
	atomicMin(&(parent->min.x), child->min.x);
	atomicMin(&(parent->min.y), child->min.y);
	atomicMin(&(parent->min.z), child->min.z);

	atomicMax(&(parent->max.x), child->max.x);
	atomicMax(&(parent->max.y), child->max.y);
	atomicMax(&(parent->max.z), child->max.z);
}

__global__
void propagateAABB(int numOfLeafs, Node *nodes) {
	int totalNodes = numOfLeafs + numOfLeafs - 1;
	int index = blockIdx.x * blockDim.x + threadIdx.x + numOfLeafs;
	if (index >= totalNodes)
		return;

	Node *node = &nodes[index];

	while (node != nullptr) {
		// if we reached root, break.
		if (node->parent == nullptr)
			break;
		encapsuleAABBCUDA(&node->parent->aabb, &node->aabb);

		node = node->parent;
	}
}

void printTree(Node *nodes, int nodesSize) {
	for (int i = 0; i < nodesSize; i++) {
		std::cout << "{" << std::endl;
		std::cout << "id: " << nodes[i].id << std::endl;
		if (!nodes[i].isLeaf) {
			std::cout << "childA: " << nodes[i].leftChild->id << std::endl;
			std::cout << "childB: " << nodes[i].rightChild->id << std::endl;
		}
		else {
			std::cout << "parent: " << nodes[i].parent->id << std::endl;
		}
		printVec3(nodes[i].aabb.min);
		printVec3(nodes[i].aabb.max);
		std::cout << "}" << std::endl;
	}
}

__global__
void assignMortonCode(glm::ivec3 bricksResolution, int brickDimensionSize, Brick *bricks, int bricksSize) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= bricksSize)
		return;

	glm::ivec3 coord;
	to3D(index, bricksResolution.x, bricksResolution.y, coord);
	coord *= brickDimensionSize;

	if (!bricks[index].isEmpty) {
		bricks[index].mortonCode = encodeMorton3D(coord);
	}
}

__global__
void findAndMarkEmptyBricks(Brick *bricks, int bricksSize, int brickGridSize) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= bricksSize)
		return;

	for (int i = 0; i < brickGridSize; i++) {
		if (bricks[index].grid[i] != 0.0f) {
			bricks[index].isEmpty = false;
			break;
		}
	}

}

__global__
void subdivideIntoBricks(Brick * bricks, int bricksSize, glm::vec3 bricksResolution, int brickDimensionSize, glm::vec3 gridResolution, float *grid, int gridSize) {
	glm::ivec3 gridPos;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= bricksSize)
		return;

	to3D(index, bricksResolution.x, bricksResolution.y, gridPos);
	gridPos = gridPos * brickDimensionSize;

	//x, y and z are in grid coordinates
	for (int x = gridPos.x; x < gridPos.x + brickDimensionSize; x++)
		for (int y = gridPos.y; y < gridPos.y + brickDimensionSize; y++)
			for (int z = gridPos.z; z < gridPos.z + brickDimensionSize; z++) {

				int gridIndex = to1D(gridResolution.x, gridResolution.y, x, y, z);
				int localBrickIndex = to1D(brickDimensionSize, brickDimensionSize, intmod(glm::ivec3(x, y, z), brickDimensionSize));
				if (gridIndex >= gridSize) {
					bricks[index].grid[localBrickIndex] = 0.0f;
					continue;
				}
				bricks[index].grid[localBrickIndex] = grid[gridIndex];
			}
}

void referenceLBVHMeasurePerformance(float *data, glm::ivec3 gridRes, std::stringstream &ss, float *results) {
	float brickDimensionSize = 8;
	int brickGridSize = brickDimensionSize * brickDimensionSize * brickDimensionSize;
	glm::vec3 bricksResolution = glm::vec3(std::ceil(gridRes.x / brickDimensionSize), std::ceil(gridRes.y / brickDimensionSize), std::ceil(gridRes.z / brickDimensionSize));
	int gridSize = gridRes.x * gridRes.y * gridRes.z;
	int bricksSize = bricksResolution.x * bricksResolution.y * bricksResolution.z;
	float *grid;
	Brick *bricks;
	Node *nodes;
	float totalElapsedTime = 0;
	float milliseconds = 0;
	int blockSize;
	int numBlocks;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMallocManaged(&bricks, bricksSize * sizeof(Brick));
	HANDLE_ERROR(cudaPeekAtLastError());

	cudaMalloc(&grid, gridSize * sizeof(float));
	HANDLE_ERROR(cudaPeekAtLastError());

	cudaMemcpy(grid, data, gridSize * sizeof(float), cudaMemcpyHostToDevice);
	HANDLE_ERROR(cudaPeekAtLastError());

	ss << "Paper pointer tree" << std::endl;
	ss << std::endl;
	ss << "Generate tree (milliseconds)" << std::endl;
	ss << std::endl;

	cudaEventRecord(start);
	blockSize = 256;
	numBlocks = (bricksSize + blockSize - 1) / blockSize;
	subdivideIntoBricks<<<numBlocks, blockSize >>>(bricks, bricksSize, bricksResolution, brickDimensionSize, gridRes, grid, gridSize);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	ss << "subdivide into bricks: \t\t" << milliseconds << std::endl;
	totalElapsedTime += milliseconds;
	HANDLE_ERROR(cudaPeekAtLastError());
	cudaFree(grid);
	HANDLE_ERROR(cudaPeekAtLastError());

	//=====================
	//1. Mark empty bricks
	for (int b = 0; b < bricksSize; b++)
		bricks[b].isEmpty = true;

	cudaEventRecord(start);
	blockSize = 256;
	numBlocks = (bricksSize + blockSize - 1) / blockSize;
	findAndMarkEmptyBricks<<<numBlocks, blockSize >>>(bricks, bricksSize, brickGridSize);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	ss << "find and mark empty bricks: " << milliseconds << std::endl;
	totalElapsedTime += milliseconds;
	HANDLE_ERROR(cudaPeekAtLastError());

	//=====================
	//2. Assign morton codes
	cudaEventRecord(start);
	blockSize = 256;
	numBlocks = (bricksSize + blockSize - 1) / blockSize;
	assignMortonCode<<<numBlocks, blockSize >>>(bricksResolution, brickDimensionSize, bricks, bricksSize);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	ss << "assign morton codes: \t\t" << milliseconds << std::endl;
	totalElapsedTime += milliseconds;
	HANDLE_ERROR(cudaPeekAtLastError());

	//=====================
	//3. Sort and compact
	cudaEventRecord(start);
	thrust::sort(thrust::device, bricks, bricks + bricksSize, BrickCmp());
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	ss << "thrust sort: \t\t\t\t" << milliseconds << std::endl;
	totalElapsedTime += milliseconds;
	Brick *newEnd = thrust::remove_if(thrust::device, bricks, bricks + bricksSize, isBrickEmpty());
	int nonEmptyBricks = (newEnd - bricks);
	HANDLE_ERROR(cudaPeekAtLastError());

	//=====================
	//4. Generate hierarchy
	int nodesSize = nonEmptyBricks + nonEmptyBricks - 1;
	cudaMallocManaged(&nodes, nodesSize * sizeof(Node));
	HANDLE_ERROR(cudaPeekAtLastError());
	
	//=====================
	//4.1 Construct leaf nodes.
	cudaEventRecord(start);
	blockSize = 256;
	numBlocks = (nonEmptyBricks + blockSize - 1) / blockSize;
	genLeafs <<<numBlocks, blockSize >>> (nonEmptyBricks, nodesSize, brickDimensionSize, nodes, bricks);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	ss << "generate leafs: \t\t\t" << milliseconds << std::endl;
	totalElapsedTime += milliseconds;
	HANDLE_ERROR(cudaPeekAtLastError());

	//=====================
	//4.2 Construct internal nodes.
	cudaEventRecord(start);
	blockSize = 256;
	numBlocks = (nonEmptyBricks + blockSize - 1) / blockSize;
	genInternalNodes <<<numBlocks, blockSize >>> (nodes, nonEmptyBricks, bricks);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	ss << "generate tree: \t\t\t\t" << milliseconds << std::endl;
	totalElapsedTime += milliseconds;
	HANDLE_ERROR(cudaPeekAtLastError());

	//=====================
	//4.3 Propagate AABB
	cudaEventRecord(start);
	blockSize = 256;
	numBlocks = (nonEmptyBricks * 2 + blockSize - 1) / blockSize;
	propagateAABB <<<numBlocks, blockSize >>> (nonEmptyBricks, nodes);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	ss << "propagate AABB: \t\t\t" << milliseconds << std::endl;
	totalElapsedTime += milliseconds;
	HANDLE_ERROR(cudaPeekAtLastError());

	glm::vec3 *res;
	cudaMallocManaged(&res, 1 * sizeof(glm::vec3));
	cudaDeviceSynchronize();
	res->x = 0;
	res->y = 0;
	res->z = 0;

	Ray ray;
	float avgTraverseTime = 0;
	float acc = 0;
	
	const int numberOfFaces = 6;
	glm::vec3 faceAxis[numberOfFaces] = { glm::vec3(1, 1, 0), glm::vec3(0, 1, 1), glm::vec3(1, 1, 0), glm::vec3(0, 1, 1), glm::vec3(1, 0, 1), glm::vec3(1, 0, 1) };
	glm::vec3 rayDirection[numberOfFaces] = { glm::vec3(0, 0, 1), glm::vec3(-1, 0, 0), glm::vec3(0, 0, -1), glm::vec3(1, 0, 0), glm::vec3(0, -1, 0), glm::vec3(0, 1, 0) };

	//For each face of the AABB compute the average traversal time of all pixels
	for (int v = 0; v < numberOfFaces; v++) {
		glm::vec3 screenRes;
		int ind = 0;
		for (int i = 0; i < 3; i++) {
			if (faceAxis[v][i] != 0)
				screenRes[ind++] = gridRes[i];
		}
		screenRes.z = screenRes.x * screenRes.y;
		ray.direction = rayDirection[v];

		float currentFaceAvgTraverseTime = 0;

		cudaEventRecord(start);
		blockSize = 1024;
		numBlocks = (screenRes.z + blockSize - 1) / blockSize;
		traversePaper << <numBlocks, blockSize >> > (ray, faceAxis[v], nodes, brickGridSize, brickDimensionSize, gridRes, res, screenRes);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&currentFaceAvgTraverseTime, start, stop);
		cudaDeviceSynchronize();

		currentFaceAvgTraverseTime /= screenRes.x * screenRes.y;
		avgTraverseTime += currentFaceAvgTraverseTime;
	}

	avgTraverseTime = avgTraverseTime / numberOfFaces;
	acc += res->z;
	HANDLE_ERROR(cudaPeekAtLastError());

	ss << "Total generation time: \t\t" << totalElapsedTime << std::endl;
	ss << std::endl;
	ss << "Accumulated density: " << acc << std::endl;
	ss << "Avg. traverse time: " << avgTraverseTime << std::endl;
	ss << std::endl;
	ss << "Number of nodes: " << nodesSize << std::endl;
	ss << "Brick size: " << brickGridSize << std::endl;

	//"Number of nodes", "Number of empty nodes", 
	//"Nodes Mem. consumption (KB)", "Offset Mem. consumption (KB)", "MortonCode Mem. consumption (KB)", "Total Mem. consumption (MB)", 
	// "Tree construction time (ms)", "Tree avg. Traversal time (ms)", "Avg. accumulated density"
	results[0] = results[0] + nodesSize;
	results[1] = results[1] + 0;
	float nodeSizeKB = sizeof(Node) * nodesSize / 1024.0f;
	float nonEmptyBricksKB = sizeof(int) * nonEmptyBricks / 1024.0f;
	results[2] = results[2] + nodeSizeKB;
	results[3] = results[3] + 0;
	results[4] = results[4] + nonEmptyBricksKB;
	results[5] = results[5] + (nodeSizeKB + 0 + nonEmptyBricksKB) / 1024.0f;
	results[6] = results[6] + totalElapsedTime;
	results[7] = results[7] + avgTraverseTime;
	results[8] = results[8] + acc;


	cudaFree(bricks);
	HANDLE_ERROR(cudaPeekAtLastError());
	cudaFree(nodes);
	HANDLE_ERROR(cudaPeekAtLastError());
}