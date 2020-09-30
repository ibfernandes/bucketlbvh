#pragma once
#include <iostream>
#include <fstream>
#include "BucketLBVH.h"
#include "ReferenceLBVH.h"
#include <assert.h>
#include <Vector>
#include <limits>
#include <cstddef>
#include <glm/glm.hpp>
#include <bitset>
#include "Timer.h"
#include <thread>


float calculateEmptyRatio2(float *grid, int size) {
	int count = 0;
	for (int i = 0; i < size; i++)
		if (grid[i] != 0.0f)
			count++;

	return float(count) / float(size);
}

/*
	Generates a grid with all voxels filled with 1.0's
*/
void generateNonEmptyGrid(float *grid, glm::ivec3 resolution) {
	for (int i = 0; i < resolution.x * resolution.y * resolution.z; i++)
		grid[i] = 1.0f;
}

void measurementTests() {
	glm::vec3 gridResolution = glm::vec3(256, 256, 256);
	float *grid = new float[gridResolution.x * gridResolution.y * gridResolution.z];
	generateNonEmptyGrid(grid , gridResolution);

	Timer *t = new Timer();

	std::cout << "Grid resolution: ";
	printVec3(gridResolution);
	float empt = calculateEmptyRatio2(grid, gridResolution.x * gridResolution.y * gridResolution.z);
	std::cout << "Emptiness: " << (1 - empt) << std::endl;

	float constructionTime1, constructionTime2;

	t->startTimer();
	BucketLBVH *lbvh2 = new BucketLBVH(grid, gridResolution);
	t->endTimer();
	constructionTime1 = t->getSeconds();

	t->startTimer();
	ReferenceLBVH *lbvh3 = new ReferenceLBVH(grid, gridResolution);
	t->endTimer();
	constructionTime2 = t->getSeconds();

	//glm::vec3 origin = glm::vec3((res.x / 2.0f) + 0.2f, (res.y / 2.0f) + 0.2f, 0.5);
	glm::vec3 origin = glm::vec3(1.0 + 0.2f, 84.0f + 0.2f, -1.0);
	glm::vec3 direction= glm::vec3(0, 0, 1);
	glm::vec3 result;

	Ray *r = new Ray(origin, direction);

	std::cout << "------------------------" << std::endl;
	float bTime = 0;
	float pTime = 0;

	t->startTimer();
	result = lbvh2->traverse(*r);
	t->endTimer();
	std::cout << "Bucket binary tree approach" << std::endl;
	std::cout << "Accumulated " << result.z << std::endl;
	std::cout << "hit [" << result.x << ", " << result.y << "]" << std::endl;
	bTime = t->getSeconds();
	t->printlnNanoSeconds();
	std::cout << std::endl;

	t->startTimer();
	result = lbvh3->traverse(Ray(origin, direction));
	t->endTimer();
	std::cout << "Linear BVH paper approach" << std::endl;
	std::cout << "Accumulated " << result.z << std::endl;
	//std::cout << "hit [" << result.x << ", " << result.y << "]" << std::endl;
	pTime = t->getSeconds();
	t->printlnNanoSeconds();
	std::cout << "------------------------" << std::endl;
	std::cout << "Performance ratio: " << bTime / pTime << std::endl;

	std::cout << "------------------------" << std::endl;
	std::cout << "Construction time" << std::endl;
	std::cout << "Bucket time: " << constructionTime1 << std::endl;
	std::cout << "Paper time: " << constructionTime2 << std::endl;
	std::cout << "------------------------" << std::endl;

	float avgBucket = 0;
	float avgPaper = 0;
	int numberOfCasesWhereILost = 0;
	int numberOfCasesItWasAccZero = 0;
	for(int x = 0; x < gridResolution.x; x++)
		for (int y = 0; y < gridResolution.y; y++) {
			origin = glm::vec3(x + 0.2f, y + 0.2f, 0.5);
			r = new Ray(lbvh2->getWCS(origin), direction);
			glm::vec3 val1, val2;
			float t1, t2;

			//Bucket
			t->startTimer();
			val1 = lbvh2->traverse(*r);
			t->endTimer();
			t1 = t->getSeconds();
			avgBucket += t1;

			//Paper
			t->startTimer();
			val2 = lbvh3->traverse(Ray(origin, direction));
			t->endTimer();
			t2 = t->getSeconds();
			avgPaper += t2;

			if (t1 > t2) {
				numberOfCasesWhereILost++;
				if (val1.z == 0)
					numberOfCasesItWasAccZero++;
			}

			if (val1.z != val2.z) {
				std::cout << "bucket: " << val1.z << std::endl;
				std::cout << "paper:  " << val2.z << std::endl;
				std::cout << "ERROR" << std::endl;
			}
		}

	std::cout << "------------------------" << std::endl;
	std::cout << "whole grid testing" << std::endl;
	std::cout << "Bucket approach" << std::endl;
	std::cout << "Avg time: " << avgBucket / (gridResolution.x * gridResolution.y) << std::endl;
	std::cout << "\t ~" << std::endl;

	std::cout << "Paper approach" << std::endl;
	std::cout << "Avg time: " << avgPaper / (gridResolution.x * gridResolution.y) << std::endl;
	std::cout << "------------------------" << std::endl;

	std::cout << "Cases lost: " << numberOfCasesWhereILost << "/" << gridResolution.x * gridResolution.y <<std::endl;
	std::cout << "Cases lost with acc 0: " << numberOfCasesItWasAccZero << "/" << gridResolution.x * gridResolution.y <<std::endl;
	std::cout << '\a';
}
