#pragma once
#define CUDA_NO_HALF
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Texture3D.h"
#include "ReferenceLBVHCUDA.cuh"
#include "BucketLBVHCUDA.cuh"
#include <iostream>
#include <fstream>

std::vector<Texture3D*> textures;

float calculateEmptyRatio(float *grid, int size) {
	int count = 0;
	for (int i = 0; i < size; i++)
		if (grid[i] == 0.0f)
			count++;

	return float(count) / float(size);
}

void appendResultsToFile(std::string filename, std::string text) {
	std::ofstream file;
	file.open("tests\\" + filename + ".txt", std::ios_base::app);

	file << text << "\n";

	file.close();
}

void appendResultsToExcel(std::string filename, std::string text) {
	std::ofstream file;
	file.open("tests\\" + filename + ".csv", std::ios_base::app);

	file << text << "\n";

	file.close();
}

void formatHeader(std::stringstream &ss, std::string fileName, glm::ivec3 res, float emptiness) {
	ss << ";" << "Grid name" << ";" << "Resolution" << ";" << "Emptiness" << ";" << "Grid size (MB)" << ";" << "" << ";" << std::endl;
	ss << ";" << fileName << ";" << res.x << "x" << res.y << "x" << res.z << "= [" << res.x * res.y * res.z << "]" << ";" << emptiness << "%;" << ((sizeof(float) * res.x * res.y * res.z) / 1024.0f) / 1024.0f  << std::endl;
	ss << ";" << ";" << ";" << ";" << ";" << ";" << std::endl;
}

void formatData(std::stringstream &ss, std::string *header, std::string *rowNames, int numberOfColumns, int numberOfRows, float *columns) {
	//First row is the header
	for (int c = 0; c < numberOfColumns; c++) {
		ss << header[c] << ";";
	}
	ss << "\n";

	for (int r = 1; r < numberOfRows; r++) {
		for (int c = 0; c < numberOfColumns; c++) {
			if (c == 0)
				ss << rowNames[r] << "; ";
			else
				// -1 for firest row header and first column labels
				ss << columns[(numberOfRows - 1) * (c - 1) + (r - 1)] << "; ";
		}
		ss << "\n";
	}
}

void performCUDATests() {
	const int numberOfRounds = 10;
	int const numberOfGrids = 6;
	int const numberOfBucketSizes = 10;
	std::string gridNames[numberOfGrids] = {"dragonHavard.vdb", "explosion.vdb", "fireball.vdb", "wdas_cloud_eighth.vdb", "wdas_cloud_quarter.vdb", "bunny_cloud.vdb"};
	int bucketSizes[numberOfBucketSizes] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};

	//Generate column labels
	const int numberOfColumns = numberOfBucketSizes + 2;
	std::string header[numberOfColumns];
	header[0] = "";
	header[1] = "Reference";
	for (int i = 2; i < numberOfColumns; i++) 
		header[i] = "Bucket (b = " + std::to_string(bucketSizes[i - 2]) + ")";
	
	//Generate row labels
	const int dataPerColumn = 9;
	const int numberOfRows = dataPerColumn + 1;
	std::string rowNames[numberOfRows] = {"", "Number of nodes", "Number of empty nodes", "Nodes Mem. consumption (KB)",
		"Offset Mem. consumption (KB)", "MortonCode Mem. consumption (KB)", "Total Mem. consumption (MB)", "Tree construction time (ms)",
		"Tree avg. Traversal time (ms)",  "Avg. accumulated density" };

	Timer *t = new Timer();

	for (int g = 0; g < numberOfGrids; g++) {
		Texture3D *currentTexture = new Texture3D("vdb/" + gridNames[g]);
		float *grid = currentTexture->floatData;
		glm::vec3 gridRes = currentTexture->getResolution();

		std::stringstream excelStringData;
		std::stringstream txtStringData;
		float emptiness = calculateEmptyRatio(grid, gridRes.x * gridRes.y * gridRes.z) * 100;
		txtStringData << "==========================================" << std::endl;
		txtStringData << "Grid name: " << gridNames[g] << std::endl;
		txtStringData << "Grid resolution: " << "[" << gridRes.x << ", " << gridRes.y << ", " << gridRes.z << "]" << std::endl;
		txtStringData << "Emptiness: " << emptiness << "% " << std::endl;
		txtStringData << "==========================================" << std::endl;
		appendResultsToFile(gridNames[g], txtStringData.str());
		txtStringData.str("");

		float column[dataPerColumn * (numberOfBucketSizes + 1)];
		for (int c = 0; c < dataPerColumn * (numberOfBucketSizes + 1); c++)
			column[c] = 0;

		for (int r = 0; r < numberOfRounds; r++) {
			std::stringstream paper;
			referenceLBVHMeasurePerformance(currentTexture->floatData, gridRes, paper, &column[0]);
			txtStringData << "------------------------------------------" << std::endl;
			txtStringData << paper.str();
			std::cout << "Grid: \"" << gridNames[g] << "\" reference round " << r << " done" << std::endl;
		}

		for (int r = 0; r < numberOfRounds; r++) {
			for (int b = 0; b < numberOfBucketSizes; b++) {
				txtStringData << std::endl << std::endl;
				bucketLBVHMeasurePerformance(currentTexture->floatData, gridRes, bucketSizes[b], txtStringData, &column[(b + 1) * dataPerColumn]);
				txtStringData << "------------------------------------------" << std::endl;
				txtStringData << std::endl;
			}
			std::cout << "Grid: \"" << gridNames[g] << "\" bucket round " << r << " done" << std::endl;
		}

		std::cout << "Grid: \"" << gridNames[g] << "\" done" << std::endl;

		for (int c = 0; c < dataPerColumn * (numberOfBucketSizes + 1); c++)
			column[c] = column[c] / float(numberOfRounds);

		formatHeader(excelStringData, gridNames[g], gridRes, emptiness);
		formatData(excelStringData, header, rowNames, numberOfColumns, numberOfRows, column);
		appendResultsToExcel(gridNames[g], excelStringData.str());
		excelStringData.str("");

		appendResultsToFile(gridNames[g], txtStringData.str());
		txtStringData.str("");
		delete currentTexture;
	}
}

void generateTestGrid(Texture3D *tex3D, glm::ivec3 resolution) {
	tex3D->floatData = new float[resolution.x * resolution.y * resolution.z];
	for (int i = 0; i < resolution.x * resolution.y * resolution.z; i++)
		tex3D->floatData[i] = 1.0f;
}

void performValidationGrid() {
	Texture3D *tex3D = new Texture3D();
	glm::ivec3 gridRes = glm::ivec3(16, 8, 4);
	generateTestGrid(tex3D, gridRes);

	int bucketSize = 8;
	float results[5];
	std::stringstream ss;
	//bucketLBVHMeasurePerformance(tex3D->floatData, gridRes, bucketSize, ss, results);
	//referenceLBVHMeasurePerformance(tex3D->floatData, gridRes, ss, results);
	std::cout << ss.str();
}

int main(void) {
	performCUDATests();
	return 0;
}