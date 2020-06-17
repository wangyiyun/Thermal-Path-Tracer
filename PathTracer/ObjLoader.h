#pragma once
#include <iostream>
#include <vector>
#include <string>
#include "cutil_math.h"

struct Scene
{
	int vertsNum;	// vertices number of the scene, also the number of uvs and normals
	float3* verts;	// pointer for all vertices
	int objsNum;
	int* objsInfo;		// [objVertsNum, matNum, uvTexNum, ambTexNum]
	float2* uvs;
	float3* normals;
	std::vector<std::string> objNames;
	void FreeScene()
	{
		delete[] verts;
		delete[] objsInfo;
		delete[] uvs;
		delete[] normals;
	}
};

bool LoadObj(
	const char* path,
	Scene &scene
)
{
	std::vector<int> vertexIndices, uvIndices, normalIndices;
	std::vector<int> temp_object_indices;
	std::vector<float3> temp_vertices;
	std::vector<float2> temp_uvs;
	std::vector<float3> temp_normals;
	FILE* file = fopen(path, "r");
	if (file == NULL) {
		std::cout << "ERROR: Can't open file: " << path << std::endl;
		return false;
	}

	while (true)
	{
		char lineHeader[128];
		// read the first word of the line
		int res = fscanf(file, "%s", lineHeader);
		if (res == EOF)
			break; // EOF = End Of File. Quit the loop.

		// else : parse lineHeader
		if (strcmp(lineHeader, "o") == 0) {
			char objectName[128];
			fscanf(file, "%s\n", objectName);
			scene.objNames.push_back(objectName);
			temp_object_indices.push_back(vertexIndices.size());
		}
		else if (strcmp(lineHeader, "v") == 0) {
			float3 vertex;
			fscanf(file, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z);
			temp_vertices.push_back(vertex);
		}
		else if (strcmp(lineHeader, "vt") == 0) {
			float2 uv;
			fscanf(file, "%f %f\n", &uv.x, &uv.y);
			temp_uvs.push_back(uv);
		}
		else if (strcmp(lineHeader, "vn") == 0) {
			float3 normal;
			fscanf(file, "%f %f %f\n", &normal.x, &normal.y, &normal.z);
			temp_normals.push_back(normal);
		}
		else if (strcmp(lineHeader, "f") == 0) {
			std::string vertex1, vertex2, vertex3;
			unsigned int vertexIndex[3], uvIndex[3], normalIndex[3];
			int matches = fscanf(file, "%d/%d/%d %d/%d/%d %d/%d/%d\n", &vertexIndex[0], &uvIndex[0], &normalIndex[0], &vertexIndex[1], &uvIndex[1], &normalIndex[1], &vertexIndex[2], &uvIndex[2], &normalIndex[2]);
			if (matches != 9) {
				printf("File can't be read by our simple parser : ( Try exporting with other options\n");
				return false;
			}
			vertexIndices.push_back(vertexIndex[0]);
			vertexIndices.push_back(vertexIndex[1]);
			vertexIndices.push_back(vertexIndex[2]);
			uvIndices.push_back(uvIndex[0]);
			uvIndices.push_back(uvIndex[1]);
			uvIndices.push_back(uvIndex[2]);
			// normal index are same!!
			normalIndices.push_back(normalIndex[0]);
			//normalIndices.push_back(normalIndex[1]);
			//normalIndices.push_back(normalIndex[2]);
		}
	}
	scene.objsNum = temp_object_indices.size();
	scene.objsInfo = new int[4 * scene.objsNum];
	for (unsigned int i = 0; i < temp_object_indices.size(); i++)
	{
		// [objVertsNum, matNum, uvTexNum, ambTexNum]
		scene.objsInfo[i * 4] = temp_object_indices[i];	//objVertsNum
		scene.objsInfo[i * 4 + 1] = -1;	//matNum
		scene.objsInfo[i * 4 + 2] = -1;	//uvTexNum
		scene.objsInfo[i * 4 + 3] = -1;	//ambTexNum
	}
	std::cout << "verts num: " << vertexIndices.size() << std::endl;
	scene.vertsNum = vertexIndices.size();
	scene.verts = new float3[scene.vertsNum];
	for (unsigned int i = 0; i < vertexIndices.size(); i++)
	{
		unsigned int vertexIndex = vertexIndices[i];
		float3 vertex = temp_vertices[vertexIndex - 1]*100.0f;
		vertex.x += 150.0f;
		//vertex.y += 50.0f;
		vertex.z += 500.0f;
		scene.verts[i] = vertex;
		//std::cout << vertex.x << " " << vertex.y << " " << vertex.z << std::endl;
	}
	//std::cout << "uvs num: " << uvIndices.size() << std::endl;
	scene.uvs = new float2[scene.vertsNum];
	for (unsigned int i = 0; i < uvIndices.size(); i++)
	{
		unsigned int uvIndex = uvIndices[i];
		float2 uv = temp_uvs[uvIndex - 1];
		scene.uvs[i] = uv;
		//std::cout << uv.x << " " << uv.y << std::endl;
	}
	//std::cout << "normals num: " << normalIndices.size() << std::endl;
	// normals num == faces num
	scene.normals = new float3[scene.vertsNum/3];
	for (unsigned int i = 0; i < normalIndices.size(); i++)
	{
		unsigned int normalIndex = normalIndices[i];
		float3 normal = temp_normals[normalIndex - 1];
		scene.normals[i] = normal;
		//std::cout << normal.x << " " << normal.y << " " << normal.z << std::endl;
	}
	return true;
	
}