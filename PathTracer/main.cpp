#include <windows.h>
// OpenGL
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/gl.h>
// OpenGL math lib
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <fstream>
#include <string>

#include "cuda_gl_interop.h"

#include "cutil_math.h"
#include <curand.h>
#include <curand_kernel.h>

#include "include/FreeImage.h"
#include "ObjLoader.h"
#include "TexLoader.h"

#include "imgui_impl_glut.h"

const int width = 640;	// width of the figure
const int height = 480;	// height of the figure
unsigned int frame = 0;	// a frame counter, used as a random seed

// For OpenGL
GLuint pbo = -1;	// pxiel buffer object, place for OpenGL and CUDA to switch data and display the result
GLuint textureID = -1;	// OpenGL texture to display the result

// For CUDA
struct cudaGraphicsResource* resource;	// pointer to the teturned object handle
float3* result;	// place for CUDA output
float3* accu;	// place for accumulate all frame result
curandState* randState;

// Implement of this function is in kernel.cu
extern "C" void launch_kernel(float3*, float3*, curandState*, unsigned int, unsigned int, unsigned int, bool, int, 
	int, float3*, int, int*, float2*, float3*,
	int, int*, float3*,
	int, float*);

// Auto output
#define OUTPUT_FRAME_NUM 256
bool camAtRight = true;	// pos of the camera, true for right side, false for left side
int waveNum = 0;
int type = 0;	// 0: emi+refl, 1: emi, 2: refl

// host
Scene SceneData;
// device
float3* scene_verts; // the cuda device pointer that points to the uploaded triangles
int* scene_objs;
float2* scene_uvs;
float3* scene_normals;

// host
Texture tex_mug_normal;
Texture tex_table_ambient;
std::vector<Texture> textures;
int texNum;
int tex_data_size;	// pixels num of all textures
int* h_tex_wh;		// [w0,h0,w1,h1...] size = texNum * 2
float3* h_tex_data;	// pixels color of all textures

// device
int* d_tex_wh;	//[w0,h0,w1,h1...] size = texNum * 2
float3* d_tex_data;
float* d_emiList;

// func
void initCuda();
void AutoOutput();
void GenFileName(std::string *s);

void draw_gui()
{
	ImGui_ImplGlut_NewFrame();
	//ImGui::ShowDemoWindow();
	ImGui::Begin("Parameter setting");
	if (ImGui::Checkbox("Cam at right", &camAtRight))
	{
		initCuda();
	}
	if (ImGui::RadioButton("emi+refl", &type, 0))
	{
		initCuda();
	}
	ImGui::SameLine();
	if (ImGui::RadioButton("emi", &type, 1))
	{
		initCuda();
	}
	ImGui::SameLine();
	if (ImGui::RadioButton("refl", &type, 2))
	{
		initCuda();
	}
	const char* materials[] = { "mat_human", "mat_marble", "mat_paint", "mat_glass", "mat_rubber", "mat_brass", "mat_road", "mat_al", "mat_al2o3", "mat_brick" };
	ImGui::BeginTabBar("Object Info");
	if (ImGui::BeginTabItem("emissivity"))
	{
		for (unsigned int i = 0; i < SceneData.objsNum; i++)
		{
			// [objVertsNum, matNum, normalTexNum, ambientTexNum, temperature, emiSource]
			
			std::string emiRes0, emiRes1, emiRes2;
			emiRes0 = SceneData.objNames[i] + " from mat";
			emiRes1 = SceneData.objNames[i] + " from tex";
			emiRes2 = SceneData.objNames[i] + " from value";
			const char* er0 = emiRes0.c_str();
			const char* er1 = emiRes1.c_str();
			const char* er2 = emiRes2.c_str();

			std::string currentMat;
			currentMat += materials[SceneData.objsInfo[i * 6 + 1]];
			currentMat += " for " + SceneData.objNames[i];
			const char* cm = currentMat.c_str();

			std::string currentTex;
			currentTex += "Emi tex for " + SceneData.objNames[i];
			const char* ct = currentTex.c_str();
			
			ImGui::RadioButton(er0, &SceneData.objsInfo[i * 6 + 5], 0);
			ImGui::SameLine();
			ImGui::RadioButton(er1, &SceneData.objsInfo[i * 6 + 5], 1);
			ImGui::SameLine();
			ImGui::RadioButton(er2, &SceneData.objsInfo[i * 6 + 5], 2);
			if (SceneData.objsInfo[i * 6 + 5] == 0)
			{
				ImGui::SliderInt(cm, &SceneData.objsInfo[i * 6 + 1], 0, 9);
			}
			else if (SceneData.objsInfo[i * 6 + 5] == 1)
			{
				ImGui::SliderInt(ct, &SceneData.objsInfo[i * 6 + 3], -1, 1);
			}
			else
			{
				ImGui::InputFloat("emi value", &SceneData.emiList[i]);
			}
		}
		ImGui::EndTabItem();
	}
	if (ImGui::BeginTabItem("temperature"))
	{
		for (unsigned int i = 0; i < SceneData.objsNum; i++)
		{
			// [objVertsNum, matNum, normalTexNum, ambientTexNum, temperature, emiSource]
			const char* objName = SceneData.objNames[i].c_str();
			ImGui::SliderInt(objName, &SceneData.objsInfo[i * 6 + 4], 0, 100);

		}
		ImGui::EndTabItem();
	}
	if (ImGui::BeginTabItem("normal texture"))
	{
		for (unsigned int i = 0; i < SceneData.objsNum; i++)
		{
			const char* objName = SceneData.objNames[i].c_str();
			ImGui::SliderInt(objName, &SceneData.objsInfo[i * 6 + 2], -1, 1);
		}
		ImGui::EndTabItem();
	}
	
	ImGui::EndTabBar();
	
	if (ImGui::Button("Compute"))
	{
		initCuda();
	}
	std::string fileName;
	GenFileName(&fileName);
	const char* fn = fileName.c_str();
	ImGui::Text("Output to:");
	ImGui::Text(fn);
	if (ImGui::Button("Output data"))
	{
		AutoOutput();
	}
	ImGui::SameLine();
	std::string frameCount = std::to_string(frame);
	const char* fc = frameCount.c_str();
	ImGui::Text(fc);
	ImGui::Render();
}

void GenFileName(std::string *s)
{
	s->clear();
	*s += "output/mesh_";
	if (type == 0)
	{
		*s += "emi_and_refl";
	}
	else if (type == 1)
	{
		*s += "emi_only";
	}
	else
	{
		*s += "refl_only";
	}
	*s += "_cam_";
	if (camAtRight) *s += "right.txt";
	else *s += "left.txt";
	return;
}

// create pixel buffer object in OpenGL
void createPBO(GLuint *pbo)
{
	if (pbo)
	{
		int num_texels = width * height;
		int num_values = num_texels * 3;

		int size_tex_data = sizeof(GLfloat) * num_values;

		glGenBuffers(1, pbo);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);

		cudaGraphicsGLRegisterBuffer(&resource, *pbo, cudaGraphicsMapFlagsWriteDiscard);
	}
}
// create texture in OpenGL
void createTexture(GLuint *textureID, unsigned int size_x, unsigned int size_y)
{
	glEnable(GL_TEXTURE_2D);

	glGenTextures(1, textureID);
	glBindTexture(GL_TEXTURE_2D, *textureID);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, NULL);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void prepareTextures()
{
	texNum = textures.size();
	tex_data_size = 0;
	h_tex_wh = new int[texNum * 2];
	for (int i = 0; i < texNum; i++)
	{
		tex_data_size += textures[i].width * textures[i].height;
		h_tex_wh[i * 2] = textures[i].width;
		h_tex_wh[i * 2 + 1] = textures[i].height;
	}
	int pixelsCount = 0;
	h_tex_data = new float3[tex_data_size];
	int offset = 0;
	for (int i = 0; i < texNum; i++)
	{
		for (int w = 0; w < textures[i].width; w++)
		{
			for (int h = 0; h < textures[i].height; h++)
			{
				int index = w * textures[i].width + h;
				// color channel: BGRA in FreeImage bytes, alpha is not used in this project
				// Red
				h_tex_data[index + offset].x = textures[i].imgData[index * 4 + 2] / 255.0f;
				// Green
				h_tex_data[index + offset].y = textures[i].imgData[index * 4 + 1] / 255.0f;
				// Blue
				h_tex_data[index + offset].z = textures[i].imgData[index * 4] / 255.0f;

				pixelsCount++;
			}		
		}
		offset += textures[i].width * textures[i].height;
	}
	//std::cout << h_tex_data[1023].x << " " << h_tex_data[1023].y << " " << h_tex_data[1023].z << std::endl;
	//// Convert to FreeImage format & save to file
	//FIBITMAP* image = FreeImage_ConvertFromRawBits(textures[0].imgData, textures[0].width, textures[0].height, textures[0].pitch, 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, false);
	//FreeImage_Save(FIF_BMP, image, "input/test.bmp", 0);
	//// Free resources
	//FreeImage_Unload(image);
}

void initCuda()
{
	frame = 0;
	// all verts in scene
	cudaMalloc((void**)& scene_verts, SceneData.vertsNum *sizeof(float3));
	cudaMemcpy(scene_verts, SceneData.verts, SceneData.vertsNum * sizeof(float3), cudaMemcpyHostToDevice);
	// all objects and info	[objVertsNum, matNum, normalTexNum, ambientTexNum, temperature, emiSource]
	cudaMalloc((void**)& scene_objs, SceneData.objsNum * 6 * sizeof(int));
	cudaMemcpy(scene_objs, SceneData.objsInfo, SceneData.objsNum * 6 * sizeof(int), cudaMemcpyHostToDevice);
	// all uvs at each vert
	cudaMalloc((void**)& scene_uvs, SceneData.vertsNum * sizeof(float2));
	cudaMemcpy(scene_uvs, SceneData.uvs, SceneData.vertsNum * sizeof(float2), cudaMemcpyHostToDevice);
	// all normals at each vert
	cudaMalloc((void**)& scene_normals, (SceneData.vertsNum/3) * sizeof(float3));
	cudaMemcpy(scene_normals, SceneData.normals, (SceneData.vertsNum/3) * sizeof(float3), cudaMemcpyHostToDevice);
	// width and height for each texture
	cudaMalloc((void**)& d_tex_wh, texNum * 2 * sizeof(int));
	cudaMemcpy(d_tex_wh, h_tex_wh, texNum * 2 * sizeof(int), cudaMemcpyHostToDevice);
	// all pixles for all data
	cudaMalloc((void**)& d_tex_data, tex_data_size * sizeof(float3));
	cudaMemcpy(d_tex_data, h_tex_data, tex_data_size * sizeof(float3), cudaMemcpyHostToDevice);
	// emi value list
	cudaMalloc((void**)& d_emiList, SceneData.objsNum * sizeof(float));
	cudaMemcpy(d_emiList, SceneData.emiList, SceneData.objsNum * sizeof(float), cudaMemcpyHostToDevice);
	// Buffer for Monte Carlo
	cudaMalloc(&accu, width * height * sizeof(float3));	// register accu buffer, this buffer won't refresh
	cudaMalloc(&randState, width * height * sizeof(curandState));	// each pixel's random seed per frame
	createPBO(&pbo);
	createTexture(&textureID, width, height);
}

void runCuda()
{
	size_t num_bytes;

	cudaGraphicsMapResources(1, &resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&result, &num_bytes, resource);

	launch_kernel(result, accu, randState, width, height, frame, camAtRight, waveNum, 
		SceneData.vertsNum, scene_verts, SceneData.objsNum, scene_objs,
		scene_uvs, scene_normals,
		texNum,d_tex_wh, d_tex_data, type, d_emiList);

	cudaGraphicsUnmapResources(1, &resource, 0);
}

// glut display callback function.
// This function gets called every time the scene gets redisplayed
// which means it runs only when some events happen, such as mouse clicked
// so currently it won't be used
void display()
{
	
}

float clamp(float n)
{
	if (n < 0) return 0;
	if (n > 1) return 1;
	return n;
}
inline int toInt(float x) { return int(clamp(x) * 255 + .5); }

int fileNum = 0;
void AutoOutput()	// output a result when achieve 8000 frame
{
	// for txt data, file size should around 3MB for 640*480 result
	GLfloat* pixels = new GLfloat[3 * width * height];
	glGetTexImage(GL_TEXTURE_2D ,0, GL_RGB, GL_FLOAT, pixels);

	std::ofstream outfile;
	std::string fileName;
	GenFileName(&fileName);

	outfile.open(fileName);

	//// ppm file debug
	//outfile << "P3\n " << width << " " << height << "\n" << "255\n";

	int i = 0;
	for (int x = 0; x < width * height * 3; x += 3)
	{
		outfile << pixels[x] << " ";

		//outfile << toInt(pixels[x]) << " ";
		//outfile << toInt(pixels[x + 1]) << " ";
		//outfile << toInt(pixels[x + 2]) << " ";

		i++;
		if (i == width)
		{
			i = 0;
			outfile << std::endl;
		}
	}

	std::cout << "Saved file: " << fileName << std::endl;

	fileNum++;
	//if (fileNum >= 6) std::cout << "Output finished!" << std::endl;
	delete[] pixels;
	outfile.close();
}

// glut idle callback.
// idle function gets called between frames
// which means it runs every frame automatically 
void idle()
{
	frame++;	// accumulate frame number

	////Auto output for all results
	//if (frame > OUTPUT_FRAME_NUM && type < 3)	// enough sample for current scene
	//{
	//	AutoOutput();
	//	if (camAtRight == false) type++;
	//	camAtRight = !camAtRight;
	//	initCuda();
	//}
	//if (type >= 3) return;	// pause the program

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	// clear current display result on the screen
	runCuda();	// run CUDA program and calculate current frame result

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

	glBindTexture(GL_TEXTURE_2D, textureID);

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
		GL_RGB, GL_FLOAT, NULL);

	// draw a quadrangle as large as the window
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 1.0f); glVertex3f(0.0f, 0.0f, 0.0f);
	glTexCoord2f(0.0f, 0.0f); glVertex3f(0.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex3f(1.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex3f(1.0f, 0.0f, 0.0f);
	glEnd();
	// unbind
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	draw_gui();

	glutSwapBuffers();
}



// Display info about the OpenGL implementation provided by the graphics driver.
void printGlInfo()
{
	std::cout << "Vendor: " << glGetString(GL_VENDOR) << std::endl;
	std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
	std::cout << "Version: " << glGetString(GL_VERSION) << std::endl;
	std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
	int X, Y, Z, total;
	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &X);
	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &Y);
	glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &Z);
	glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &total);
	std::cout << "Max Compute Work Group Size: " << X << " " << Y << " " << Z << std::endl;
	std::cout << "Max Compute Work Group Invocations: " << total << std::endl;
}

void initOpenGl()
{
	frame = 0;
	//Initialize glew so that new OpenGL function names can be used
	glewInit();

	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glOrtho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glEnable(GL_DEPTH_TEST);
	glClearColor(0.2, 0.2, 0.2, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

// glut callbacks need to send keyboard and mouse events to imgui
void keyboard(unsigned char key, int x, int y)
{
	ImGui_ImplGlut_KeyCallback(key);
}
// some callback functions here
void keyboard_up(unsigned char key, int x, int y)
{
	ImGui_ImplGlut_KeyUpCallback(key);
}

void special_up(int key, int x, int y)
{
	ImGui_ImplGlut_SpecialUpCallback(key);
}

void passive(int x, int y)
{
	ImGui_ImplGlut_PassiveMouseMotionCallback(x, y);
}

void special(int key, int x, int y)
{
	ImGui_ImplGlut_SpecialCallback(key);
}

void motion(int x, int y)
{
	ImGui_ImplGlut_MouseMotionCallback(x, y);
}

void mouse(int button, int state, int x, int y)
{
	ImGui_ImplGlut_MouseButtonCallback(button, state);
}

int main(int argc, char **argv)
{
	//Configure initial window state using freeglut
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowPosition(5, 5);
	glutInitWindowSize(width, height);
	int win = glutCreateWindow("Thermal Path Tracer");

	//Register callback functions with glut. 
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(special);
	glutKeyboardUpFunc(keyboard_up);
	glutSpecialUpFunc(special_up);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutPassiveMotionFunc(motion);

	glutIdleFunc(idle);

	
	initOpenGl();
	
	// load scene before init CUDA! Need mesh data for initialize
	LoadObj("input/scene2.obj", SceneData);
	// load texture
	tex_mug_normal.LoadTex("input/texture/mug_normal.jpg");
	textures.push_back(tex_mug_normal);
	tex_table_ambient.LoadTex("input/texture/table_ambient.jpg");
	textures.push_back(tex_table_ambient);
	//tex_table_ambient.LoadTex("input/texture/test_tex.jpg");
	//textures.push_back(tex_table_ambient);
	// set all data to 
	prepareTextures();
	//std::cout << SceneData.verts.size() << std::endl;
	
	initCuda();
	printGlInfo();
	ImGui_ImplGlut_Init();	// initialize the imgui system
	//Enter the glut event loop.
	glutMainLoop();
	cudaThreadExit();
	glutDestroyWindow(win);

	// free buffer before close
	cudaFree(result);
	cudaFree(accu);
	cudaFree(randState);
	cudaFree(scene_verts);
	cudaFree(scene_objs);
	cudaFree(scene_uvs);
	cudaFree(scene_normals);
	cudaFree(d_tex_wh);
	cudaFree(d_tex_data);


	delete[] h_tex_wh;
	delete[] h_tex_data;
	SceneData.FreeScene();
	tex_mug_normal.FreeTexture();
	tex_table_ambient.FreeTexture();

	ImGui_ImplGlut_Shutdown();

	return 0;
}


