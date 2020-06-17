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
//#include "imgui_impl_glut.h"

#include "cuda_gl_interop.h"

#include "cutil_math.h"
#include <curand.h>
#include <curand_kernel.h>

#include "include/FreeImage.h"
#include "ObjLoader.h"

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
extern "C" void launch_kernel(float3*, float3*, curandState*, unsigned int, unsigned int, unsigned int, bool, int, int, float3*, int, int*);

// Auto output
#define OUTPUT_FRAME_NUM 500
bool camAtRight = true;	// pos of the camera, true for right side, false for left side
int waveNum = 0;

Scene SceneData;
float3* scene_verts; // the cuda device pointer that points to the uploaded triangles
int* scene_objs;

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

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_FLOAT, NULL);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

void initCuda()
{
	cudaMalloc((void**)& scene_verts, SceneData.vertsNum *sizeof(float3));
	cudaMemcpy(scene_verts, SceneData.verts, SceneData.vertsNum * sizeof(float3), cudaMemcpyHostToDevice);
	cudaMalloc((void**)& scene_objs, SceneData.objsNum * sizeof(int));
	cudaMemcpy(scene_objs, SceneData.objs, SceneData.objsNum * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc(&accu, width * height * sizeof(float3));	// register accu buffer, this buffer won't refresh
	cudaMalloc(&randState, width * height * sizeof(curandState));
	createPBO(&pbo);
	createTexture(&textureID, width, height);
}

void runCuda()
{
	size_t num_bytes;

	cudaGraphicsMapResources(1, &resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&result, &num_bytes, resource);

	launch_kernel(result, accu, randState, width, height, frame, camAtRight, waveNum, 
		SceneData.vertsNum, scene_verts, SceneData.objsNum, scene_objs);

	cudaGraphicsUnmapResources(1, &resource, 0);
}

// glut display callback function.
// This function gets called every time the scene gets redisplayed
// which means it runs only when some events happen, such as mouse clicked
// so currently it won't be used
void display()
{
	
}
const char* files[] =
{
	"output/0_r.bmp",
	"output/0_l.bmp",
	"output/1_r.bmp",
	"output/1_l.bmp",
	"output/2_r.bmp",
	"output/2_l.bmp",
	"output/3_r.bmp",
	"output/3_l.bmp",
	"output/4_r.bmp",
	"output/4_l.bmp",
	"output/5_r.bmp",
	"output/5_l.bmp",
	"output/6_r.bmp",
	"output/6_l.bmp",
	"output/7_r.bmp",
	"output/7_l.bmp",
	"output/8_r.bmp",
	"output/8_l.bmp",
	"output/9_r.bmp",
	"output/9_l.bmp",
	"output/10_r.bmp",
	"output/10_l.bmp",
};


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
	// for txt data, file size should around 8MB for 1280*720 result
	GLfloat* pixels = new GLfloat[3 * width * height];
	glGetTexImage(GL_TEXTURE_2D ,0, GL_RGB, GL_FLOAT, pixels);

	//// for bmp image
	//BYTE* pixels = new BYTE[3 * width * height];
	//glReadPixels(0, 0, width, height, GL_RGB, GL_BYTE, pixels);

	std::ofstream outfile;
	std::string  fileName;	// output/wave_?_cam_?.txt
	fileName += "output/wave_";
	fileName += std::to_string(waveNum);
	fileName += "_cam_";
	if (camAtRight) fileName += "right.txt";
	else fileName += "left.txt";

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

	//// Convert to FreeImage format & save to file
	//FIBITMAP* image = FreeImage_ConvertFromRawBits(pixels, width, height, 3 * width, 24, 0x0000FF, 0xFF0000, 0x00FF00, false);
	//FreeImage_Save(FIF_BMP, image, files[fileNum], 0);
	//// Free resources
	//FreeImage_Unload(image);

	fileNum++;
	if (fileNum >= 22) std::cout << "Output finished!" << std::endl;
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
	//if (frame > OUTPUT_FRAME_NUM && waveNum < 11)	// enough sample for current scene
	//{
	//	AutoOutput();
	//	if (camAtRight == false) waveNum++;
	//	camAtRight = !camAtRight;
	//	frame = 0;
	//	initCuda();
	//}
	//if (waveNum >= 11) return;	// pause the program

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
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

// glut callbacks need to send keyboard and mouse events to imgui
void keyboard(unsigned char key, int x, int y)
{
	//ImGui_ImplGlut_KeyCallback(key);
	std::cout << "key : " << key << ", x: " << x << ", y: " << y << std::endl;
}
// some callback functions here
void keyboard_up(unsigned char key, int x, int y)
{
	//ImGui_ImplGlut_KeyUpCallback(key);
}

void special_up(int key, int x, int y)
{
	//ImGui_ImplGlut_SpecialUpCallback(key);
}

void passive(int x, int y)
{
	//ImGui_ImplGlut_PassiveMouseMotionCallback(x, y);
}

void special(int key, int x, int y)
{
	//ImGui_ImplGlut_SpecialCallback(key);
}

void motion(int x, int y)
{
	//ImGui_ImplGlut_MouseMotionCallback(x, y);
}

void mouse(int button, int state, int x, int y)
{
	//ImGui_ImplGlut_MouseButtonCallback(button, state);
}

int main(int argc, char **argv)
{
	//Configure initial window state using freeglut
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowPosition(5, 5);
	glutInitWindowSize(width, height);
	int win = glutCreateWindow("Path Tracer");

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
	LoadObj("input/test.obj", SceneData);
	//std::cout << SceneData.verts.size() << std::endl;

	initCuda();
	//ImGui_ImplGlut_Init(); // initialize the imgui system
	printGlInfo();

	//Enter the glut event loop.
	glutMainLoop();
	cudaThreadExit();
	glutDestroyWindow(win);

	cudaFree(result);
	cudaFree(accu);
	cudaFree(scene_verts);
	cudaFree(randState);

	delete[] SceneData.verts;

	return 0;
}


