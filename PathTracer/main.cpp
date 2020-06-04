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

// may used later when import mesh object
//#include "LoadMesh.h"
//#include "LoadTexture.h"
//#include "imgui_impl_glut.h"

#include "cuda_gl_interop.h"

#include "cutil_math.h"
#include <curand.h>
#include <curand_kernel.h>

#include "include/FreeImage.h"


const int width = 1280;	// width of the figure
const int height = 720;	// height of the figure
unsigned int frame = 0;	// a frame counter, used as a random seed

// For OpenGL
GLuint pbo = 1;	// pxiel buffer object, place for OpenGL and CUDA to switch data and display the result
GLuint textureID = 1;	// OpenGL texture to display the result

// For CUDA
struct cudaGraphicsResource* resource;	// pointer to the teturned object handle
uchar4* dptr;	// place for CUDA output
float3* accu;	// place for accumulate all frame result
curandState* randState;

// Implement of this function is in kernel.cu
extern "C" void launch_kernel(uchar4*, float3*, curandState*, unsigned int, unsigned int, unsigned int);

// create pixel buffer object in OpenGL
void createPBO(GLuint *pbo)
{
	if (pbo)
	{
		int num_texels = width * height;
		int num_values = num_texels * 4;

		int size_tex_data = sizeof(GLubyte) * num_values;

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

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

void runCuda()
{
	size_t num_bytes;

	cudaGraphicsMapResources(1, &resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, resource);

	launch_kernel(dptr, accu, randState, width, height, frame);

	cudaGraphicsUnmapResources(1, &resource, 0);
}

// glut display callback function.
// This function gets called every time the scene gets redisplayed
// which means it runs only when some events happen, such as mouse clicked
// so currently it won't be used
void display()
{
	
}



// glut idle callback.
// idle function gets called between frames
// which means it runs every frame automatically 
void idle()
{
	frame++;	// accumulate frame number
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	// clear current display result on the screen
	runCuda();	// run CUDA program and calculate current frame result

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

	glBindTexture(GL_TEXTURE_2D, textureID);

	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
		GL_RGBA, GL_UNSIGNED_BYTE, NULL);

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

void initCuda()
{
	// register accu buffer, this buffer won't refresh
	cudaMalloc(&accu, width * height * sizeof(float3));
	cudaMalloc(&randState, width * height * sizeof(curandState));
	createPBO(&pbo);
	createTexture(&textureID, width, height);
	runCuda();
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

	if (key == 's')
	{
		// Make the BYTE array, factor of 3 because it's RBG.
		//BYTE* pixels = new BYTE[3 * width * height];

		GLfloat* pixels = new GLfloat[3 * width * height];

		//glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);
		glReadPixels(0, 0, width, height, GL_RGB, GL_FLOAT, pixels);

		std::ofstream outfile;
		outfile.open("output/test.txt");
		
		for (int x = 0; x < width*3; x += 3)
		{
			for (int y = 0; y < height; y++)
			{
				outfile << pixels[x * height + y] << " ";
			}
			outfile << std::endl;
		}
		std::cout << "Saved file." << std::endl;

		//// Convert to FreeImage format & save to file
		//FIBITMAP* image = FreeImage_ConvertFromRawBits(pixels, width, height, 3 * width, 24, 0x0000FF, 0xFF0000, 0x00FF00, false);
		//FreeImage_Save(FIF_BMP, image, "output/test.bmp", 0);

		// Free resources
		//FreeImage_Unload(image);
		delete[] pixels;
		outfile.close();
	}

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
	initCuda();
	//ImGui_ImplGlut_Init(); // initialize the imgui system

	printGlInfo();

	//Enter the glut event loop.
	glutMainLoop();
	cudaThreadExit();
	glutDestroyWindow(win);

	cudaFree(dptr);
	cudaFree(accu);
	cudaFree(randState);

	return 0;
}


