#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
using namespace std;
#include <stdio.h>
#include "cutil_math.h"
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>
#include <unordered_map>

#define M_PI 3.14159265358979323846;

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA error: %s: %s. \n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

// hash function to calculate new seed for each frame
// see http://www.reedbeta.com/blog/2013/01/12/quick-and-easy-gpu-random-numbers-in-d3d11/
uint WangHash(uint a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

// "__host__": This function called by CPU and runs on CPU
// "__device__": This function called by GPU and runs on GPU (inside one thread)
// "__global__": This is a kernel function, called by CPU and runs on GPU
// "__constant__": This data won't and can't be modified

// Changing variables
__constant__ float3 camPos = { 30.0f, 0.0f, 300.0f };	// -left, +right
#define USING_WAVE 0	// from 0 to 10

// reflection type (DIFFuse, SPECular, REFRactive)
enum Refl_t { DIFF, SPEC, REFR };
// geometry type
enum Geom_t { SPHERE, CONE };

// mat name
#define mat_human 0
#define mat_marble 1
#define mat_paint 2
#define mat_glass 3
#define mat_rubber 4
#define mat_brass 5
#define mat_road 6
#define mat_al 7
#define mat_al2o3 8
#define mat_brick 9

__constant__ float wave[11] = {
	7.8576538e+02,
	8.1770000e+02,
	8.6250000e+02,
	9.1025000e+02,
	9.4255000e+02,
	9.7750000e+02,
	1.0277500e+03,
	1.0780000e+03,
	1.1255000e+03,
	1.1860000e+03,
	1.2766667e+03
};


// emiLib[matName][waveNum]
__constant__ float emiLib[10][11] = {
	9.9000000e-01,	9.5834758e-01,	8.7470001e-01,	5.0455443e-01,	9.2789246e-01,	1.2250251e-01,	9.6426578e-01,	5.5701898e-01,	4.1617280e-02,	9.7773773e-01,
	9.9000000e-01,	9.5462609e-01,	8.8365367e-01,	2.8523451e-01,	9.2827028e-01,	1.1789014e-01,	9.7194589e-01,	5.4616836e-01,	4.1602933e-02,	9.7348785e-01,
	9.9000000e-01,	9.5099592e-01,	9.6279529e-01,	3.8887318e-01,	9.2640468e-01,	1.2078545e-01,	9.6430868e-01,	5.2990503e-01,	4.0821044e-02,	9.6252597e-01,
	9.9000000e-01,	9.5741246e-01,	8.6909910e-01,	4.2252257e-01,	9.2027605e-01,	1.2892990e-01,	9.4494491e-01,	5.1621436e-01,	4.8036999e-02,	9.4693874e-01,
	9.9000000e-01,	9.6385735e-01,	8.5889954e-01,	4.4505789e-01,	9.2317386e-01,	1.3452107e-01,	9.5513005e-01,	5.0484414e-01,	1.4619579e-01,	9.3275042e-01,
	9.9000000e-01,	9.6087765e-01,	9.3344199e-01,	4.7704424e-01,	8.9968776e-01,	1.4311263e-01,	9.5631467e-01,	4.9568769e-01,	2.6974721e-01,	9.1201603e-01,
	9.9000000e-01,	9.5962251e-01,	9.4205163e-01,	5.6399482e-01,	8.6774658e-01,	1.4932587e-01,	9.5258259e-01,	4.7984848e-01,	4.2480553e-01,	8.7901868e-01,
	9.9000000e-01,	9.5305901e-01,	9.4627694e-01,	3.2859562e-01,	8.8061124e-01,	1.4229701e-01,	9.1783893e-01,	4.6578646e-01,	4.7823023e-01,	8.5128884e-01,
	9.9000000e-01,	9.5385122e-01,	9.5199753e-01,	4.2369253e-02,	8.9911606e-01,	1.3455656e-01,	9.1771733e-01,	4.5454008e-01,	5.1389488e-01,	9.0261137e-01,
	9.9000000e-01,	9.5852822e-01,	9.5649050e-01,	2.7487807e-02,	9.1817783e-01,	1.2604779e-01,	9.1884949e-01,	4.3838823e-01,	5.4462383e-01,	9.3754130e-01,
	9.9000000e-01,	9.5240096e-01,	9.5069231e-01,	8.9005827e-02,	9.3104627e-01,	1.1098321e-01,	9.5362853e-01,	4.1783501e-01,	5.6727138e-01,	9.7270040e-01
};

__device__ float BBp (float T, float v)
{
	// 2e8*2*pi*h_bar*c^2
	double c1 = 1.1910429524674593e-08;
	// 100*2*pi*h_bar*c/k
	double c2 = 1.4387773536379256;
	// BBp = c1*pow(v,3)/(exp(c2*v/T)-1)
	return float(c1 * pow(v, 3) / (exp(c2 * v / T) - 1));
}

struct Ray {
	float3 origin;
	float3 direction;
	// create a ray
	__device__ Ray(float3 o_, float3 d_) : origin(o_), direction(d_) {}
};

struct Hit
{
	float hitDist;		//hitDistance
	float3 normal;
	float3 oriNormal;	// oriented normal (for rafraction)
	float3 nextDir;		// direction for next segment
	int matName;
	float temperature;
	float emi;	// 1 - emiLib[matName("matName")][waveNum("wave_1")]
	Refl_t reflectType;
	Geom_t geomtryType;
	int geomID;
	__device__ void Init() {
		hitDist = 1e20;
		normal = make_float3(0.0f);
		oriNormal = make_float3(0.0f);
		nextDir = make_float3(0.0f);
		matName = -1;
		temperature = 0.0f;
		emi = 0.0f;
		reflectType = DIFF;
		geomtryType = SPHERE;
		geomID = -1;
	}
};

struct Sphere {

	float radius;
	float3 position;
	int matName;
	float temperature;
	Refl_t reflectType;	//DIFF, SPEC, REFR
	__device__ float intersect(const Ray& ray) const { // returns distance, 0 if nohit 

		// Ray/sphere intersection
		// Quadratic formula required to solve ax^2 + bx + c = 0 
		// Solution x = (-b +- sqrt(b*b - 4ac)) / 2a
		// Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 

		float3 op = position - ray.origin;
		float t, epsilon = 0.01f;
		float b = dot(op, ray.direction);
		float disc = b * b - dot(op, op) + radius * radius; // discriminant
		if (disc < 0) return 0; else disc = sqrtf(disc);
		return (t = b - disc) > epsilon ? t : ((t = b + disc) > epsilon ? t : 0);
	}
};

struct Cone {
	float3 tip, axis;
	float cosA, height;
	int matName;
	float temperature;
	Refl_t reflectType;	//DIFF, SPEC, REFR
	__device__ float intersect(const Ray& ray) const { // returns distance, 0 if nohit  

		float3 co = ray.origin - tip; float cos2t = cosA; cos2t *= cos2t;
		float t, dotDV = dot(ray.direction, axis), dotCOV = dot(co, axis);
		float a = dotDV * dotDV - cos2t, b = 2.0f * (dotDV * dotCOV - dot(ray.direction, co) * cos2t),
			c = dotCOV * dotCOV - dot(co, co) * cos2t, delta = b * b - 4 * a * c;
		if (delta <= 0.0f) return 0; else delta = sqrt(delta);
		t = (-b + delta) / 2.0f / a > 0.01f ? (-b + delta) / 2.0f / a : max((-b - delta) / 2.0f / a, 0.0f);
		float3 hit = ray.origin + t * ray.direction;
		if (dot(hit - tip, axis) <= 0.0f) return 0;
		return t;
	}
};

__constant__ Sphere spheres[] = {
	/* cornell box
	{radius	position						matName		temperature			reflectType*/
	{1e5f,	{-1e5f - 100.0f, 0.0f, 0.0f},	mat_brick,	20.0f + 273.15f,	DIFF},// left wall
	{1e5f,	{1e5f + 100.0f, 0.0f, 0.0f},	mat_brick,	20.0f + 273.15f,	DIFF},// right wall
	{1e5f,	{0.0f, 0.0f, -1e5f - 100.0f},	mat_brick,	20.0f + 273.15f,	DIFF},// back wall
	{1e5f,	{0.0f, 0.0f, 1e5f + 500.0f},	mat_brick,	20.0f + 273.15f,	DIFF},// front wall
	{1e5f,	{0.0f, -1e5f - 100.0f, 0.0f},	mat_road,	20.0f + 273.15f,	DIFF},// floor
	{1e5f,	{0.0f, 1e5f + 100.0f, 0.0f},	mat_brick,	20.0f + 273.15f,	DIFF},// ceiling  
	{40.0f,	{50.0f ,-70.0f, 0.0f},			mat_al,		72.5f + 273.15f,	DIFF},// sphere 
	{50.0f,	{0.0f ,135.0f, 0.0f},			mat_glass,	100.0f + 273.15f,	DIFF} // lamp 
};

__constant__ Cone cones[] = {
	/*
	tip							axis					cosA	height	matName		temperature			reflectType*/
	{{-50.0f, -20.0f, -80.0f},	{0.0f, -1.0f, 0.0f},	0.976296f,	80.0f,	mat_rubber,	37.0f + 273.15f,	DIFF}
};

__device__ inline bool intersect_scene(const Ray& ray, Hit& bestHit)
{
	float d = 1e20;
	float INF = 1e20;

	// intersect all spheres in the scene
	float spheresNum = sizeof(spheres) / sizeof(Sphere);
	for (int i = 0; i < spheresNum; i++)  // for all spheres in scene
	{
		// keep track of distance from origin to closest intersection point
		if ((d = spheres[i].intersect(ray)) && d < bestHit.hitDist && d > 0)
		{ 
			bestHit.hitDist = d;
			bestHit.geomtryType = SPHERE;
			bestHit.geomID = i;
		}
	}

	// intersect all cones in the scene
	float conesNum = sizeof(cones) / sizeof(Cone);
	for (int i = 0; i < conesNum; i++)  // for all cones in scene
	{
		// keep track of distance from origin to closest intersection point
		if ((d = cones[i].intersect(ray)) && d < bestHit.hitDist && d > 0)
		{
			bestHit.hitDist = d;
			bestHit.geomtryType = CONE;
			bestHit.geomID = i;
		}
	}

	// t is distance to closest intersection of ray with all primitives in the scene
	if (bestHit.hitDist < INF)
	{
		float3 hitPostion = ray.origin + ray.direction * bestHit.hitDist;
		switch (bestHit.geomtryType)
		{
		case SPHERE:
			bestHit.normal = normalize(hitPostion - spheres[bestHit.geomID].position);
			bestHit.oriNormal = dot(bestHit.normal, ray.direction) < 0.0f ? bestHit.normal : bestHit.normal * -1.0f;
			bestHit.matName = spheres[bestHit.geomID].matName;
			bestHit.temperature = spheres[bestHit.geomID].temperature;
			bestHit.emi = emiLib[bestHit.matName][USING_WAVE];	// start from 0 
			bestHit.reflectType = spheres[bestHit.geomID].reflectType;
			break;
		case CONE:
			float3 cp = hitPostion - cones[bestHit.geomID].tip;
			bestHit.normal = normalize(cp * dot(cones[bestHit.geomID].axis, cp) / dot(cp, cp) - cones[bestHit.geomID].axis);
			bestHit.oriNormal = dot(bestHit.normal, ray.direction) < 0.0f ? bestHit.normal : bestHit.normal * -1.0f;
			bestHit.matName = cones[bestHit.geomID].matName;
			bestHit.temperature = cones[bestHit.geomID].temperature;
			bestHit.emi = emiLib[bestHit.matName][USING_WAVE];	// start from 0 
			bestHit.reflectType = cones[bestHit.geomID].reflectType;
			break;
		default:
			break;
		}
		return true;
	}
	else return false;
}

// radiance function
// compute path bounces in scene and accumulate returned color from each path sgment
__device__ float radiance(Ray& ray, curandState* randstate, int frameNum) { // returns ray color

	Hit bestHit;
	// color mask
	float mask = 1.0f;
	// accumulated color for current pixel
	float accuIntensity = 0.0f;

	//// hit debug
	//bestHit.Init();
	//if (!intersect_scene(ray, bestHit))
	//	return 0.0f; // if miss, return black
	//else
	//{
	//	return bestHit.temperature/500.f;
	//	//return bestHit.emission;
	//}
	//// hit debug end
	

	int bounces = 0;
	while(bounces < 5 || curand_uniform(randstate) < 0.5f)
	{  
		bounces++;
		bestHit.Init();
		// intersect ray with scene
		if (!intersect_scene(ray, bestHit))
			return 0.0f; // if miss, return black
		// else: we've got a hit with a scene primitive
		accuIntensity += (mask * BBp(bestHit.temperature, wave[USING_WAVE])*bestHit.emi);
		float3 hitPosition = ray.origin + ray.direction * bestHit.hitDist;

		// SHADING: diffuse, specular or refractive

		// ideal diffuse reflection
		if (bestHit.reflectType == DIFF)
		{
			// create 2 random numbers
			float r1 = 2 * 3.1415926 * curand_uniform(randstate);
			float r2 = curand_uniform(randstate);
			float r2s = sqrtf(r2);

			// compute orthonormal coordinate frame uvw with hitpoint as origin 
			float3 w = bestHit.oriNormal;
			float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
			float3 v = cross(w, u);

			// compute cosine weighted random ray direction on hemisphere 
			bestHit.nextDir = normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrtf(1 - r2));

			// offset origin next path segment to prevent self intersection
			hitPosition += bestHit.oriNormal * 0.03;

			// multiply mask with color of object
			mask *= 1.0f - bestHit.emi;
		}

		// ideal specular reflection
		if (bestHit.reflectType == SPEC)
		{

			// reflect
			bestHit.nextDir = ray.direction - 2.0f * bestHit.normal * dot(bestHit.normal, ray.direction);

			// offset origin next path segment to prevent self intersection
			hitPosition += bestHit.oriNormal * 0.01;

			// multiply color to the object
			mask *= 1.0f - bestHit.emi;
		}

		// ideal refraction (based on smallpt code by Kevin Beason)
		if (bestHit.reflectType == REFR)
		{

			bool into = dot(bestHit.normal, bestHit.oriNormal) > 0; // is ray entering or leaving refractive material?
			float nc = 1.0f;  // Index of Refraction air
			float nt = 1.5f;  // Index of Refraction glass/water
			float nnt = into ? nc / nt : nt / nc;  // IOR ratio of refractive materials
			float ddn = dot(ray.direction, bestHit.oriNormal);
			float cos2t = 1.0f - nnt * nnt * (1.f - ddn * ddn);

			if (cos2t < 0.0f) // total internal reflection 
			{
				bestHit.nextDir = reflect(ray.direction, bestHit.normal); //d = r.dir - 2.0f * n * dot(n, r.dir);
				hitPosition += bestHit.oriNormal * 0.01f;
			}
			else // cos2t > 0
			{
				// compute direction of transmission ray
				float3 tdir = normalize(ray.direction * nnt - bestHit.normal * ((into ? 1 : -1) * (ddn * nnt + sqrtf(cos2t))));

				float R0 = (nt - nc) * (nt - nc) / (nt + nc) * (nt + nc);
				float c = 1.f - (into ? -ddn : dot(tdir, bestHit.normal));
				float Re = R0 + (1.f - R0) * c * c * c * c * c;
				float Tr = 1 - Re; // Transmission
				float P = .25f + .5f * Re;
				float RP = Re / P;
				float TP = Tr / (1.f - P);

				// randomly choose reflection or transmission ray
				if (curand_uniform(randstate) < 0.25) // reflection ray
				{
					mask *= RP;
					bestHit.nextDir = reflect(ray.direction, bestHit.normal);
					hitPosition += bestHit.oriNormal * 0.01f;
				}
				else // transmission ray
				{
					mask *= TP;
					bestHit.nextDir = tdir; //r = Ray(x, tdir); 
					hitPosition += bestHit.oriNormal * 0.0005f; // epsilon must be small to avoid artefacts
				}
			}
		}

		// set up origin and direction of next path segment
		ray.origin = hitPosition;
		ray.direction = bestHit.nextDir;
	}

	// add radiance up to a certain ray depth
	// return accumulated color after all bounces are computed
	return accuIntensity;
}


__device__ unsigned char Color(float c)
{
	c = clamp(c, 0.0f, 1.0f);
	return int(c * 255.99) & 0xff;
}
__device__ float3 gammaCorrect(float3 c)
{
	float3 g;
	g.x = pow(c.x, 1 / 2.2f);
	g.y = pow(c.y, 1 / 2.2f);
	g.z = pow(c.z, 1 / 2.2f);
	return g;
}


__global__ void rand_init(int max_x, int max_y, curandState* rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y))
		return;
	int pixel_index = j * max_x + i;
	// Each thread gets same seed, a different sequence number, no offset
	curand_init(1997 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(uchar4 *pos, float3* accumbuffer, curandState* randSt, int width, int height, int frameNum, int HashedFrameNum)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= width) || (j >= height)) 
		return;
	
	// unique id for the pixel
	int index = j * width + i;
	// create random number generator, see RichieSams blogspot
	curandState randState; // state of the random number generator, to prevent repetition, need refresh per frame
	curand_init(HashedFrameNum + index, 0, 0, &randState);
	float3 pixelColor = make_float3(0);
	// offset inside each pixel
	float offsetX = curand_uniform(&randState);	// get random float between (0, 1)
	float offsetY = curand_uniform(&randState);
	//float offsetX = m_rand(frameNum, make_float2(i, j));	// get random float between (0, 1)
	//float offsetY = m_rand(frameNum, make_float2(i, j));
	//if(index == 0 && frameNum < 100) printf("%f, %f\n", offsetX, offsetY);
	// uv(-0.5, 0.5)
	float2 uv = make_float2((i + offsetX) / width, (j + offsetY) / height) - make_float2(0.5f, 0.5f);
	Ray cam(camPos, normalize(make_float3(0.0f, 0.0f, -1.0f)));
	float3 screen = make_float3(uv.x * width, -uv.y * height, -500);
	float3 dir = normalize(screen - cam.origin);

	float intensity = radiance(Ray(cam.origin, dir), &randState, frameNum);
	pixelColor = make_float3(intensity);
	if (frameNum == 0) accumbuffer[index] = make_float3(0.0);	//init
	accumbuffer[index] += pixelColor;

	float3 tempCol = accumbuffer[index]/(float)frameNum;
	//tempCol = gammaCorrect(tempCol);

	// (0.0f, 1.0f) -> (0, 255)
	unsigned char r = Color(tempCol.x);
	unsigned char g = Color(tempCol.y);
	unsigned char b = Color(tempCol.z);
	//debug
	//unsigned char r = Color(dir.x);
	//unsigned char g = Color(dir.y);
	//unsigned char b = Color(dir.z);

	pos[index].w = 0;
	pos[index].x = r;
	pos[index].y = g;
	pos[index].z = b;
}

extern "C" void launch_kernel(uchar4* pos, float3* accumbuffer, curandState* randState, unsigned int w, unsigned int h, unsigned int frame) {

	//set thread number
	int tx = 16;
	int ty = 16;

	dim3 blocks(w / tx + 1, h / ty + 1);
	dim3 threads(tx, ty);
	render <<<blocks, threads >>> (pos, accumbuffer, randState, w, h, frame, WangHash(frame));

	cudaThreadSynchronize();
	checkCUDAError("kernel failed!");
}

