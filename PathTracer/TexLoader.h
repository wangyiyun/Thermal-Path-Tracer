#pragma once
#include <iostream>
#include <vector>
#include <string>
#include "cutil_math.h"
#include "include/FreeImage.h"

struct Texture
{
	int width;
	int height;
	int pitch;
	BYTE* imgData;
	void LoadTex(const char* fname)
	{
		FIBITMAP* tempImg = FreeImage_Load(FreeImage_GetFileType(fname, 0), fname);
		FIBITMAP* img = FreeImage_ConvertTo32Bits(tempImg);

		FreeImage_Unload(tempImg);

		width = FreeImage_GetWidth(img);
		height = FreeImage_GetHeight(img);
		pitch = FreeImage_GetPitch(img);
		imgData = new BYTE[height * pitch];

		FreeImage_ConvertToRawBits(imgData, img, pitch, 32, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, FALSE);
		FreeImage_Unload(img);
		printf("Load texture %s\n", fname);
	}
	void FreeTexture()
	{
		delete[] imgData;
	}
};