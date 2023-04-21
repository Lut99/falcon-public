
#pragma once
#include "LayerConfig.h"
#include "globals.h"
#include <iostream>
using namespace std;

class CNNConfig : public LayerConfig
{
public:
	size_t imageHeight = 0;
	size_t imageWidth = 0;

	size_t inputFeatures = 0;	//#Input feature maps
	size_t filters = 0;			//#Output feature maps
	size_t filterSize = 0;

	size_t stride = 0;
	size_t padding = 0;
	size_t batchSize = 0;

	size_t poolSizeX = 1;
	size_t poolSizeY = 1;

	CNNConfig(size_t _imageHeight, size_t _imageWidth, size_t _inputFeatures, size_t _filters, 
	size_t _filterSize, size_t _stride, size_t _padding, size_t _batchSize)
	:imageHeight(_imageHeight),
	 imageWidth(_imageWidth),
	 inputFeatures(_inputFeatures),
	 filters(_filters),
	 filterSize(_filterSize),
	 stride(_stride),
	 padding(_padding),
	 batchSize(_batchSize),
	 LayerConfig("CNN")
	{};
};
