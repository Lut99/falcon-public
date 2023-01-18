
#pragma once
#include "NeuralNetConfig.h"
#include "Layer.h"
#include "globals.h"
using namespace std;

class NeuralNetwork
{
public:
	RSSVectorMyType inputData;
	RSSVectorMyType outputData;
	vector<Layer*> layers;

	NeuralNetwork(NeuralNetConfig* config);
	~NeuralNetwork();
	void forward();
	void backward();
	void computeDelta();
	void updateEquations();
	void predict(RSSVectorMyType &maxIndex);
	void getAccuracy(const RSSVectorMyType &maxIndex, vector<size_t> &counter);

	/* TIM: Function that computes the metrics (accuracy, recall, precision, F1-score) for this neural network.
	 * 
	 * Note that we assume that it has already been trained.
	 * 
	 * # Arguments
	 * - `n_samples`: The number of samples to collect the metrics for. Should be divisible by the current batch size (i.e., MINI_BATCH_SIZE).
	 * - `width`: The width of the input images, in pixels.
	 * - `height`: The height of the input images, in pixels.
	 * - `depth`: The depth of the input image pixels, i.e., how many bytes are needed (1 for grayscale, 3 for RGB).
	 * 
	 * # Returns
	 * Nothing directly, but does print the metrics to `stdout` in a table-like fashion.
	 */
	void collectMetrics(size_t n_samples, size_t width, size_t height, size_t depth);
};
