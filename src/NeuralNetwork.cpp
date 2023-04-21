
#pragma once
#include "tools.h"
#include "secondary.h"
#include "FCLayer.h"
#include "CNNLayer.h"
#include "MaxpoolLayer.h"
#include "ReLULayer.h"
#include "BNLayer.h"
#include "NeuralNetwork.h"
#include "Functionalities.h"
#include "Accuracy.h"
using namespace std;
#include <iostream>

extern size_t INPUT_SIZE;
extern size_t LAST_LAYER_SIZE;
extern bool WITH_NORMALIZATION;
extern bool LARGE_NETWORK;


/***** NEURALNETWORK CLASS *****/
NeuralNetwork::NeuralNetwork(NeuralNetConfig* config)
:inputData(INPUT_SIZE * MINI_BATCH_SIZE),
 outputData(LAST_LAYER_SIZE * MINI_BATCH_SIZE)
{
	for (size_t i = 0; i < NUM_LAYERS; ++i)
	{
		if (config->layerConf[i]->type.compare("FC") == 0) {
			FCConfig *cfg = static_cast<FCConfig *>(config->layerConf[i]);
			layers.push_back(new FCLayer(cfg, i));
		}
		else if (config->layerConf[i]->type.compare("CNN") == 0) {
			CNNConfig *cfg = static_cast<CNNConfig *>(config->layerConf[i]);
			layers.push_back(new CNNLayer(cfg, i));
		}
		else if (config->layerConf[i]->type.compare("Maxpool") == 0) {
			MaxpoolConfig *cfg = static_cast<MaxpoolConfig *>(config->layerConf[i]);
			layers.push_back(new MaxpoolLayer(cfg, i));
		}
		else if (config->layerConf[i]->type.compare("ReLU") == 0) {
			ReLUConfig *cfg = static_cast<ReLUConfig *>(config->layerConf[i]);
			layers.push_back(new ReLULayer(cfg, i));
		}
		else if (config->layerConf[i]->type.compare("BN") == 0) {
			BNConfig *cfg = static_cast<BNConfig *>(config->layerConf[i]);
			layers.push_back(new BNLayer(cfg, i));
		}
		else
			error("Only FC, CNN, ReLU, Maxpool, and BN layer types currently supported");
	}
}


NeuralNetwork::~NeuralNetwork()
{
	for (vector<Layer*>::iterator it = layers.begin() ; it != layers.end(); ++it) {
		delete (*it);
	}

	layers.clear();
}

void NeuralNetwork::forward()
{
	log_print("NN.forward");
	cout << "layers[0] is " << layers.size() << "and size input layer data is " << inputData.size() << endl;
	layers[0]->forward(inputData);
	if (LARGE_NETWORK)
		cout << "Forward \t" << layers[0]->layerNum << " completed..." << endl;

	// cout << "----------------------------------------------" << endl;
	// cout << "DEBUG: forward() at NeuralNetwork.cpp" << endl;
	// print_vector(inputData, "FLOAT", "inputData:", 784);
	// print_vector(*((CNNLayer*)layers[0])->getWeights(), "FLOAT", "w0:", 20);
	// print_vector((*layers[0]->getActivation()), "FLOAT", "a0:", 1000);

	for (size_t i = 1; i < NUM_LAYERS; ++i)
	{
		layers[i]->forward(*(layers[i-1]->getActivation()));
		if (LARGE_NETWORK)
			cout << "Forward \t" << layers[i]->layerNum << " completed..." << endl;

		// print_vector((*layers[i]->getActivation()), "FLOAT", "Activation Layer"+to_string(i), 
		// 			(*layers[i]->getActivation()).size());
		// print_vector((*layers[i]->getActivation()), "FLOAT", "Activation Layer "+to_string(i), 100);
	}
	// print_vector(inputData, "FLOAT", "Input:", 784);
	// cout << "size of output: " << (*layers[NUM_LAYERS-1]->getActivation()).size() << endl;
	// print_vector((*layers[NUM_LAYERS-1]->getActivation()), "FLOAT", "Output:", 10);
}

void NeuralNetwork::backward()
{
	log_print("NN.backward");
	computeDelta();	
	updateEquations();
}

void NeuralNetwork::computeDelta()
{
	log_print("NN.computeDelta");
	
	size_t rows = MINI_BATCH_SIZE;
	size_t columns = LAST_LAYER_SIZE;
	size_t size = rows*columns;
	size_t index;

	if (WITH_NORMALIZATION)
	{
		RSSVectorMyType rowSum(size, make_pair(0,0));
		RSSVectorMyType quotient(size, make_pair(0,0));

		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < columns; ++j)
				rowSum[i*columns] = rowSum[i*columns] + 
									(*(layers[NUM_LAYERS-1]->getActivation()))[i * columns + j];

		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < columns; ++j)
				rowSum[i*columns + j] = rowSum[i*columns];

		#ifdef MM_TRACE
		cout << "NeuralNetwork::computeDelta(): calling division *(layers[NUM_LAYERS-1]->getActivation()) / rowSum = quotient (" << size << ')' << endl;
		#endif
		funcDivision(*(layers[NUM_LAYERS-1]->getActivation()), rowSum, quotient, size);

		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < columns; ++j)
			{
				index = i * columns + j;
				(*(layers[NUM_LAYERS-1]->getDelta()))[index] = quotient[index] - outputData[index];
			}
	}
	else
	{
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < columns; ++j)
			{
				index = i * columns + j;
				(*(layers[NUM_LAYERS-1]->getDelta()))[index] = 
				(*(layers[NUM_LAYERS-1]->getActivation()))[index] - outputData[index];
			}
	}

	if (LARGE_NETWORK)		
		cout << "Delta last layer completed." << endl;

	for (size_t i = NUM_LAYERS-1; i > 0; --i)
	{
		layers[i]->computeDelta(*(layers[i-1]->getDelta()));
		if (LARGE_NETWORK)
			cout << "Delta \t\t" << layers[i]->layerNum << " completed..." << endl;
	}
}

void NeuralNetwork::updateEquations()
{
	log_print("NN.updateEquations");

	for (size_t i = NUM_LAYERS-1; i > 0; --i)
	{
		layers[i]->updateEquations(*(layers[i-1]->getActivation()));	
		if (LARGE_NETWORK)
			cout << "Update Eq. \t" << layers[i]->layerNum << " completed..." << endl;	
	}

	layers[0]->updateEquations(inputData);
	if (LARGE_NETWORK)
		cout << "First layer update Eq. completed." << endl;		
}

void NeuralNetwork::predict(RSSVectorMyType &maxIndex)
{
	log_print("NN.predict");

	size_t rows = MINI_BATCH_SIZE;
	size_t columns = LAST_LAYER_SIZE;
	RSSVectorMyType max(rows);
	RSSVectorSmallType maxPrime(rows*columns);

	funcMaxpool(*(layers[NUM_LAYERS-1]->getActivation()), max, maxPrime, rows, columns);
}

/* new implementation, may still have bug and security flaws */
void NeuralNetwork::getAccuracy(const RSSVectorMyType &maxIndex, vector<size_t> &counter)
{
	log_print("NN.getAccuracy");

	size_t rows = MINI_BATCH_SIZE;
	size_t columns = LAST_LAYER_SIZE;
	
	RSSVectorMyType max(rows);
	RSSVectorSmallType maxPrime(rows*columns);
	RSSVectorMyType temp_max(rows), temp_groundTruth(rows);
	RSSVectorSmallType temp_maxPrime(rows*columns);
	
	vector<myType> groundTruth(rows*columns);
	vector<smallType> prediction(rows*columns);
	
	// reconstruct ground truth from output data
	funcReconstruct(outputData, groundTruth, rows*columns, "groundTruth", false);
	print_vector(outputData, "FLOAT", "outputData:", rows*columns);
	
	// reconstruct prediction from neural network
	funcMaxpool((*(layers[NUM_LAYERS-1])->getActivation()), temp_max, temp_maxPrime, rows, columns);
	funcReconstructBit(temp_maxPrime, prediction, rows*columns, "prediction", false);
	
	for (int i = 0, index = 0; i < rows; ++i){
		counter[1]++;
		for (int j = 0; j < columns; j++){
			index = i * columns + j;
			if ((int) groundTruth[index] * (int) prediction[index] || 
				(!(int) groundTruth[index] && !(int) prediction[index])){
				if (j == columns - 1){
					counter[0]++;
				}
			} else {
				break;
			}
		}
	}

	cout << "Rolling accuracy: " << counter[0] << " out of " 
		 << counter[1] << " (" << (counter[0]*100/counter[1]) << " %)" << endl;
}

/* TIM: Function that computes the metrics (accuracy, recall, precision, F1-score) for this neural network.
 * 
 * Note that we assume that it has already been trained.
 * 
 * # Arguments
 * - `width`: The width of the input images, in pixels.
 * - `height`: The height of the input images, in pixels.
 * - `depth`: The depth of the input image pixels, i.e., how many bytes are needed (1 for grayscale, 3 for RGB).
 * 
 * # Returns
 * Nothing directly, but does print the metrics to `stdout` in a table-like fashion.
 */
void NeuralNetwork::collectMetrics(size_t n_samples, size_t width, size_t height, size_t depth) {
	log_print("NN.getMetrics");

	// Get the old data somewhere so we can collect the metrics using custom data
	RSSVectorMyType old_input  = this->inputData;
	RSSVectorMyType old_output = this->outputData;

	// Do the number of batches
	size_t rows    = MINI_BATCH_SIZE;
	size_t columns = LAST_LAYER_SIZE;
	vector<float> groundTruth_float(n_samples*columns);
	vector<float> prediction_float(n_samples*columns);
	assert(n_samples % MINI_BATCH_SIZE == 0 && "The given number of samples must be a multiple of the MINI_BATCH_SIZE");
	for (size_t i = 0; i < n_samples / MINI_BATCH_SIZE; i++) {
		// Get the next test batch
		readMiniBatch(this, "TESTING");
		// printNetwork(this);

		// Do a forward pass to compute the activations
		this->forward();

		// Reconstruct the prediction of the neural network, using the functions as presented in `NeuralNetwork::getAccuracy()`.
		RSSVectorMyType max(rows);
		RSSVectorSmallType maxPrime(rows*columns);
		RSSVectorMyType temp_max(rows), temp_groundTruth(rows);
		RSSVectorSmallType temp_maxPrime(rows*columns);

		vector<myType> groundTruth(rows*columns);
		vector<smallType> prediction(rows*columns);
		
		// reconstruct ground truth from output data
		
		funcReconstruct(outputData, groundTruth, rows*columns, "groundTruth", false);
		cout << "GROUND TRUTTTTH ";
		for (int i = 0; i < groundTruth.size(); i++)
		{
			cout << groundTruth[i] << " "; 
			
		}
		cout << endl;
			

		// reconstruct prediction from neural network
		funcMaxpool((*(layers[NUM_LAYERS-1])->getActivation()), temp_max, temp_maxPrime, rows, columns);
		funcReconstructBit(temp_maxPrime, prediction, rows*columns, "prediction", false);
		cout << "PREDICTIONS ";

		for (int i = 0; i < prediction.size(); i++)
		{
			cout << prediction[i] << " "; 
			
		}
		cout << endl;
		// Cast both to floats
		for (size_t i = 0; i < (rows*columns); i++) {
			groundTruth_float[i] = ((float) groundTruth[i]) / (1 << FLOAT_PRECISION);
			prediction_float[i]  = (float) prediction[i];
		}
	}
	cout << "predicted " << endl;
	for (float i : prediction_float)
		cout << i << endl; 
	// Now print the metrics over the entire thing
	printMetrics(groundTruth_float, prediction_float);

	// Finally, restore the old data
	this->inputData  = old_input;
	this->outputData = old_output;

	// Done
}

// original implmentation of NeuralNetwork::getAccuracy(.)
/* void NeuralNetwork::getAccuracy(const RSSVectorMyType &maxIndex, vector<size_t> &counter)
{
	log_print("NN.getAccuracy");

	size_t rows = MINI_BATCH_SIZE;
	size_t columns = LAST_LAYER_SIZE;
	RSSVectorMyType max(rows);
	RSSVectorSmallType maxPrime(rows*columns);

	//Needed maxIndex here
	funcMaxpool(outputData, max, maxPrime, rows, columns);

	//Reconstruct things
	RSSVectorMyType temp_max(rows), temp_groundTruth(rows);
	// if (partyNum == PARTY_B)
	// 	sendTwoVectors<RSSMyType>(max, groundTruth, PARTY_A, rows, rows);

	// if (partyNum == PARTY_A)
	// {
	// 	receiveTwoVectors<RSSMyType>(temp_max, temp_groundTruth, PARTY_B, rows, rows);
	// 	addVectors<RSSMyType>(temp_max, max, temp_max, rows);
//		dividePlain(temp_max, (1 << FLOAT_PRECISION));
	// 	addVectors<RSSMyType>(temp_groundTruth, groundTruth, temp_groundTruth, rows);	
	// }

	for (size_t i = 0; i < MINI_BATCH_SIZE; ++i)
	{
		counter[1]++;
		if (temp_max[i] == temp_groundTruth[i])
			counter[0]++;
	}		

	cout << "Rolling accuracy: " << counter[0] << " out of " 
		 << counter[1] << " (" << (counter[0]*100/counter[1]) << " %)" << endl;
} */
