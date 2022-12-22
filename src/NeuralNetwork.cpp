
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
using namespace std;

extern size_t INPUT_SIZE;
extern size_t LAST_LAYER_SIZE;
extern bool WITH_NORMALIZATION;
extern bool LARGE_NETWORK;
extern RSSVectorMyType testData, testLabels;


/***** HELPER MACROS *****/
#define N_LENGTH(N) \
	((N) > 0 ? (size_t) floor(log10((double) (N)) + 1.0) : 1)





/***** HELPER FUNCTIONS *****/
/* Prints the header of the confusion matrix.
 * 
 * # Arguments
 * - `largest_cell_size`: The size (in number of digits) of the largest value in any of the cells.
 * 
 * # Returns
 * Nothing, but does write it to stdout with proper formatting.
 */
void print_confusion_matrix_header(size_t largest_cell_size) {
	// Print the empty cell
	size_t largest_class_size = N_LENGTH(LAST_LAYER_SIZE);
	cout << " | " << std::string(largest_class_size, ' ');

	// Next, print all of the classes
	for (int i = 0; i < LAST_LAYER_SIZE; i++) {
		cout << " | " << std::string(largest_cell_size - N_LENGTH(i), ' ') << i;
	}

	// End the line
	cout << " |" << endl;
}

/* Prints a single row of the confusion matrix.
 * 
 * We assume that the matrix is LAST_LAYER_SIZE * LAST_LAYER_SIZE, implying that the given array contains at least LAST_LAYER_SIZE elements (since it's a single row).
 * 
 * # Arguments
 * - `confusion_row`: A point to a row of at least LAST_LAYER_SIZE elements.
 * - `class_number`: The class that is represented at this row (vertically).
 * - `largest_cell_size`: The size (in number of digits) of the largest value in any of the cells.
 * 
 * # Returns
 * Nothing, but does write it to stdout with proper formatting.
 */
void print_confusion_matrix_row(int* confusion_row, int class_number, size_t largest_cell_size) {
	// Print the class first
	size_t largest_class_size = N_LENGTH(LAST_LAYER_SIZE);
	cout << " | " << std::string(largest_class_size - N_LENGTH(class_number), ' ') << class_number;

	// Next, print all of the cells
	for (int i = 0; i < LAST_LAYER_SIZE; i++) {
		int clss = confusion_row[i];
		cout << " | " << std::string(largest_cell_size - N_LENGTH(clss), ' ') << clss;
	}

	// End the line
	cout << " |" << endl;
}

/* Prints a single row of the confusion matrix, but the "line" only.
 * 
 * # Arguments
 * - `largest_cell_size`: The size (in number of digits) of the largest value in any of the cells.
 * - `left`: The character to print at the leftmost part of the line.
 * - `middle`: The character to print at each intersection of a vertical and horizontal line.
 * - `right`: The character to print the rightmost part of the line.
 * 
 * # Returns
 * Nothing, but does write it to stdout with proper formatting.
 */
void print_confusion_matrix_row_lines(size_t largest_cell_size, const char* left, const char* middle, const char* right) {
	// Print the class cell
	size_t largest_class_size = N_LENGTH(LAST_LAYER_SIZE);
	cout << ' ' << left << '-' << std::string(largest_class_size, '-');

	// Next, print all of the cells
	for (int i = 0; i < LAST_LAYER_SIZE; i++) {
		cout << '-' << middle << '-' << std::string(largest_cell_size, '-');
	}

	// End the line
	cout << '-' << right << endl;
}





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
	// print_vector(outputData, "FLOAT", "outputData:", rows*columns);
	
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
void NeuralNetwork::collectMetrics(size_t width, size_t height, size_t depth) {
	log_print("NN.getMetrics");

	// Truncate the test labels such that the 

	// Set the input data as the test data
	RSSVectorMyType old_data = this->inputData;
	// this->inputData = test_data;
	readMiniBatch(this, "TESTING", testData.size() / (width * height * depth));
	// printNetwork(this);

	// Do a forward pass to compute the activations
	this->forward();

	// Reconstruct the prediction of the neural network, using the functions as presented in `NeuralNetwork::getAccuracy()`.
	size_t rows = testLabels.size() / LAST_LAYER_SIZE;
	size_t columns = LAST_LAYER_SIZE;

	RSSVectorMyType max(rows);
	RSSVectorSmallType maxPrime(rows*columns);
	RSSVectorMyType temp_max(rows), temp_groundTruth(rows);
	RSSVectorSmallType temp_maxPrime(rows*columns);

	vector<myType> groundTruth(rows*columns);
	vector<smallType> prediction(rows*columns);

	// reconstruct ground truth from output data
	// funcReconstruct(outputData, groundTruth, rows*columns, "groundTruth", false);
	// print_vector(outputData, "FLOAT", "outputData:", rows*columns);

	// Reconstruct the ground truth from the testLabels
	for (size_t i = 0; i < testLabels.size(); i++) {
		groundTruth[i] = ((float) testLabels[i].first) / (1 << FLOAT_PRECISION);
	}
	
	// reconstruct prediction from neural network
	funcMaxpool((*(layers[NUM_LAYERS-1])->getActivation()), temp_max, temp_maxPrime, rows, columns);
	funcReconstructBit(temp_maxPrime, prediction, rows*columns, "prediction", false);

	// Sanity check that the prediction is what we're looking for
	assert(groundTruth.size() == prediction.size());

	// Now we can collect the metrics by comparing the results
	std::vector<int> confusion_matrix(LAST_LAYER_SIZE * LAST_LAYER_SIZE, 0);

	// Start computing the accuracy and stuff
	double accuracy = 0.0;
	double total    = 0.0;
	for (size_t i = 0; i < groundTruth.size(); i += LAST_LAYER_SIZE) {
		// Compress the string of nodes to a single number (the class)
		int act = -1;
		for (size_t j = 0; j < LAST_LAYER_SIZE; j++) {
			if (groundTruth[i + j] > 0.5) {
				if (act > -1) { cerr << "WARNING: Golden truth sample " << (i / LAST_LAYER_SIZE) << " has multiple output bits set (assuming last one)" << endl; }
				act = j;
			}
		}
		if (act < 0) { cerr << "FATAL: No class in golden truth sample " << (i / LAST_LAYER_SIZE) << endl; exit(1); }
		int pred = -1;
		for (size_t j = 0; j < LAST_LAYER_SIZE; j++) {
			if (prediction[i + j] > 0.5) {
				if (pred > -1) { cerr << "WARNING: Predicted sample " << (i / LAST_LAYER_SIZE) << " has multiple output bits set (assuming last one)" << endl; }
				pred = j;
			}
		}
		if (pred < 0) { cerr << "FATAL: No class in predicted sample " << (i / LAST_LAYER_SIZE) << endl; exit(1); }

		// Populate the confusion matrix accordingly
		confusion_matrix[pred * LAST_LAYER_SIZE + act] += 1;

		// Keep track for the global accuracy
		if (act == pred) { accuracy += 1; }
		total += 1;
	}
	accuracy /= total;

	// Print 'em
	size_t longest_len = 1;
	for (size_t i = 0; i < confusion_matrix.size(); i++) {
		size_t len = N_LENGTH(confusion_matrix[i]);
		if (len > longest_len) {
			longest_len = len;
		}
	}
	cout << "----------------------------------------------" << endl;
	cout << "Confusion matrix:" << endl;
	cout << "      Pred" << endl;
	// Print the top lines
	print_confusion_matrix_row_lines(longest_len, "┌", "┬", "┐");
	// Print the header
	print_confusion_matrix_header(longest_len);
	// Print the rows themselves
	for (size_t y = 0; y < LAST_LAYER_SIZE; y++) {
		print_confusion_matrix_row_lines(longest_len, "├", "┼", "┤");
		print_confusion_matrix_row(confusion_matrix.data() + (y * LAST_LAYER_SIZE), y, longest_len);
	}
	// Print the bottom lines
	print_confusion_matrix_row_lines(longest_len, "└", "┴", "┘");
	cout << "----------------------------------------------" << endl;

	// Print the global metrics
	cout << "Global" << endl;
	cout << " - Accuracy : " << (accuracy * 100.0) << '%' << endl;
	cout << "----------------------------------------------" << endl;

	// Then print the metrics per class
	for (size_t clss = 0; clss < LAST_LAYER_SIZE; clss++) {
		// Compute the TP, TN, FP and FN for this class
		int tp = 0, tn = 0, fp = 0, fn = 0;
		for (size_t y = 0; y < LAST_LAYER_SIZE; y++) {
			for (size_t x = 0; x < LAST_LAYER_SIZE; x++) {
				int n = confusion_matrix[y * LAST_LAYER_SIZE + x];

				// True positives is a single cell, namely all those predicted as this class that were this class (easy)
				if (x == clss && y == clss) { tp += n; }
				// True negatives are all the cells that do not have anything to do as this class (i.e., were not this class and not predicted as such)
				if (x != clss && y != clss) { tn += n; }
				// False positives are all the cells predicted as this class that were not in fact this class
				if (x != clss && y == clss) { fp += n; }
				// Finally, false negatives are all the cells predicted as not this class that in fact were
				if (x == clss && y != clss) { fn += n; }
			}
		}

		// Compute the metrics from this matrix
		double accuracy  = ((double) (tp + tn)) / ((double) (tp + tn + fp + fn));
		double precision = tp > 0 ? ((double) tp) / ((double) (tp + fp)) : 0;
		double recall    = tp > 0 ? ((double) tp) / ((double) (tp + fn)) : 0;
		double f1        = precision * recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

		// Print them
		cout << "Class " << clss << endl;
		cout << " - Accuracy  : " << (accuracy * 100.0) << '%' << endl;
		cout << " - Precision : " << precision << endl;
		cout << " - Recall    : " << recall << endl;
		cout << " - F1        : " << f1 << endl;
		cout << "----------------------------------------------" << endl;
	}

	// Uncomment to see what we're actually comparing!
	cout << "Predicted labels (raw):";
	for (size_t i = 0; i < prediction.size(); i++) {
		cout << ' ' << ((int) prediction[i]);
		if (i % 10 == 9) { cout << "   "; }
	}
	cout << endl;
	cout << "Ground truth (raw):";
	for (size_t i = 0; i < groundTruth.size(); i++) {
		cout << ' ' << ((int) groundTruth[i]);
		if (i % 10 == 9) { cout << "   "; }
	}
	cout << endl;
	cout << "Predicted labels (class):";
	for (size_t i = 0; i < prediction.size(); i += LAST_LAYER_SIZE) {
		// Compress the string of nodes to a single number (the class)
		int clss = -1;
		for (size_t j = 0; j < LAST_LAYER_SIZE; j++) {
			if (prediction[i + j] > 0.5) {
				clss = j;
			}
		}
		cout << ' ' << clss;
	}
	cout << endl;
	cout << "Ground truth (class):";
	for (size_t i = 0; i < groundTruth.size(); i += LAST_LAYER_SIZE) {
		// Compress the string of nodes to a single number (the class)
		int clss = -1;
		for (size_t j = 0; j < LAST_LAYER_SIZE; j++) {
			if (groundTruth[i + j] > 0.5) {
				clss = j;
			}
		}
		cout << ' ' << clss;
	}
	cout << endl;

	// Finally, restore the old data
	this->inputData = old_data;
	for (size_t i = 0; i < this->layers.size(); i++) {
		this->layers[i]->setInputRows(MINI_BATCH_SIZE);
	}

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
