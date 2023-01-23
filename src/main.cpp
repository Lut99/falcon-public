#include <iostream>
#include <string>
#include "globals.h"
#include "AESObject.h"
#include "Precompute.h"
#include "secondary.h"
#include "connect.h"
#include "NeuralNetConfig.h"
#include "NeuralNetwork.h"
#include "unitTests.h"

#include "CNNLayer.h"
#include "FCLayer.h"

int partyNum;
AESObject* aes_indep;
AESObject* aes_next;
AESObject* aes_prev;
Precompute PrecomputeObject;

void print_vec(const RSSVectorMyType& vec) {
	cout << '[';
	for (size_t i = 0; i < (vec.size() > 10 ? 10 : vec.size()); i++) {
		cout << " (" << ((int) vec[i].first) << ", " << ((int) vec[i].second) << ')';
	}
	if (vec.size() > 10) { cout << " ..."; }
	cout << " ]" << endl;
}


int main(int argc, char** argv)
{
/****************************** PREPROCESSING ******************************/ 
	parseInputs(argc, argv);
	string network, dataset, security;
	bool PRELOADING = false;

/****************************** SELECT NETWORK ******************************/ 
	//Network {SecureML, Sarda, MiniONN, LeNet, AlexNet, and VGG16}
	//Dataset {MNIST, CIFAR10, and ImageNet}
	//Security {Semi-honest or Malicious}
	if (argc == 9)
	{network = argv[6]; dataset = argv[7]; security = argv[8];}
	else
	{
		network = "LeNet";
		dataset = "MNIST";
		security = "Semi-honest";
	}

	// Create the network
	NeuralNetwork* net;
	{
		// Populate the config
		NeuralNetConfig config(NUM_ITERATIONS);
		selectNetwork(network, dataset, security, &config);
		config.checkNetwork();

		// Create it
		net = new NeuralNetwork(&config);
	}

/****************************** AES SETUP and SYNC ******************************/ 
	aes_indep = new AESObject(argv[3]);
	aes_next = new AESObject(argv[4]);
	aes_prev = new AESObject(argv[5]);

	initializeCommunication(argv[2], partyNum);
	synchronize(2000000);

/****************************** RUN NETWORK/UNIT TESTS ******************************/ 
	//Run these if you want a preloaded network to be tested
	//assert(NUM_ITERATION == 1 and "check if readMiniBatch is false in test(net)")
	//First argument {SecureML, Sarda, MiniONN, or LeNet}
	// network += " preloaded"; PRELOADING = true;
	// preload_network(PRELOADING, network, net);

	#ifdef PRELOAD_NETWORK
	preload_network(true, network, net);
	#endif
	// {
	// 	RSSVectorMyType* weights;
	// 	RSSVectorMyType* biases;

	// 	// Layer 0
	// 	weights = ((CNNLayer*) net->layers[0])->getWeights();
	// 	biases  = ((CNNLayer*) net->layers[0])->getBias();
	// 	cout << "0 (CNNLayer) weights: ";
	// 	print_vec(*weights);
	// 	cout << "0 (CNNLayer) biases: ";
	// 	print_vec(*biases);

	// 	// Layer 3
	// 	weights = ((CNNLayer*) net->layers[3])->getWeights();
	// 	biases  = ((CNNLayer*) net->layers[3])->getBias();
	// 	cout << "3 (CNNLayer) weights: ";
	// 	print_vec(*weights);
	// 	cout << "3 (CNNLayer) biases: ";
	// 	print_vec(*biases);

	// 	// Layer 6
	// 	weights = ((FCLayer*) net->layers[6])->getWeights();
	// 	biases  = ((FCLayer*) net->layers[6])->getBias();
	// 	cout << "6 (FCLayer) weights: ";
	// 	print_vec(*weights);
	// 	cout << "6 (FCLayer) biases: ";
	// 	print_vec(*biases);

	// 	// Layer 8
	// 	weights = ((FCLayer*) net->layers[8])->getWeights();
	// 	biases  = ((FCLayer*) net->layers[8])->getBias();
	// 	cout << "8 (FCLayer) weights: ";
	// 	print_vec(*weights);
	// 	cout << "8 (FCLayer) biases: ";
	// 	print_vec(*biases);
	// }

	start_m();
	//Run unit tests in two modes: 
	//	1. Debug {Mat-Mul, DotProd, PC, Wrap, ReLUPrime, ReLU, Division, BN, SSBits, SS, and Maxpool}
	//	2. Test {Mat-Mul1, Mat-Mul2, Mat-Mul3 (and similarly) Conv*, ReLU*, ReLUPrime*, and Maxpool*} where * = {1,2,3}
	// runTest("Debug", "BN", network);
	// runTest("Test", "ReLUPrime1", network);

	// Run forward/backward for single layers
	//  1. what {F, D, U}
	// 	2. l {0,1,....NUM_LAYERS-1}
	// size_t l = 0;
	// string what = "F";
	// runOnly(net, l, what, network);

	//Run training if no preloading happened
	#ifndef PRELOAD_NETWORK
	// network += " train";
	train(net);
	#endif

	//Run inference (possibly with preloading a network)
	// network += " test";
	#ifdef PRELOAD_NETWORK
	test(true, network, net);
	#endif

	end_m(network);
	cout << "----------------------------------------------" << endl;  	
	cout << "Run details: " << NUM_OF_PARTIES << "PC (P" << partyNum 
		 << "), " << NUM_ITERATIONS << " iterations, batch size " << MINI_BATCH_SIZE << endl 
		 << "Running " << security << " " << network << " on " << dataset << " dataset" << endl;
	cout << "----------------------------------------------" << endl << endl;  

	printNetwork(net);

	// Finally, show the accuracy and junk
	if (dataset.compare("MNIST") == 0) {
		// I don't know the actual image sizes, just that the total should be 784...
		net->collectMetrics(MINI_BATCH_SIZE, 784, 1, 1);
	} else if (network.compare("AlexNet") == 0 && dataset.compare("CIFAR10") == 0) {
		net->collectMetrics(MINI_BATCH_SIZE, 33, 33, 3);
	} else if (network.compare("AlexNet") == 0 && dataset.compare("ImageNet") == 0) {
		net->collectMetrics(MINI_BATCH_SIZE, 56, 56, 3);
	} else if (network.compare("VGG16") == 0 && dataset.compare("CIFAR10") == 0) {
		net->collectMetrics(MINI_BATCH_SIZE, 32, 32, 3);
	} else if (network.compare("VGG16") == 0 && dataset.compare("ImageNet") == 0) {
		net->collectMetrics(MINI_BATCH_SIZE, 64, 64, 3);
	} else {
		cerr << "Encountered unknown neural network / dataset combination '" << network << "' / '" << dataset << '\'' << endl;
		return 1;
	}

/****************************** CLEAN-UP ******************************/ 
	delete aes_indep;
	delete aes_next;
	delete aes_prev;
	delete net;
	deleteObjects();

	return 0;
}




