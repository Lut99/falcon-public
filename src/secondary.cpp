
#include "connect.h" 
#include "secondary.h"

extern CommunicationObject commObject;
extern int partyNum;
extern string * addrs;
extern BmrNet ** communicationSenders;
extern BmrNet ** communicationReceivers;
extern void log_print(string str);
#define NANOSECONDS_PER_SEC 1E9

//For time measurements
clock_t tStart;
struct timespec requestStart, requestEnd;
bool alreadyMeasuringTime = false;
int roundComplexitySend = 0;
int roundComplexityRecv = 0;
bool alreadyMeasuringRounds = false;

//For faster modular operations
extern smallType additionModPrime[PRIME_NUMBER][PRIME_NUMBER];
extern smallType subtractModPrime[PRIME_NUMBER][PRIME_NUMBER];
extern smallType multiplicationModPrime[PRIME_NUMBER][PRIME_NUMBER];

// RSSVectorMyType trainData, testData;
// RSSVectorMyType trainLabels, testLabels;
// size_t trainDataBatchCounter = 0;
// size_t trainLabelsBatchCounter = 0;
// size_t testDataBatchCounter = 0;
// size_t testLabelsBatchCounter = 0;

ifstream trainDataPrev, trainDataNext, testDataPrev, testDataNext;
ifstream trainLabelsPrev, trainLabelsNext, testLabelsPrev, testLabelsNext;

size_t INPUT_SIZE;
size_t LAST_LAYER_SIZE;
size_t NUM_LAYERS;
bool WITH_NORMALIZATION;
bool LARGE_NETWORK;
size_t TRAINING_DATA_SIZE;
size_t TEST_DATA_SIZE;
string SECURITY_TYPE;

extern void print_linear(myType var, string type);
extern void funcReconstruct(const RSSVectorMyType &a, vector<myType> &b, size_t size, string str, bool print);


/***** HELPER MACROS *****/
// Opens the file with an associated error check.
#define OPEN_FILE(NUM, NAME, PATH) \
	ifstream NAME(PATH); if (!NAME.is_open()) { cerr << NUM << ": Failed to load dataset '" << PATH << "': " << strerror(errno) << endl; }



/******************* Main train and test functions *******************/
void parseInputs(int argc, char* argv[])
{	
	if (argc < 6) 
		print_usage(argv[0]);

	partyNum = atoi(argv[1]);

	for (int i = 0; i < PRIME_NUMBER; ++i)
		for (int j = 0; j < PRIME_NUMBER; ++j)
		{
			additionModPrime[i][j] = ((i + j) % PRIME_NUMBER);
			subtractModPrime[i][j] = ((PRIME_NUMBER + i - j) % PRIME_NUMBER);
			multiplicationModPrime[i][j] = ((i * j) % PRIME_NUMBER); //How come you give the right answer multiplying in 8-bits??
		}
}

void train(NeuralNetwork* net)
{
	log_print("train");

	for (int i = 0; i < NUM_ITERATIONS; ++i)
	{
		// cout << "----------------------------------" << endl;  
		cout << "Iteration " << i << endl;
		readMiniBatch(net, "TRAINING");
		net->forward();
		net->backward();
		// cout << "----------------------------------" << endl;  
	}
}


extern void print_vector(RSSVectorMyType &var, string type, string pre_text, int print_nos);
extern string which_network(string network);
void test(bool PRELOADING, string network, NeuralNetwork* net)
{
	log_print("test");

	//counter[0]: Correct samples, counter[1]: total samples
	vector<size_t> counter(2,0);
	RSSVectorMyType maxIndex(MINI_BATCH_SIZE);

	for (int i = 0; i < NUM_ITERATIONS; ++i)
	{
		if (!PRELOADING)
			readMiniBatch(net, "TESTING");

		net->forward();
		// net->predict(maxIndex);
		// net->getAccuracy(maxIndex, counter);
	}
	print_vector((*(net->layers[NUM_LAYERS-1])->getActivation()), "FLOAT", "MPC Output over uint32_t:", 1280);

	// Write output to file
	if (PRELOADING)
	{
		ofstream data_file;
		data_file.open("files/preload/"+dataset+"/"+which_network(network)+"/"+which_network(network)+".txt");
		
		vector<myType> b(MINI_BATCH_SIZE * LAST_LAYER_SIZE);
		funcReconstruct((*(net->layers[NUM_LAYERS-1])->getActivation()), b, MINI_BATCH_SIZE * LAST_LAYER_SIZE, "anything", false);
		for (int i = 0; i < MINI_BATCH_SIZE; ++i)
		{
			for (int j = 0; j < LAST_LAYER_SIZE; ++j)
				data_file << b[i*(LAST_LAYER_SIZE) + j] << " ";
			data_file << endl;
		}
	}
}


// Generate a file with 0's of appropriate size
void generate_zeros(string name, size_t number, string network)
{
	string default_path = "files/preload/"+dataset+"/"+which_network(network)+"/";
	ofstream data_file;
	data_file.open(default_path+name);

	for (int i = 0; i < number; ++i)
		data_file << (int)0 << " ";
}


extern size_t nextParty(size_t party);
#include "FCLayer.h"
#include "CNNLayer.h"



/***** HELPER FUNCTIONS *****/
// Loads the input for a given layer.
void load_input(NeuralNetwork* net, string net_name, string path, const bool ZEROS) {
	string path_input_1 = path+"input_"+to_string(partyNum);
	string path_input_2 = path+"input_"+to_string(nextParty(partyNum));
	ifstream f_input_1(path_input_1), f_input_2(path_input_2);
	if (!f_input_1.is_open()) { cerr << "Failed to open preload input file '" << path_input_1 << "': " << strerror(errno) << endl; }
	if (!f_input_2.is_open()) { cerr << "Failed to open preload input file '" << path_input_2 << "': " << strerror(errno) << endl; }

	float temp_next = 0, temp_prev = 0;
	for (int i = 0; i < INPUT_SIZE * MINI_BATCH_SIZE; ++i)
	{
		f_input_1 >> temp_next; f_input_2 >> temp_prev;
		net->inputData.at(i) = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
	}
	f_input_1.close(); f_input_2.close();
	if (ZEROS)
	{
		generate_zeros("input_1", INPUT_SIZE * MINI_BATCH_SIZE, net_name);
		generate_zeros("input_2", INPUT_SIZE * MINI_BATCH_SIZE, net_name);
	}
}

// Loads the weights for a given layer.
template<class LAYER> void load_weights(NeuralNetwork* net, string net_name, string path, size_t file_n, size_t layer, size_t size, const bool ZEROS) {
	string path_weight1_1 = path+"weight"+to_string(file_n)+"_"+to_string(partyNum);
	string path_weight1_2 = path+"weight"+to_string(file_n)+"_"+to_string(nextParty(partyNum));
	ifstream f_weight1_1(path_weight1_1), f_weight1_2(path_weight1_2);
	if (!f_weight1_1.is_open()) { cerr << "Failed to open preload weights file '" << path_weight1_1 << "' for layer " << layer << ": " << strerror(errno) << endl; }
	if (!f_weight1_2.is_open()) { cerr << "Failed to open preload weights file '" << path_weight1_2 << "' for layer " << layer << ": " << strerror(errno) << endl; }

	float temp_next = 0, temp_prev = 0;
	for (int i = 0; i < size; ++i)
	{
		f_weight1_1 >> temp_next; f_weight1_2 >> temp_prev;
		(*((LAYER*)net->layers.at(layer))->getWeights()).at(i) = 
				std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
	}
	f_weight1_1.close(); f_weight1_2.close();
	if (ZEROS)
	{
		generate_zeros("weight"+to_string(file_n)+"_1", size, net_name);
		generate_zeros("weight"+to_string(file_n)+"_2", size, net_name);
	}
}

// Loads the biases for a given layer.
template<class LAYER> void load_biases(NeuralNetwork* net, string net_name, string path, size_t file_n, size_t layer, size_t size, const bool ZEROS) {
	string path_bias1_1 = path+"bias"+to_string(file_n)+"_"+to_string(partyNum);
	string path_bias1_2 = path+"bias"+to_string(file_n)+"_"+to_string(nextParty(partyNum));
	ifstream f_bias1_1(path_bias1_1), f_bias1_2(path_bias1_2);
	if (!f_bias1_1.is_open()) { cerr << "Failed to open preload bias file '" << path_bias1_1 << "' for layer " << layer << ": " << strerror(errno) << endl; }
	if (!f_bias1_2.is_open()) { cerr << "Failed to open preload bias file '" << path_bias1_2 << "' for layer " << layer << ": " << strerror(errno) << endl; }

	float temp_next = 0, temp_prev = 0;
	for (int i = 0; i < size; ++i)
	{
		f_bias1_1 >> temp_next; f_bias1_2 >> temp_prev;
		(*((LAYER*)net->layers.at(layer))->getBias()).at(i) = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
	}
	f_bias1_1.close(); f_bias1_2.close();
	if (ZEROS)
	{
		generate_zeros("bias"+to_string(file_n)+"_1", size, net_name);
		generate_zeros("bias"+to_string(file_n)+"_2", size, net_name);
	}
}

/* Loads both the weights and the biases for the given layer.
 * 
 * The sizes for each operation (i.e., the number of numbers in a weight matrix) are automatically deduced based on the size of the `weights` array in the respective layer.
 * 
 * # Arguments
 * - `net`: The NeuralNetwork to put the loaded weights and biases in.
 * - `net_name`: The name of the network. Only used when `ZEROS` is set to `true`.
 * - `path`: The path of the folder where we can find the pretrained weights/biases file(s).
 * - `file_n`: The index of the file. This is essentially the "meta layer", i.e., the layer index without stuff like ReLU or Maxpool layers
 * - `layer`: The index of the layer in the NeuralNetwork `net`. This is _with_ all the stuff like ReLU or Maxpool layers.
 * - `ZEROS`: A constant that enables the generation of empty ZEROS files. Recommended to set to true at least once per fresh set of weights, lest errors follow.
 * 
 * # Returns
 * Nothing directly, but changes the given `net` to populate the vector returned by `layer::getWeights()` and `layer::getBiases()`.
 */
template<class LAYER> void load_layer(NeuralNetwork* net, const string& net_name, const string& path, size_t file_n, size_t layer, const bool ZEROS) {
	// Fetch the weights & biases sizes
	LAYER* layer_ptr    = (LAYER*) net->layers[layer];
	size_t weights_size = layer_ptr->getWeights()->size();
	size_t biases_size  = layer_ptr->getBias()->size();

	// Load the weights
	load_weights<LAYER>(net, net_name, path, file_n, layer, weights_size, ZEROS);
	// Load the biases
	load_biases<LAYER>(net, net_name, path, file_n, layer, biases_size, ZEROS);
}


void preload_network(bool PRELOADING, string network, NeuralNetwork* net)
{
	log_print("preload_network");
	assert((PRELOADING) and (NUM_ITERATIONS == 1) and (MINI_BATCH_SIZE == 128) && "Preloading conditions fail");

	float temp_next = 0, temp_prev = 0;
	string default_path = "files/preload/"+dataset+"/"+which_network(network)+"/";
	//Set to true if you want the zeros files generated.
	const bool ZEROS = true;

	if (which_network(network).compare("SecureML") == 0)
	{
		string temp = "SecureML";
		/************************** Input **********************************/
		string path_input_1 = default_path+"input_"+to_string(partyNum);
		string path_input_2 = default_path+"input_"+to_string(nextParty(partyNum));
		ifstream f_input_1(path_input_1), f_input_2(path_input_2);

		for (int i = 0; i < INPUT_SIZE * MINI_BATCH_SIZE; ++i)
		{
			f_input_1 >> temp_next; f_input_2 >> temp_prev;
			net->inputData[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_input_1.close(); f_input_2.close();
		if (ZEROS)
		{
			generate_zeros("input_1", 784*128, temp);
			generate_zeros("input_2", 784*128, temp);
		}

		// print_vector(net->inputData, "FLOAT", "inputData:", 784);

		/************************** Weight1 **********************************/
		string path_weight1_1 = default_path+"weight1_"+to_string(partyNum);
		string path_weight1_2 = default_path+"weight1_"+to_string(nextParty(partyNum));
		ifstream f_weight1_1(path_weight1_1), f_weight1_2(path_weight1_2);

		for (int column = 0; column < 128; ++column)
		{
			for (int row = 0; row < 784; ++row)
			{
				f_weight1_1 >> temp_next; f_weight1_2 >> temp_prev;
				(*((FCLayer*)net->layers[0])->getWeights())[128*row + column] = 
						std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
			}
		}
		f_weight1_1.close(); f_weight1_2.close();
		if (ZEROS)
		{
			generate_zeros("weight1_1", 784*128, temp);
			generate_zeros("weight1_2", 784*128, temp);
		}

		/************************** Weight2 **********************************/
		string path_weight2_1 = default_path+"weight2_"+to_string(partyNum);
		string path_weight2_2 = default_path+"weight2_"+to_string(nextParty(partyNum));
		ifstream f_weight2_1(path_weight2_1), f_weight2_2(path_weight2_2);

		for (int column = 0; column < 128; ++column)
		{
			for (int row = 0; row < 128; ++row)
			{
				f_weight2_1 >> temp_next; f_weight2_2 >> temp_prev;
				(*((FCLayer*)net->layers[2])->getWeights())[128*row + column] = 
						std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
			}
		}
		f_weight2_1.close(); f_weight2_2.close();
		if (ZEROS)
		{
			generate_zeros("weight2_1", 128*128, temp);
			generate_zeros("weight2_2", 128*128, temp);
		}

		/************************** Weight3 **********************************/
		string path_weight3_1 = default_path+"weight3_"+to_string(partyNum);
		string path_weight3_2 = default_path+"weight3_"+to_string(nextParty(partyNum));
		ifstream f_weight3_1(path_weight3_1), f_weight3_2(path_weight3_2);

		for (int column = 0; column < 10; ++column)
		{
			for (int row = 0; row < 128; ++row)
			{
				f_weight3_1 >> temp_next; f_weight3_2 >> temp_prev;
				(*((FCLayer*)net->layers[4])->getWeights())[10*row + column] = 
						std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
			}
		}
		f_weight3_1.close(); f_weight3_2.close();
		if (ZEROS)
		{
			generate_zeros("weight3_1", 128*10, temp);
			generate_zeros("weight3_2", 128*10, temp);
		}


		/************************** Bias1 **********************************/
		string path_bias1_1 = default_path+"bias1_"+to_string(partyNum);
		string path_bias1_2 = default_path+"bias1_"+to_string(nextParty(partyNum));
		ifstream f_bias1_1(path_bias1_1), f_bias1_2(path_bias1_2);

		for (int i = 0; i < 128; ++i)
		{
			f_bias1_1 >> temp_next; f_bias1_2 >> temp_prev;
			(*((FCLayer*)net->layers[0])->getBias())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_bias1_1.close(); f_bias1_2.close();
		if (ZEROS)
		{
			generate_zeros("bias1_1", 128, temp);
			generate_zeros("bias1_2", 128, temp);
		}


		/************************** Bias2 **********************************/
		string path_bias2_1 = default_path+"bias2_"+to_string(partyNum);
		string path_bias2_2 = default_path+"bias2_"+to_string(nextParty(partyNum));
		ifstream f_bias2_1(path_bias2_1), f_bias2_2(path_bias2_2);

		for (int i = 0; i < 128; ++i)
		{
			f_bias2_1 >> temp_next; f_bias2_2 >> temp_prev;
			(*((FCLayer*)net->layers[2])->getBias())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_bias2_1.close(); f_bias2_2.close();
		if (ZEROS)
		{
			generate_zeros("bias2_1", 128, temp);
			generate_zeros("bias2_2", 128, temp);
		}


		/************************** Bias3 **********************************/
		string path_bias3_1 = default_path+"bias3_"+to_string(partyNum);
		string path_bias3_2 = default_path+"bias3_"+to_string(nextParty(partyNum));
		ifstream f_bias3_1(path_bias3_1), f_bias3_2(path_bias3_2);

		for (int i = 0; i < 10; ++i)
		{
			f_bias3_1 >> temp_next; f_bias3_2 >> temp_prev;
			(*((FCLayer*)net->layers[4])->getBias())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_bias3_1.close(); f_bias3_2.close();
		if (ZEROS)
		{
			generate_zeros("bias3_1", 10, temp);
			generate_zeros("bias3_2", 10, temp);
		}
	}
	else if (which_network(network).compare("Sarda") == 0)
	{
		string temp = "Sarda";
		/************************** Input **********************************/
		string path_input_1 = default_path+"input_"+to_string(partyNum);
		string path_input_2 = default_path+"input_"+to_string(nextParty(partyNum));
		ifstream f_input_1(path_input_1), f_input_2(path_input_2);

		for (int i = 0; i < INPUT_SIZE * MINI_BATCH_SIZE; ++i)
		{
			f_input_1 >> temp_next; f_input_2 >> temp_prev;
			net->inputData[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_input_1.close(); f_input_2.close();
		if (ZEROS)
		{
			generate_zeros("input_1", 784*128, temp);
			generate_zeros("input_2", 784*128, temp);
		}

		// print_vector(net->inputData, "FLOAT", "inputData:", 784);

		/************************** Weight1 **********************************/
		string path_weight1_1 = default_path+"weight1_"+to_string(partyNum);
		string path_weight1_2 = default_path+"weight1_"+to_string(nextParty(partyNum));
		ifstream f_weight1_1(path_weight1_1), f_weight1_2(path_weight1_2);

		for (int column = 0; column < 5; ++column)
		{
			for (int row = 0; row < 4; ++row)
			{
				f_weight1_1 >> temp_next; f_weight1_2 >> temp_prev;
				(*((CNNLayer*)net->layers[0])->getWeights())[4*column + row] = 
						std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
			}
		}
		f_weight1_1.close(); f_weight1_2.close();
		if (ZEROS)
		{
			generate_zeros("weight1_1", 2*2*1*5, temp);
			generate_zeros("weight1_2", 2*2*1*5, temp);
		}

		/************************** Weight2 **********************************/
		string path_weight2_1 = default_path+"weight2_"+to_string(partyNum);
		string path_weight2_2 = default_path+"weight2_"+to_string(nextParty(partyNum));
		ifstream f_weight2_1(path_weight2_1), f_weight2_2(path_weight2_2);

		for (int column = 0; column < 100; ++column)
		{
			for (int row = 0; row < 980; ++row)
			{
				f_weight2_1 >> temp_next; f_weight2_2 >> temp_prev;
				(*((FCLayer*)net->layers[2])->getWeights())[100*row + column] = 
						std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
			}
		}
		f_weight2_1.close(); f_weight2_2.close();
		if (ZEROS)
		{
			generate_zeros("weight2_1", 980*100, temp);
			generate_zeros("weight2_2", 980*100, temp);
		}


		/************************** Weight3 **********************************/
		string path_weight3_1 = default_path+"weight3_"+to_string(partyNum);
		string path_weight3_2 = default_path+"weight3_"+to_string(nextParty(partyNum));
		ifstream f_weight3_1(path_weight3_1), f_weight3_2(path_weight3_2);

		for (int column = 0; column < 10; ++column)
		{
			for (int row = 0; row < 100; ++row)
			{
				f_weight3_1 >> temp_next; f_weight3_2 >> temp_prev;
				(*((FCLayer*)net->layers[4])->getWeights())[10*row + column] = 
						std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
			}
		}
		f_weight3_1.close(); f_weight3_2.close();
		if (ZEROS)
		{
			generate_zeros("weight3_1", 100*10, temp);
			generate_zeros("weight3_2", 100*10, temp);
		}

		/************************** Bias1 **********************************/
		string path_bias1_1 = default_path+"bias1_"+to_string(partyNum);
		string path_bias1_2 = default_path+"bias1_"+to_string(nextParty(partyNum));
		ifstream f_bias1_1(path_bias1_1), f_bias1_2(path_bias1_2);

		for (int i = 0; i < 5; ++i)
		{
			f_bias1_1 >> temp_next; f_bias1_2 >> temp_prev;
			(*((CNNLayer*)net->layers[0])->getBias())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_bias1_1.close(); f_bias1_2.close();
		if (ZEROS)
		{
			generate_zeros("bias1_1", 5, temp);
			generate_zeros("bias1_2", 5, temp);
		}

		/************************** Bias2 **********************************/
		string path_bias2_1 = default_path+"bias2_"+to_string(partyNum);
		string path_bias2_2 = default_path+"bias2_"+to_string(nextParty(partyNum));
		ifstream f_bias2_1(path_bias2_1), f_bias2_2(path_bias2_2);

		for (int i = 0; i < 100; ++i)
		{
			f_bias2_1 >> temp_next; f_bias2_2 >> temp_prev;
			(*((FCLayer*)net->layers[2])->getBias())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_bias2_1.close(); f_bias2_2.close();
		if (ZEROS)
		{
			generate_zeros("bias2_1", 100, temp);
			generate_zeros("bias2_2", 100, temp);
		}

		/************************** Bias3 **********************************/
		string path_bias3_1 = default_path+"bias3_"+to_string(partyNum);
		string path_bias3_2 = default_path+"bias3_"+to_string(nextParty(partyNum));
		ifstream f_bias3_1(path_bias3_1), f_bias3_2(path_bias3_2);

		for (int i = 0; i < 10; ++i)
		{
			f_bias3_1 >> temp_next; f_bias3_2 >> temp_prev;
			(*((FCLayer*)net->layers[4])->getBias())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_bias3_1.close(); f_bias3_2.close();
		if (ZEROS)
		{
			generate_zeros("bias3_1", 10, temp);
			generate_zeros("bias3_2", 10, temp);
		}
	}
	else if (which_network(network).compare("MiniONN") == 0)
	{
		string temp = "MiniONN";
		/************************** Input **********************************/
		string path_input_1 = default_path+"input_"+to_string(partyNum);
		string path_input_2 = default_path+"input_"+to_string(nextParty(partyNum));
		ifstream f_input_1(path_input_1), f_input_2(path_input_2);

		for (int i = 0; i < INPUT_SIZE * MINI_BATCH_SIZE; ++i)
		{
			f_input_1 >> temp_next; f_input_2 >> temp_prev;
			net->inputData[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_input_1.close(); f_input_2.close();
		if (ZEROS)
		{
			generate_zeros("input_1", 784*128, temp);
			generate_zeros("input_2", 784*128, temp);
		}

		// print_vector(net->inputData, "FLOAT", "inputData:", 784);

		/************************** Weight1 **********************************/
		string path_weight1_1 = default_path+"weight1_"+to_string(partyNum);
		string path_weight1_2 = default_path+"weight1_"+to_string(nextParty(partyNum));
		ifstream f_weight1_1(path_weight1_1), f_weight1_2(path_weight1_2);

		for (int row = 0; row < 5*5*1*16; ++row)
		{
			f_weight1_1 >> temp_next; f_weight1_2 >> temp_prev;
			(*((CNNLayer*)net->layers[0])->getWeights())[row] = 
					std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_weight1_1.close(); f_weight1_2.close();
		if (ZEROS)
		{
			generate_zeros("weight1_1", 5*5*1*16, temp);
			generate_zeros("weight1_2", 5*5*1*16, temp);
		}

		/************************** Weight2 **********************************/
		string path_weight2_1 = default_path+"weight2_"+to_string(partyNum);
		string path_weight2_2 = default_path+"weight2_"+to_string(nextParty(partyNum));
		ifstream f_weight2_1(path_weight2_1), f_weight2_2(path_weight2_2);


		for (int row = 0; row < 25*16*16; ++row)
		{
			f_weight2_1 >> temp_next; f_weight2_2 >> temp_prev;
			(*((CNNLayer*)net->layers[3])->getWeights())[row] = 
					std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_weight2_1.close(); f_weight2_2.close();
		if (ZEROS)
		{
			generate_zeros("weight2_1", 5*5*16*16, temp);
			generate_zeros("weight2_2", 5*5*16*16, temp);
		}

		/************************** Weight3 **********************************/
		string path_weight3_1 = default_path+"weight3_"+to_string(partyNum);
		string path_weight3_2 = default_path+"weight3_"+to_string(nextParty(partyNum));
		ifstream f_weight3_1(path_weight3_1), f_weight3_2(path_weight3_2);

		for (int column = 0; column < 100; ++column)
		{
			for (int row = 0; row < 256; ++row)
			{
				f_weight3_1 >> temp_next; f_weight3_2 >> temp_prev;
				(*((FCLayer*)net->layers[6])->getWeights())[100*row + column] = 
						std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
			}
		}
		f_weight3_1.close(); f_weight3_2.close();
		if (ZEROS)
		{
			generate_zeros("weight3_1", 256*100, temp);
			generate_zeros("weight3_2", 256*100, temp);
		}


		/************************** Weight4 **********************************/
		string path_weight4_1 = default_path+"weight4_"+to_string(partyNum);
		string path_weight4_2 = default_path+"weight4_"+to_string(nextParty(partyNum));
		ifstream f_weight4_1(path_weight4_1), f_weight4_2(path_weight4_2);

		for (int column = 0; column < 10; ++column)
		{
			for (int row = 0; row < 100; ++row)
			{
				f_weight4_1 >> temp_next; f_weight4_2 >> temp_prev;
				(*((FCLayer*)net->layers[8])->getWeights())[10*row + column] = 
						std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
			}
		}
		f_weight4_1.close(); f_weight4_2.close();
		if (ZEROS)
		{
			generate_zeros("weight4_1", 100*10, temp);
			generate_zeros("weight4_2", 100*10, temp);
		}

		/************************** Bias1 **********************************/
		string path_bias1_1 = default_path+"bias1_"+to_string(partyNum);
		string path_bias1_2 = default_path+"bias1_"+to_string(nextParty(partyNum));
		ifstream f_bias1_1(path_bias1_1), f_bias1_2(path_bias1_2);

		for (int i = 0; i < 16; ++i)
		{
			f_bias1_1 >> temp_next; f_bias1_2 >> temp_prev;
			(*((CNNLayer*)net->layers[0])->getBias())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_bias1_1.close(); f_bias1_2.close();
		if (ZEROS)
		{
			generate_zeros("bias1_1", 16, temp);
			generate_zeros("bias1_2", 16, temp);
		}

		/************************** Bias2 **********************************/
		string path_bias2_1 = default_path+"bias2_"+to_string(partyNum);
		string path_bias2_2 = default_path+"bias2_"+to_string(nextParty(partyNum));
		ifstream f_bias2_1(path_bias2_1), f_bias2_2(path_bias2_2);

		for (int i = 0; i < 16; ++i)
		{
			f_bias2_1 >> temp_next; f_bias2_2 >> temp_prev;
			(*((CNNLayer*)net->layers[3])->getBias())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_bias2_1.close(); f_bias2_2.close();
		if (ZEROS)
		{
			generate_zeros("bias2_1", 16, temp);
			generate_zeros("bias2_2", 16, temp);
		}

		/************************** Bias3 **********************************/
		string path_bias3_1 = default_path+"bias3_"+to_string(partyNum);
		string path_bias3_2 = default_path+"bias3_"+to_string(nextParty(partyNum));
		ifstream f_bias3_1(path_bias3_1), f_bias3_2(path_bias3_2);

		for (int i = 0; i < 100; ++i)
		{
			f_bias3_1 >> temp_next; f_bias3_2 >> temp_prev;
			(*((FCLayer*)net->layers[6])->getBias())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_bias3_1.close(); f_bias3_2.close();
		if (ZEROS)
		{
			generate_zeros("bias3_1", 100, temp);
			generate_zeros("bias3_2", 100, temp);
		}

		/************************** Bias4 **********************************/
		string path_bias4_1 = default_path+"bias4_"+to_string(partyNum);
		string path_bias4_2 = default_path+"bias4_"+to_string(nextParty(partyNum));
		ifstream f_bias4_1(path_bias4_1), f_bias4_2(path_bias4_2);

		for (int i = 0; i < 10; ++i)
		{
			f_bias4_1 >> temp_next; f_bias4_2 >> temp_prev;
			(*((FCLayer*)net->layers[8])->getBias())[i] = std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
		}
		f_bias4_1.close(); f_bias4_2.close();
		if (ZEROS)
		{
			generate_zeros("bias4_1", 10, temp);
			generate_zeros("bias4_2", 10, temp);
		}
	}
	else if (which_network(network).compare("LeNet") == 0)
	{
		string temp = "LeNet";
		/************************** Input **********************************/
		load_input(net, temp, default_path, ZEROS);

		/************************** Weight1 **********************************/
		load_weights<CNNLayer>(net, temp, default_path, 1, 0, 5*5*1*20, ZEROS);

		/************************** Weight2 **********************************/
		load_weights<CNNLayer>(net, temp, default_path, 2, 3, 25*20*50, ZEROS);

		/************************** Weight3 **********************************/
		load_weights<FCLayer>(net, temp, default_path, 3, 6, 500*800, ZEROS);

		/************************** Weight4 **********************************/
		load_weights<FCLayer>(net, temp, default_path, 4, 8, 10*500, ZEROS);

		/************************** Bias1 **********************************/
		load_biases<CNNLayer>(net, temp, default_path, 1, 0, 20, ZEROS);

		/************************** Bias2 **********************************/
		load_biases<CNNLayer>(net, temp, default_path, 2, 3, 50, ZEROS);

		/************************** Bias3 **********************************/
		load_biases<FCLayer>(net, temp, default_path, 3, 6, 500, ZEROS);

		/************************** Bias4 **********************************/
		load_biases<FCLayer>(net, temp, default_path, 4, 8, 10, ZEROS);
	}
	else if (which_network(network).compare("AlexNet") == 0)
	{
		string temp = "AlexNet";
		/************************** Input **********************************/
		load_input(net, temp, default_path, ZEROS);

		// Determine any offsets for enabling BNLayers yay or nay
		#ifndef DISABLE_BN_LAYER
		const size_t bn_layer_size = 1;
		#else
		const size_t bn_layer_size = 0;
		#endif

		// Note: we use `layer_i` below to compute the index of the layers in the config. This is literally the zero-indexed position of the layer in the definition in `selectNetwork`, except that we dynamically take into account whether BNLayers are enabled or not.

		/* Layer 1 */
		size_t layer_i = 0;
		load_layer<CNNLayer>(net, temp, default_path, 1, layer_i, ZEROS);
		/* Layer 2 */
		layer_i += 3 + bn_layer_size;
		load_layer<CNNLayer>(net, temp, default_path, 2, layer_i, ZEROS);
		/* Layer 3 */
		layer_i += 3 + bn_layer_size;
		load_layer<CNNLayer>(net, temp, default_path, 3, layer_i, ZEROS);
		/* Layer 4 */
		layer_i += 2;
		load_layer<CNNLayer>(net, temp, default_path, 4, layer_i, ZEROS);
		/* Layer 5 */
		layer_i += 2;
		load_layer<CNNLayer>(net, temp, default_path, 5, layer_i, ZEROS);

		/* Layer 6 */
		layer_i += 2;
		load_layer<FCLayer>(net, temp, default_path, 6, layer_i, ZEROS);
		/* Layer 7 */
		layer_i += 2;
		load_layer<FCLayer>(net, temp, default_path, 7, layer_i, ZEROS);
		/* Layer 8 */
		layer_i += 2;
		load_layer<FCLayer>(net, temp, default_path, 8, layer_i, ZEROS);
	}
	else 
		error("Preloading network error");



	cout << "Preloading completed..." << endl;
}

void loadData(string net, string dataset)
{
	if (dataset.compare("MNIST") == 0)
	{
		INPUT_SIZE = 784;
		LAST_LAYER_SIZE = 10;
		TRAINING_DATA_SIZE = 8;
		TEST_DATA_SIZE = 8;
		LARGE_NETWORK = false;
	}
	else if (dataset.compare("CIFAR10") == 0)
	{
		LARGE_NETWORK = false;
		if (net.compare("AlexNet") == 0)
		{
			INPUT_SIZE = 33*33*3;
			LAST_LAYER_SIZE = 10;
			TRAINING_DATA_SIZE = 8;
			TEST_DATA_SIZE = 8;			
		}
		else if (net.compare("VGG16") == 0)
		{
			INPUT_SIZE = 32*32*3;
			LAST_LAYER_SIZE = 10;
			TRAINING_DATA_SIZE = 8;
			TEST_DATA_SIZE = 8;	
		}
		else
			assert(false && "Only AlexNet and VGG16 supported on CIFAR10");
	}
	else if (dataset.compare("ImageNet") == 0)
	{
		LARGE_NETWORK = true;
		//https://medium.com/@smallfishbigsea/a-walk-through-of-alexnet-6cbd137a5637
		//https://medium.com/@RaghavPrabhu/cnn-architectures-lenet-alexnet-vgg-googlenet-and-resnet-7c81c017b848
		//https://neurohive.io/en/popular-networks/vgg16/

		//Tiny ImageNet
		//http://cs231n.stanford.edu/reports/2017/pdfs/930.pdf
		//http://cs231n.stanford.edu/reports/2017/pdfs/931.pdf
		if (net.compare("AlexNet") == 0)
		{
			INPUT_SIZE = 56*56*3;
			LAST_LAYER_SIZE = 200;
			TRAINING_DATA_SIZE = 8;
			TEST_DATA_SIZE = 8;			
		}
		else if (net.compare("VGG16") == 0)
		{
			INPUT_SIZE = 64*64*3;
			LAST_LAYER_SIZE = 200;
			TRAINING_DATA_SIZE = 8;
			TEST_DATA_SIZE = 8;			
		}
		else
			assert(false && "Only AlexNet and VGG16 supported on ImageNet");
	}
	else
		assert(false && "Only MNIST, CIFAR10, and ImageNet supported");


	string filename_train_data_next, filename_train_data_prev;
	string filename_test_data_next, filename_test_data_prev;
	string filename_train_labels_next, filename_train_labels_prev;
	string filename_test_labels_next, filename_test_labels_prev;

	// modified to let each party holding a share of data
	char* party_num = NULL;
	if (partyNum == PARTY_A)
	{
		party_num = "A";
		filename_train_data_next = "files/"+dataset+"/train_data_A";
		filename_train_data_prev = "files/"+dataset+"/train_data_B";
		filename_test_data_next = "files/"+dataset+"/test_data_A";
		filename_test_data_prev = "files/"+dataset+"/test_data_B";
		filename_train_labels_next = "files/"+dataset+"/train_labels_A";
		filename_train_labels_prev = "files/"+dataset+"/train_labels_B";
		filename_test_labels_next = "files/"+dataset+"/test_labels_A";
		filename_test_labels_prev = "files/"+dataset+"/test_labels_B";
	} else if (partyNum == PARTY_B)
	{
		party_num = "B";
		filename_train_data_next = "files/"+dataset+"/train_data_B";
		filename_train_data_prev = "files/"+dataset+"/train_data_C";
		filename_test_data_next = "files/"+dataset+"/test_data_B";
		filename_test_data_prev = "files/"+dataset+"/test_data_C";
		filename_train_labels_next = "files/"+dataset+"/train_labels_B";
		filename_train_labels_prev = "files/"+dataset+"/train_labels_C";
		filename_test_labels_next = "files/"+dataset+"/test_labels_B";
		filename_test_labels_prev = "files/"+dataset+"/test_labels_C";
	} else if (partyNum == PARTY_C)
	{
		party_num = "C";
		filename_train_data_next = "files/"+dataset+"/train_data_C";
		filename_train_data_prev = "files/"+dataset+"/train_data_A";
		filename_test_data_next = "files/"+dataset+"/test_data_C";
		filename_test_data_prev = "files/"+dataset+"/test_data_A";
		filename_train_labels_next = "files/"+dataset+"/train_labels_C";
		filename_train_labels_prev = "files/"+dataset+"/train_labels_A";
		filename_test_labels_next = "files/"+dataset+"/test_labels_C";
		filename_test_labels_prev = "files/"+dataset+"/test_labels_A";
	} else {
		party_num = "<unknown party>";
	}

	// float temp_next = 0, temp_prev = 0;
	// OPEN_FILE(party_num, f_next, filename_train_data_next);
	// OPEN_FILE(party_num, f_prev, filename_train_data_prev);
	// for (int i = 0; i < TRAINING_DATA_SIZE * INPUT_SIZE; ++i)
	// {
	// 	f_next >> temp_next; f_prev >> temp_prev;
	// 	trainData.push_back(std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev)));
	// }
	// f_next.close(); f_prev.close();

	// OPEN_FILE(party_num, g_next, filename_train_labels_next);
	// OPEN_FILE(party_num, g_prev, filename_train_labels_prev);
	// for (int i = 0; i < TRAINING_DATA_SIZE * LAST_LAYER_SIZE; ++i)
	// {
	// 	g_next >> temp_next; g_prev >> temp_prev;
	// 	trainLabels.push_back(std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev)));
	// }
	// g_next.close(); g_prev.close();

	// OPEN_FILE(party_num, h_next, filename_test_data_next);
	// OPEN_FILE(party_num, h_prev, filename_test_data_prev);
	// for (int i = 0; i < TEST_DATA_SIZE * INPUT_SIZE; ++i)
	// {
	// 	h_next >> temp_next; h_prev >> temp_prev;
	// 	testData.push_back(std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev)));
	// }
	// h_next.close(); h_prev.close();

	// OPEN_FILE(party_num, k_next, filename_test_labels_next);
	// OPEN_FILE(party_num, k_prev, filename_test_labels_prev);
	// for (int i = 0; i < TEST_DATA_SIZE * LAST_LAYER_SIZE; ++i)
	// {
	// 	k_next >> temp_next; k_prev >> temp_prev;
	// 	testLabels.push_back(std::make_pair(floatToMyType(temp_next), floatToMyType(temp_prev)));
	// }
	// k_next.close(); k_prev.close();

	// Open the training data
	trainDataPrev = ifstream(filename_train_data_prev);
	if (!trainDataPrev.is_open()) { throw std::runtime_error(string(party_num) + ") Failed to open training dataset (prev) '" + filename_train_data_prev + "': " + strerror(errno)); }
	trainDataNext = ifstream(filename_train_data_next);
	if (!trainDataNext.is_open()) { throw std::runtime_error(string(party_num) + ") Failed to open training dataset (next) '" + filename_train_data_next + "': " + strerror(errno)); }

	// Open the test data
	testDataPrev = ifstream(filename_test_data_prev);
	if (!testDataPrev.is_open()) { throw std::runtime_error(string(party_num) + ") Failed to open test dataset (prev) '" + filename_test_data_prev + "': " + strerror(errno)); }
	testDataNext = ifstream(filename_test_data_next);
	if (!testDataNext.is_open()) { throw std::runtime_error(string(party_num) + ") Failed to open test dataset (next) '" + filename_test_data_next + "': " + strerror(errno)); }

	// Open the training labels
	trainLabelsPrev = ifstream(filename_train_labels_prev);
	if (!trainLabelsPrev.is_open()) { throw std::runtime_error(string(party_num) + ") Failed to open training labels (prev) '" + filename_train_labels_prev + "': " + strerror(errno)); }
	trainLabelsNext = ifstream(filename_train_labels_next);
	if (!trainLabelsNext.is_open()) { throw std::runtime_error(string(party_num) + ") Failed to open training labels (next) '" + filename_train_labels_next + "': " + strerror(errno)); }

	// Open the test labels
	testLabelsPrev = ifstream(filename_test_labels_prev);
	if (!testLabelsPrev.is_open()) { throw std::runtime_error(string(party_num) + ") Failed to open test labels (prev) '" + filename_test_labels_prev + "': " + strerror(errno)); }
	testLabelsNext = ifstream(filename_test_labels_next);
	if (!testLabelsNext.is_open()) { throw std::runtime_error(string(party_num) + ") Failed to open test labels (next) '" + filename_test_labels_next + "': " + strerror(errno)); }
}


void readMiniBatch(NeuralNetwork* net, string phase)
{
	// size_t s = trainData.size();
	// size_t t = trainLabels.size();

	// if (phase == "TRAINING")
	// {
	// 	net->inputData.resize(INPUT_SIZE * batch_size);
	// 	for (int i = 0; i < INPUT_SIZE * batch_size; ++i)
	// 		net->inputData[i] = trainData[(trainDataBatchCounter + i)%s];

	// 	net->outputData.resize(LAST_LAYER_SIZE * batch_size);
	// 	for (int i = 0; i < LAST_LAYER_SIZE * batch_size; ++i)
	// 		net->outputData[i] = trainLabels[(trainLabelsBatchCounter + i)%t];

	// 	trainDataBatchCounter += INPUT_SIZE * batch_size;
	// 	trainLabelsBatchCounter += LAST_LAYER_SIZE * batch_size;
	// }

	// if (trainDataBatchCounter > s)
	// 	trainDataBatchCounter -= s;

	// if (trainLabelsBatchCounter > t)
	// 	trainLabelsBatchCounter -= t;



	// size_t p = testData.size();
	// size_t q = testLabels.size();

	// if (phase == "TESTING")
	// {
	// 	net->inputData.resize(INPUT_SIZE * batch_size);
	// 	for (int i = 0; i < INPUT_SIZE * batch_size; ++i)
	// 		net->inputData[i] = testData[(testDataBatchCounter + i)%p];

	// 	net->outputData.resize(LAST_LAYER_SIZE * batch_size);
	// 	for (int i = 0; i < LAST_LAYER_SIZE * batch_size; ++i)
	// 		net->outputData[i] = testLabels[(testLabelsBatchCounter + i)%q];

	// 	testDataBatchCounter += INPUT_SIZE * batch_size;
	// 	testLabelsBatchCounter += LAST_LAYER_SIZE * batch_size;
	// }

	// if (testDataBatchCounter > p)
	// 	testDataBatchCounter -= p;

	// if (testLabelsBatchCounter > q)
	// 	testLabelsBatchCounter -= q;


	// // Always update the layers' input sizes too
	// for (size_t i = 0; i < net->layers.size(); i++) {
	// 	net->layers[i]->setInputRows(batch_size);
	// }

	// Set the handles to read from
	ifstream& dataPrev   = (phase == "TRAINING" ? trainDataPrev : testDataPrev);
	ifstream& dataNext   = (phase == "TRAINING" ? trainDataNext : testDataNext);
	ifstream& labelsPrev = (phase == "TRAINING" ? trainLabelsPrev : testLabelsPrev);
	ifstream& labelsNext = (phase == "TRAINING" ? trainLabelsNext : testLabelsNext);

	// Read the next input batch
	net->inputData.resize(INPUT_SIZE * MINI_BATCH_SIZE);
	float temp_prev, temp_next;
	for (size_t i = 0; i < INPUT_SIZE * MINI_BATCH_SIZE; i++) {
		// Read the two next values from the files
		if (!(dataPrev >> temp_prev)) {
			// Re-try after seeking back to the start
			dataPrev.seekg(streampos(0));
			if (!(dataPrev >> temp_prev)) { throw std::runtime_error("Failed to read next float from " + phase + " data (prev): " + strerror(errno)); }
		}
		if (!(dataNext >> temp_next)) {
			// Re-try after seeking back to the start
			dataNext.seekg(streampos(0));
			if (!(dataNext >> temp_next)) { throw std::runtime_error("Failed to read next float from " + phase + " data (next): " + strerror(errno)); }
		}

		// Store them into the inputData
		net->inputData.at(i) = make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
	}

	// Read the next label batch
	net->outputData.resize(LAST_LAYER_SIZE * MINI_BATCH_SIZE);
	for (size_t i = 0; i < LAST_LAYER_SIZE * MINI_BATCH_SIZE; i++) {
		// Read the two next values from the files
		if (!(labelsPrev >> temp_prev)) {
			// Re-try after seeking back to the start
			labelsPrev.seekg(streampos(0));
			if (!(labelsPrev >> temp_prev)) { throw std::runtime_error("Failed to read next float from " + phase + " labels (prev): " + strerror(errno)); }
		}
		if (!(labelsNext >> temp_next)) {
			// Re-try after seeking back to the start
			labelsNext.seekg(streampos(0));
			if (!(labelsNext >> temp_next)) { throw std::runtime_error("Failed to read next float from " + phase + " labels (next): " + strerror(errno)); }
		}

		// Store them into the inputData
		net->outputData.at(i) = make_pair(floatToMyType(temp_next), floatToMyType(temp_prev));
	}
}

void printNetwork(NeuralNetwork* net)
{
	for (int i = 0; i < net->layers.size(); ++i)
		net->layers[i]->printLayer();
	cout << "----------------------------------------------" << endl;  	
}


void selectNetwork(string network, string dataset, string security, NeuralNetConfig* config)
{
	assert(((security.compare("Semi-honest") == 0) or (security.compare("Malicious") == 0)) && 
			"Only Semi-honest or Malicious security allowed");
	SECURITY_TYPE = security;
	loadData(network, dataset);

	if (network.compare("SecureML") == 0)
	{
		assert((dataset.compare("MNIST") == 0) && "SecureML only over MNIST");
		NUM_LAYERS = 6;
		WITH_NORMALIZATION = true;
		FCConfig* l0 = new FCConfig(784, MINI_BATCH_SIZE, 128); 
		ReLUConfig* l1 = new ReLUConfig(128, MINI_BATCH_SIZE);
		FCConfig* l2 = new FCConfig(128, MINI_BATCH_SIZE, 128); 
		ReLUConfig* l3 = new ReLUConfig(128, MINI_BATCH_SIZE);
		FCConfig* l4 = new FCConfig(128, MINI_BATCH_SIZE, 10); 
		ReLUConfig* l5 = new ReLUConfig(10, MINI_BATCH_SIZE);
		// BNConfig* l6 = new BNConfig(10, MINI_BATCH_SIZE);
		config->addLayer(l0);
		config->addLayer(l1);
		config->addLayer(l2);
		config->addLayer(l3);
		config->addLayer(l4);
		config->addLayer(l5);
		// config->addLayer(l6);
	}
	else if (network.compare("Sarda") == 0)
	{
		assert((dataset.compare("MNIST") == 0) && "Sarda only over MNIST");
		NUM_LAYERS = 5;
		WITH_NORMALIZATION = true;
		CNNConfig* l0 = new CNNConfig(28,28,1,5,2,2,0,MINI_BATCH_SIZE);
		ReLUConfig* l1 = new ReLUConfig(980, MINI_BATCH_SIZE);
		FCConfig* l2 = new FCConfig(980, MINI_BATCH_SIZE, 100);
		ReLUConfig* l3 = new ReLUConfig(100, MINI_BATCH_SIZE);
		FCConfig* l4 = new FCConfig(100, MINI_BATCH_SIZE, 10);
		config->addLayer(l0);
		config->addLayer(l1);
		config->addLayer(l2);
		config->addLayer(l3);
		config->addLayer(l4);
	}
	else if (network.compare("MiniONN") == 0)
	{
		assert((dataset.compare("MNIST") == 0) && "MiniONN only over MNIST");
		NUM_LAYERS = 10;
		WITH_NORMALIZATION = true;
		CNNConfig* l0 = new CNNConfig(28,28,1,16,5,1,0,MINI_BATCH_SIZE);
		MaxpoolConfig* l1 = new MaxpoolConfig(24,24,16,2,2,MINI_BATCH_SIZE);
		ReLUConfig* l2 = new ReLUConfig(12*12*16, MINI_BATCH_SIZE);
		CNNConfig* l3 = new CNNConfig(12,12,16,16,5,1,0,MINI_BATCH_SIZE);
		MaxpoolConfig* l4 = new MaxpoolConfig(8,8,16,2,2,MINI_BATCH_SIZE);
		ReLUConfig* l5 = new ReLUConfig(4*4*16, MINI_BATCH_SIZE);
		FCConfig* l6 = new FCConfig(4*4*16, MINI_BATCH_SIZE, 100);
		ReLUConfig* l7 = new ReLUConfig(100, MINI_BATCH_SIZE);
		FCConfig* l8 = new FCConfig(100, MINI_BATCH_SIZE, 10);
		ReLUConfig* l9 = new ReLUConfig(10, MINI_BATCH_SIZE);
		config->addLayer(l0);
		config->addLayer(l1);
		config->addLayer(l2);
		config->addLayer(l3);
		config->addLayer(l4);
		config->addLayer(l5);
		config->addLayer(l6);
		config->addLayer(l7);
		config->addLayer(l8);
		config->addLayer(l9);
	}
	else if (network.compare("LeNet") == 0)
	{
		assert((dataset.compare("MNIST") == 0) && "LeNet only over MNIST");
		NUM_LAYERS = 10;
		WITH_NORMALIZATION = true;
		CNNConfig* l0 = new CNNConfig(28,28,1,20,5,1,0,MINI_BATCH_SIZE);
		MaxpoolConfig* l1 = new MaxpoolConfig(24,24,20,2,2,MINI_BATCH_SIZE);
		ReLUConfig* l2 = new ReLUConfig(12*12*20, MINI_BATCH_SIZE);
		CNNConfig* l3 = new CNNConfig(12,12,20,50,5,1,0,MINI_BATCH_SIZE);
		MaxpoolConfig* l4 = new MaxpoolConfig(8,8,50,2,2,MINI_BATCH_SIZE);
		ReLUConfig* l5 = new ReLUConfig(4*4*50, MINI_BATCH_SIZE);
		FCConfig* l6 = new FCConfig(4*4*50, MINI_BATCH_SIZE, 500);
		ReLUConfig* l7 = new ReLUConfig(500, MINI_BATCH_SIZE);
		FCConfig* l8 = new FCConfig(500, MINI_BATCH_SIZE, 10);
		ReLUConfig* l9 = new ReLUConfig(10, MINI_BATCH_SIZE);
		config->addLayer(l0);
		config->addLayer(l1);
		config->addLayer(l2);
		config->addLayer(l3);
		config->addLayer(l4);
		config->addLayer(l5);
		config->addLayer(l6);
		config->addLayer(l7);
		config->addLayer(l8);
		config->addLayer(l9);
	}
	else if (network.compare("AlexNet") == 0)
	{
		if(dataset.compare("MNIST") == 0) {
			NUM_LAYERS = 20;
			// NUM_LAYERS = 18;		//Without BN
			WITH_NORMALIZATION = false;
			CNNConfig* l0 = new CNNConfig(28,28,1,96,11,1,5,MINI_BATCH_SIZE);
			MaxpoolConfig* l1 = new MaxpoolConfig(11,11,96,3,2,MINI_BATCH_SIZE);
			ReLUConfig* l2 = new ReLUConfig(5*5*96,MINI_BATCH_SIZE);
			BNConfig * l3 = new BNConfig(5*5*96,MINI_BATCH_SIZE);

			CNNConfig* l4 = new CNNConfig(5,5,96,256,5,1,1,MINI_BATCH_SIZE);
			MaxpoolConfig* l5 = new MaxpoolConfig(3,3,256,3,2,MINI_BATCH_SIZE);
			ReLUConfig* l6 = new ReLUConfig(1*1*256,MINI_BATCH_SIZE);
			BNConfig * l7 = new BNConfig(1*1*256,MINI_BATCH_SIZE);

			CNNConfig* l8 = new CNNConfig(1,1,256,384,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l9 = new ReLUConfig(1*1*384,MINI_BATCH_SIZE);
			CNNConfig* l10 = new CNNConfig(1,1,384,384,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l11 = new ReLUConfig(1*1*384,MINI_BATCH_SIZE);
			CNNConfig* l12 = new CNNConfig(1,1,384,256,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l13 = new ReLUConfig(1*1*256,MINI_BATCH_SIZE);

			FCConfig* l14 = new FCConfig(1*1*256,MINI_BATCH_SIZE,256);
			ReLUConfig* l15 = new ReLUConfig(256,MINI_BATCH_SIZE);
			FCConfig* l16 = new FCConfig(256,MINI_BATCH_SIZE,256);
			ReLUConfig* l17 = new ReLUConfig(256,MINI_BATCH_SIZE);
			FCConfig* l18 = new FCConfig(256,MINI_BATCH_SIZE,10);
			ReLUConfig* l19 = new ReLUConfig(10,MINI_BATCH_SIZE);
			config->addLayer(l0);
			config->addLayer(l1);
			config->addLayer(l2);
			config->addLayer(l3);
			config->addLayer(l4);
			config->addLayer(l5);
			config->addLayer(l6);
			config->addLayer(l7);
			config->addLayer(l8);
			config->addLayer(l9);
			config->addLayer(l10);
			config->addLayer(l11);
			config->addLayer(l12);
			config->addLayer(l13);
			config->addLayer(l14);
			config->addLayer(l15);
			config->addLayer(l16);
			config->addLayer(l17);
			config->addLayer(l18);
			config->addLayer(l19);

		} else if (dataset.compare("CIFAR10") == 0) {
			#ifndef DISABLE_BN_LAYER
			NUM_LAYERS = 20;
			#else
			NUM_LAYERS = 18;		//Without BN
			#endif

			WITH_NORMALIZATION = false;
			CNNConfig* l0 = new CNNConfig(33,33,3,96,11,4,9,MINI_BATCH_SIZE);
			MaxpoolConfig* l1 = new MaxpoolConfig(11,11,96,3,2,MINI_BATCH_SIZE);
			ReLUConfig* l2 = new ReLUConfig(5*5*96,MINI_BATCH_SIZE);
			#ifndef DISABLE_BN_LAYER
			BNConfig * l3 = new BNConfig(5*5*96,MINI_BATCH_SIZE);
			#endif

			CNNConfig* l4 = new CNNConfig(5,5,96,256,5,1,1,MINI_BATCH_SIZE);
			MaxpoolConfig* l5 = new MaxpoolConfig(3,3,256,3,2,MINI_BATCH_SIZE);
			ReLUConfig* l6 = new ReLUConfig(1*1*256,MINI_BATCH_SIZE);
			#ifndef DISABLE_BN_LAYER
			BNConfig * l7 = new BNConfig(1*1*256,MINI_BATCH_SIZE);
			#endif

			CNNConfig* l8 = new CNNConfig(1,1,256,384,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l9 = new ReLUConfig(1*1*384,MINI_BATCH_SIZE);
			CNNConfig* l10 = new CNNConfig(1,1,384,384,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l11 = new ReLUConfig(1*1*384,MINI_BATCH_SIZE);
			CNNConfig* l12 = new CNNConfig(1,1,384,256,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l13 = new ReLUConfig(1*1*256,MINI_BATCH_SIZE);

			FCConfig* l14 = new FCConfig(1*1*256,MINI_BATCH_SIZE,256);
			ReLUConfig* l15 = new ReLUConfig(256,MINI_BATCH_SIZE);
			FCConfig* l16 = new FCConfig(256,MINI_BATCH_SIZE,256);
			ReLUConfig* l17 = new ReLUConfig(256,MINI_BATCH_SIZE);
			FCConfig* l18 = new FCConfig(256,MINI_BATCH_SIZE,10);
			ReLUConfig* l19 = new ReLUConfig(10,MINI_BATCH_SIZE);
			config->addLayer(l0);
			config->addLayer(l1);
			config->addLayer(l2);
			#ifndef DISABLE_BN_LAYER
			config->addLayer(l3);
			#endif
			config->addLayer(l4);
			config->addLayer(l5);
			config->addLayer(l6);
			#ifndef DISABLE_BN_LAYER
			config->addLayer(l7);
			#endif
			config->addLayer(l8);
			config->addLayer(l9);
			config->addLayer(l10);
			config->addLayer(l11);
			config->addLayer(l12);
			config->addLayer(l13);
			config->addLayer(l14);
			config->addLayer(l15);
			config->addLayer(l16);
			config->addLayer(l17);
			config->addLayer(l18);
			config->addLayer(l19);
		}
		else if (dataset.compare("ImageNet") == 0)
		{
			NUM_LAYERS = 19;
			// NUM_LAYERS = 17;		//Without BN
			WITH_NORMALIZATION = false;
			CNNConfig* l0 = new CNNConfig(56,56,3,64,7,1,3,MINI_BATCH_SIZE);
			CNNConfig* l1 = new CNNConfig(56,56,64,64,5,1,2,MINI_BATCH_SIZE);
			MaxpoolConfig* l2 = new MaxpoolConfig(56,56,64,2,2,MINI_BATCH_SIZE);
			ReLUConfig* l3 = new ReLUConfig(28*28*64,MINI_BATCH_SIZE);		
			BNConfig * l4 = new BNConfig(28*28*64,MINI_BATCH_SIZE);

			CNNConfig* l5 = new CNNConfig(28,28,64,128,5,1,2,MINI_BATCH_SIZE);
			MaxpoolConfig* l6 = new MaxpoolConfig(28,28,128,2,2,MINI_BATCH_SIZE);
			ReLUConfig* l7 = new ReLUConfig(14*14*128,MINI_BATCH_SIZE);		
			BNConfig * l8 = new BNConfig(14*14*128,MINI_BATCH_SIZE);

			CNNConfig* l9 = new CNNConfig(14,14,128,256,3,1,1,MINI_BATCH_SIZE);
			CNNConfig* l10 = new CNNConfig(14,14,256,256,3,1,1,MINI_BATCH_SIZE);
			MaxpoolConfig* l11 = new MaxpoolConfig(14,14,256,2,2,MINI_BATCH_SIZE);
			ReLUConfig* l12 = new ReLUConfig(7*7*256,MINI_BATCH_SIZE);

			FCConfig* l13 = new FCConfig(7*7*256,MINI_BATCH_SIZE,1024);
			ReLUConfig* l14 = new ReLUConfig(1024,MINI_BATCH_SIZE);
			FCConfig* l15 = new FCConfig(1024,MINI_BATCH_SIZE,1024);
			ReLUConfig* l16 = new ReLUConfig(1024,MINI_BATCH_SIZE);
			FCConfig* l17 = new FCConfig(1024,MINI_BATCH_SIZE,200);
			ReLUConfig* l18 = new ReLUConfig(200,MINI_BATCH_SIZE);
			config->addLayer(l0);
			config->addLayer(l1);
			config->addLayer(l2);
			config->addLayer(l3);
			config->addLayer(l4);
			config->addLayer(l5);
			config->addLayer(l6);
			config->addLayer(l7);
			config->addLayer(l8);
			config->addLayer(l9);
			config->addLayer(l10);
			config->addLayer(l11);
			config->addLayer(l12);
			config->addLayer(l13);
			config->addLayer(l14);
			config->addLayer(l15);
			config->addLayer(l16);
			config->addLayer(l17);
			config->addLayer(l18);
		}
	}
	else if (network.compare("VGG16") == 0)
	{
		if(dataset.compare("MNIST") == 0)
			assert(false && "No VGG16 on MNIST");
		else if (dataset.compare("CIFAR10") == 0)
		{
			NUM_LAYERS = 37;
			WITH_NORMALIZATION = false;
			CNNConfig* l0 = new CNNConfig(32,32,3,64,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l1 = new ReLUConfig(32*32*64,MINI_BATCH_SIZE);		
			CNNConfig* l2 = new CNNConfig(32,32,64,64,3,1,1,MINI_BATCH_SIZE);
			MaxpoolConfig* l3 = new MaxpoolConfig(32,32,64,2,2,MINI_BATCH_SIZE);
			ReLUConfig* l4 = new ReLUConfig(16*16*64,MINI_BATCH_SIZE);

			CNNConfig* l5 = new CNNConfig(16,16,64,128,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l6 = new ReLUConfig(16*16*128,MINI_BATCH_SIZE);
			CNNConfig* l7 = new CNNConfig(16,16,128,128,3,1,1,MINI_BATCH_SIZE);
			MaxpoolConfig* l8 = new MaxpoolConfig(16,16,128,2,2,MINI_BATCH_SIZE);
			ReLUConfig* l9 = new ReLUConfig(8*8*128,MINI_BATCH_SIZE);

			CNNConfig* l10 = new CNNConfig(8,8,128,256,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l11 = new ReLUConfig(8*8*256,MINI_BATCH_SIZE);
			CNNConfig* l12 = new CNNConfig(8,8,256,256,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l13 = new ReLUConfig(8*8*256,MINI_BATCH_SIZE);
			CNNConfig* l14 = new CNNConfig(8,8,256,256,3,1,1,MINI_BATCH_SIZE);
			MaxpoolConfig* l15 = new MaxpoolConfig(8,8,256,2,2,MINI_BATCH_SIZE);
			ReLUConfig* l16 = new ReLUConfig(4*4*256,MINI_BATCH_SIZE);

			CNNConfig* l17 = new CNNConfig(4,4,256,512,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l18 = new ReLUConfig(4*4*512,MINI_BATCH_SIZE);
			CNNConfig* l19 = new CNNConfig(4,4,512,512,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l20 = new ReLUConfig(4*4*512,MINI_BATCH_SIZE);
			CNNConfig* l21 = new CNNConfig(4,4,512,512,3,1,1,MINI_BATCH_SIZE);
			MaxpoolConfig* l22 = new MaxpoolConfig(4,4,512,2,2,MINI_BATCH_SIZE);
			ReLUConfig* l23 = new ReLUConfig(2*2*512,MINI_BATCH_SIZE);

			CNNConfig* l24 = new CNNConfig(2,2,512,512,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l25 = new ReLUConfig(2*2*512,MINI_BATCH_SIZE);
			CNNConfig* l26 = new CNNConfig(2,2,512,512,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l27 = new ReLUConfig(2*2*512,MINI_BATCH_SIZE);
			CNNConfig* l28 = new CNNConfig(2,2,512,512,3,1,1,MINI_BATCH_SIZE);
			MaxpoolConfig* l29 = new MaxpoolConfig(2,2,512,2,2,MINI_BATCH_SIZE);
			ReLUConfig* l30 = new ReLUConfig(1*1*512,MINI_BATCH_SIZE);

			FCConfig* l31 = new FCConfig(1*1*512,MINI_BATCH_SIZE,4096);
			ReLUConfig* l32 = new ReLUConfig(4096,MINI_BATCH_SIZE);
			FCConfig* l33 = new FCConfig(4096, MINI_BATCH_SIZE, 4096);
			ReLUConfig* l34 = new ReLUConfig(4096, MINI_BATCH_SIZE);
			FCConfig* l35 = new FCConfig(4096, MINI_BATCH_SIZE, 1000);
			ReLUConfig* l36 = new ReLUConfig(1000, MINI_BATCH_SIZE);
			config->addLayer(l0);
			config->addLayer(l1);
			config->addLayer(l2);
			config->addLayer(l3);
			config->addLayer(l4);
			config->addLayer(l5);
			config->addLayer(l6);
			config->addLayer(l7);
			config->addLayer(l8);
			config->addLayer(l9);
			config->addLayer(l10);
			config->addLayer(l11);
			config->addLayer(l12);
			config->addLayer(l13);
			config->addLayer(l14);
			config->addLayer(l15);
			config->addLayer(l16);
			config->addLayer(l17);
			config->addLayer(l18);
			config->addLayer(l19);
			config->addLayer(l20);
			config->addLayer(l21);
			config->addLayer(l22);
			config->addLayer(l23);
			config->addLayer(l24);
			config->addLayer(l25);
			config->addLayer(l26);
			config->addLayer(l27);
			config->addLayer(l28);
			config->addLayer(l29);
			config->addLayer(l30);
			config->addLayer(l31);
			config->addLayer(l32);
			config->addLayer(l33);
			config->addLayer(l34);
			config->addLayer(l35);
			config->addLayer(l36);
		}
		else if (dataset.compare("ImageNet") == 0)
		{
			NUM_LAYERS = 37;
			WITH_NORMALIZATION = false;
			CNNConfig* l0 = new CNNConfig(64,64,3,64,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l1 = new ReLUConfig(64*64*64,MINI_BATCH_SIZE);		
			CNNConfig* l2 = new CNNConfig(64,64,64,64,3,1,1,MINI_BATCH_SIZE);
			MaxpoolConfig* l3 = new MaxpoolConfig(64,64,64,2,2,MINI_BATCH_SIZE);
			ReLUConfig* l4 = new ReLUConfig(32*32*64,MINI_BATCH_SIZE);

			CNNConfig* l5 = new CNNConfig(32,32,64,128,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l6 = new ReLUConfig(32*32*128,MINI_BATCH_SIZE);
			CNNConfig* l7 = new CNNConfig(32,32,128,128,3,1,1,MINI_BATCH_SIZE);
			MaxpoolConfig* l8 = new MaxpoolConfig(32,32,128,2,2,MINI_BATCH_SIZE);
			ReLUConfig* l9 = new ReLUConfig(16*16*128,MINI_BATCH_SIZE);

			CNNConfig* l10 = new CNNConfig(16,16,128,256,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l11 = new ReLUConfig(16*16*256,MINI_BATCH_SIZE);
			CNNConfig* l12 = new CNNConfig(16,16,256,256,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l13 = new ReLUConfig(16*16*256,MINI_BATCH_SIZE);
			CNNConfig* l14 = new CNNConfig(16,16,256,256,3,1,1,MINI_BATCH_SIZE);
			MaxpoolConfig* l15 = new MaxpoolConfig(16,16,256,2,2,MINI_BATCH_SIZE);
			ReLUConfig* l16 = new ReLUConfig(8*8*256,MINI_BATCH_SIZE);

			CNNConfig* l17 = new CNNConfig(8,8,256,512,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l18 = new ReLUConfig(8*8*512,MINI_BATCH_SIZE);
			CNNConfig* l19 = new CNNConfig(8,8,512,512,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l20 = new ReLUConfig(8*8*512,MINI_BATCH_SIZE);
			CNNConfig* l21 = new CNNConfig(8,8,512,512,3,1,1,MINI_BATCH_SIZE);
			MaxpoolConfig* l22 = new MaxpoolConfig(8,8,512,2,2,MINI_BATCH_SIZE);
			ReLUConfig* l23 = new ReLUConfig(4*4*512,MINI_BATCH_SIZE);

			CNNConfig* l24 = new CNNConfig(4,4,512,512,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l25 = new ReLUConfig(4*4*512,MINI_BATCH_SIZE);
			CNNConfig* l26 = new CNNConfig(4,4,512,512,3,1,1,MINI_BATCH_SIZE);
			ReLUConfig* l27 = new ReLUConfig(4*4*512,MINI_BATCH_SIZE);
			CNNConfig* l28 = new CNNConfig(4,4,512,512,3,1,1,MINI_BATCH_SIZE);
			MaxpoolConfig* l29 = new MaxpoolConfig(4,4,512,2,2,MINI_BATCH_SIZE);
			ReLUConfig* l30 = new ReLUConfig(2*2*512,MINI_BATCH_SIZE);

			FCConfig* l31 = new FCConfig(2*2*512,MINI_BATCH_SIZE,2048);
			ReLUConfig* l32 = new ReLUConfig(2048,MINI_BATCH_SIZE);
			FCConfig* l33 = new FCConfig(2048, MINI_BATCH_SIZE, 2048);
			ReLUConfig* l34 = new ReLUConfig(2048, MINI_BATCH_SIZE);
			FCConfig* l35 = new FCConfig(2048, MINI_BATCH_SIZE, 200);
			ReLUConfig* l36 = new ReLUConfig(200, MINI_BATCH_SIZE);
			config->addLayer(l0);
			config->addLayer(l1);
			config->addLayer(l2);
			config->addLayer(l3);
			config->addLayer(l4);
			config->addLayer(l5);
			config->addLayer(l6);
			config->addLayer(l7);
			config->addLayer(l8);
			config->addLayer(l9);
			config->addLayer(l10);
			config->addLayer(l11);
			config->addLayer(l12);
			config->addLayer(l13);
			config->addLayer(l14);
			config->addLayer(l15);
			config->addLayer(l16);
			config->addLayer(l17);
			config->addLayer(l18);
			config->addLayer(l19);
			config->addLayer(l20);
			config->addLayer(l21);
			config->addLayer(l22);
			config->addLayer(l23);
			config->addLayer(l24);
			config->addLayer(l25);
			config->addLayer(l26);
			config->addLayer(l27);
			config->addLayer(l28);
			config->addLayer(l29);
			config->addLayer(l30);
			config->addLayer(l31);
			config->addLayer(l32);
			config->addLayer(l33);
			config->addLayer(l34);
			config->addLayer(l35);
			config->addLayer(l36);
		}
	}
	else
		assert(false && "Only SecureML, Sarda, Gazelle, LeNet, AlexNet, and VGG16 Networks supported");
}

void runOnly(NeuralNetwork* net, size_t l, string what, string& network)
{
	size_t total_layers = net->layers.size();
	assert((l >= 0 and l < total_layers) && "Incorrect layer number for runOnly"); 
	network = network + " L" + std::to_string(l) + " " + what;

	if (what.compare("F") == 0)
	{
		if (l == 0)
			net->layers[0]->forward(net->inputData);
		else
			net->layers[l]->forward(*(net->layers[l-1]->getActivation()));
	}
	else if (what.compare("D") == 0)
	{
		if (l != 0)
			net->layers[l]->computeDelta(*(net->layers[l-1]->getDelta()));	
	}
	else if (what.compare("U") == 0)
	{
		if (l == 0)
			net->layers[0]->updateEquations(net->inputData);
		else
			net->layers[l]->updateEquations(*(net->layers[l-1]->getActivation()));
	}
	else
		assert(false && "Only F,D or U allowed in runOnly");
}






/********************* COMMUNICATION AND HELPERS *********************/

void start_m()
{
	// cout << endl;
	start_time();
	start_communication();
}

void end_m(string str)
{
	end_time(str);
	pause_communication();
	aggregateCommunication();
	end_communication(str);
}

void start_time()
{
	if (alreadyMeasuringTime)
	{
		cout << "Nested timing measurements" << endl;
		exit(-1);
	}

	tStart = clock();
	clock_gettime(CLOCK_REALTIME, &requestStart);
	alreadyMeasuringTime = true;
}

void end_time(string str)
{
	if (!alreadyMeasuringTime)
	{
		cout << "start_time() never called" << endl;
		exit(-1);
	}

	clock_gettime(CLOCK_REALTIME, &requestEnd);
	cout << "----------------------------------------------" << endl;
	cout << "Wall Clock time for " << str << ": " << diff(requestStart, requestEnd) << " sec\n";
	cout << "CPU time for " << str << ": " << (double)(clock() - tStart)/CLOCKS_PER_SEC << " sec\n";
	cout << "----------------------------------------------" << endl;	
	alreadyMeasuringTime = false;
}


void start_rounds()
{
	if (alreadyMeasuringRounds)
	{
		cout << "Nested round measurements" << endl;
		exit(-1);
	}

	roundComplexitySend = 0;
	roundComplexityRecv = 0;
	alreadyMeasuringRounds = true;
}

void end_rounds(string str)
{
	if (!alreadyMeasuringTime)
	{
		cout << "start_rounds() never called" << endl;
		exit(-1);
	}

	cout << "----------------------------------------------" << endl;
	cout << "Send Round Complexity of " << str << ": " << roundComplexitySend << endl;
	cout << "Recv Round Complexity of " << str << ": " << roundComplexityRecv << endl;
	cout << "----------------------------------------------" << endl;	
	alreadyMeasuringRounds = false;
}

void aggregateCommunication()
{
	vector<myType> vec(4, 0), temp(4, 0);
	vec[0] = commObject.getSent();
	vec[1] = commObject.getRecv();
	vec[2] = commObject.getRoundsSent();
	vec[3] = commObject.getRoundsRecv();

	if (partyNum == PARTY_B or partyNum == PARTY_C)
		sendVector<myType>(vec, PARTY_A, 4);

	if (partyNum == PARTY_A)
	{
		receiveVector<myType>(temp, PARTY_B, 4);
		for (size_t i = 0; i < 4; ++i)
			vec[i] = temp[i] + vec[i];
		receiveVector<myType>(temp, PARTY_C, 4);
		for (size_t i = 0; i < 4; ++i)
			vec[i] = temp[i] + vec[i];
	}

	if (partyNum == PARTY_A)
	{
		cout << "----------------------------------------------" << endl;
		cout << "Total communication: " << (float)vec[0]/1000000 << "MB (sent) and " << (float)vec[1]/1000000 << "MB (recv)\n";
		cout << "Total calls: " << vec[2] << " (sends) and " << vec[3] << " (recvs)" << endl;
		cout << "----------------------------------------------" << endl;
	}
}


void print_usage (const char * bin) 
{
    cout << "Usage: ./" << bin << " PARTY_NUM IP_ADDR_FILE AES_SEED_INDEP AES_SEED_NEXT AES_SEED_PREV" << endl;
    cout << endl;
    cout << "Required Arguments:\n";
    cout << "PARTY_NUM			Party Identifier (0,1, or 2)\n";
    cout << "IP_ADDR_FILE		\tIP Address file (use makefile for automation)\n";
    cout << "AES_SEED_INDEP		\tAES seed file independent\n";
    cout << "AES_SEED_NEXT		\t \tAES seed file next\n";
    cout << "AES_SEED_PREV		\t \tAES seed file previous\n";
    cout << endl;
    cout << "Report bugs to swagh@princeton.edu" << endl;
    exit(-1);
}

double diff(timespec start, timespec end)
{
    timespec temp;

    if ((end.tv_nsec-start.tv_nsec)<0)
    {
            temp.tv_sec = end.tv_sec-start.tv_sec-1;
            temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    }
    else 
    {
            temp.tv_sec = end.tv_sec-start.tv_sec;
            temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    return temp.tv_sec + (double)temp.tv_nsec/NANOSECONDS_PER_SEC;
}


void deleteObjects()
{
	//close connection
	for (int i = 0; i < NUM_OF_PARTIES; i++)
	{
		if (i != partyNum)
		{
			delete communicationReceivers[i];
			delete communicationSenders[i];
		}
	}
	delete[] communicationReceivers;
	delete[] communicationSenders;
	delete[] addrs;
}


/************************ AlexNet on ImageNet ************************/
// NUM_LAYERS = 21;
// WITH_NORMALIZATION = false;
// CNNConfig* l0 = new CNNConfig(227,227,3,96,11,4,0,MINI_BATCH_SIZE);
// MaxpoolConfig* l1 = new MaxpoolConfig(55,55,96,3,2,MINI_BATCH_SIZE);
// ReLUConfig* l2 = new ReLUConfig(27*27*96,MINI_BATCH_SIZE);		
// BNConfig * l3 = new BNConfig(27*27*96,MINI_BATCH_SIZE);

// CNNConfig* l4 = new CNNConfig(27,27,96,256,5,1,2,MINI_BATCH_SIZE);
// MaxpoolConfig* l5 = new MaxpoolConfig(27,27,256,3,2,MINI_BATCH_SIZE);
// ReLUConfig* l6 = new ReLUConfig(13*13*256,MINI_BATCH_SIZE);		
// BNConfig * l7 = new BNConfig(13*13*256,MINI_BATCH_SIZE);

// CNNConfig* l8 = new CNNConfig(13,13,256,384,3,1,1,MINI_BATCH_SIZE);
// ReLUConfig* l9 = new ReLUConfig(13*13*384,MINI_BATCH_SIZE);
// CNNConfig* l10 = new CNNConfig(13,13,384,384,3,1,1,MINI_BATCH_SIZE);
// ReLUConfig* l11 = new ReLUConfig(13*13*384,MINI_BATCH_SIZE);
// CNNConfig* l12 = new CNNConfig(13,13,384,256,3,1,1,MINI_BATCH_SIZE);
// MaxpoolConfig* l13 = new MaxpoolConfig(13,13,256,3,2,MINI_BATCH_SIZE);
// ReLUConfig* l14 = new ReLUConfig(6*6*256,MINI_BATCH_SIZE);

// FCConfig* l15 = new FCConfig(6*6*256,MINI_BATCH_SIZE,4096);
// ReLUConfig* l16 = new ReLUConfig(4096,MINI_BATCH_SIZE);
// FCConfig* l17 = new FCConfig(4096,MINI_BATCH_SIZE,4096);
// ReLUConfig* l18 = new ReLUConfig(4096,MINI_BATCH_SIZE);
// FCConfig* l19 = new FCConfig(4096,MINI_BATCH_SIZE,1000);
// ReLUConfig* l20 = new ReLUConfig(1000,MINI_BATCH_SIZE);
// config->addLayer(l0);
// config->addLayer(l1);
// config->addLayer(l2);
// config->addLayer(l3);
// config->addLayer(l4);
// config->addLayer(l5);
// config->addLayer(l6);
// config->addLayer(l7);
// config->addLayer(l8);
// config->addLayer(l9);
// config->addLayer(l10);
// config->addLayer(l11);
// config->addLayer(l12);
// config->addLayer(l13);
// config->addLayer(l14);
// config->addLayer(l15);
// config->addLayer(l16);
// config->addLayer(l17);
// config->addLayer(l18);
// config->addLayer(l19);
// config->addLayer(l20);


/************************ VGG16 on ImageNet ************************/
// NUM_LAYERS = 37;
// WITH_NORMALIZATION = false;
// CNNConfig* l0 = new CNNConfig(224,224,3,64,3,1,1,MINI_BATCH_SIZE);
// ReLUConfig* l1 = new ReLUConfig(224*224*64,MINI_BATCH_SIZE);		
// CNNConfig* l2 = new CNNConfig(224,224,64,64,3,1,1,MINI_BATCH_SIZE);
// MaxpoolConfig* l3 = new MaxpoolConfig(224,224,64,2,2,MINI_BATCH_SIZE);
// ReLUConfig* l4 = new ReLUConfig(112*112*64,MINI_BATCH_SIZE);

// CNNConfig* l5 = new CNNConfig(112,112,64,128,3,1,1,MINI_BATCH_SIZE);
// ReLUConfig* l6 = new ReLUConfig(112*112*128,MINI_BATCH_SIZE);
// CNNConfig* l7 = new CNNConfig(112,112,128,128,3,1,1,MINI_BATCH_SIZE);
// MaxpoolConfig* l8 = new MaxpoolConfig(112,112,128,2,2,MINI_BATCH_SIZE);
// ReLUConfig* l9 = new ReLUConfig(56*56*128,MINI_BATCH_SIZE);

// CNNConfig* l10 = new CNNConfig(56,56,128,256,3,1,1,MINI_BATCH_SIZE);
// ReLUConfig* l11 = new ReLUConfig(56*56*256,MINI_BATCH_SIZE);
// CNNConfig* l12 = new CNNConfig(56,56,256,256,3,1,1,MINI_BATCH_SIZE);
// ReLUConfig* l13 = new ReLUConfig(56*56*256,MINI_BATCH_SIZE);
// CNNConfig* l14 = new CNNConfig(56,56,256,256,3,1,1,MINI_BATCH_SIZE);
// MaxpoolConfig* l15 = new MaxpoolConfig(56,56,256,2,2,MINI_BATCH_SIZE);
// ReLUConfig* l16 = new ReLUConfig(28*28*256,MINI_BATCH_SIZE);

// CNNConfig* l17 = new CNNConfig(28,28,256,512,3,1,1,MINI_BATCH_SIZE);
// ReLUConfig* l18 = new ReLUConfig(28*28*512,MINI_BATCH_SIZE);
// CNNConfig* l19 = new CNNConfig(28,28,512,512,3,1,1,MINI_BATCH_SIZE);
// ReLUConfig* l20 = new ReLUConfig(28*28*512,MINI_BATCH_SIZE);
// CNNConfig* l21 = new CNNConfig(28,28,512,512,3,1,1,MINI_BATCH_SIZE);
// MaxpoolConfig* l22 = new MaxpoolConfig(28,28,512,2,2,MINI_BATCH_SIZE);
// ReLUConfig* l23 = new ReLUConfig(14*14*512,MINI_BATCH_SIZE);

// CNNConfig* l24 = new CNNConfig(14,14,512,512,3,1,1,MINI_BATCH_SIZE);
// ReLUConfig* l25 = new ReLUConfig(14*14*512,MINI_BATCH_SIZE);
// CNNConfig* l26 = new CNNConfig(14,14,512,512,3,1,1,MINI_BATCH_SIZE);
// ReLUConfig* l27 = new ReLUConfig(14*14*512,MINI_BATCH_SIZE);
// CNNConfig* l28 = new CNNConfig(14,14,512,512,3,1,1,MINI_BATCH_SIZE);
// MaxpoolConfig* l29 = new MaxpoolConfig(14,14,512,2,2,MINI_BATCH_SIZE);
// ReLUConfig* l30 = new ReLUConfig(7*7*512,MINI_BATCH_SIZE);

// FCConfig* l31 = new FCConfig(7*7*512,MINI_BATCH_SIZE,4096);
// ReLUConfig* l32 = new ReLUConfig(4096,MINI_BATCH_SIZE);
// FCConfig* l33 = new FCConfig(4096, MINI_BATCH_SIZE, 4096);
// ReLUConfig* l34 = new ReLUConfig(4096, MINI_BATCH_SIZE);
// FCConfig* l35 = new FCConfig(4096, MINI_BATCH_SIZE, 1000);
// ReLUConfig* l36 = new ReLUConfig(1000, MINI_BATCH_SIZE);
// config->addLayer(l0);
// config->addLayer(l1);
// config->addLayer(l2);
// config->addLayer(l3);
// config->addLayer(l4);
// config->addLayer(l5);
// config->addLayer(l6);
// config->addLayer(l7);
// config->addLayer(l8);
// config->addLayer(l9);
// config->addLayer(l10);
// config->addLayer(l11);
// config->addLayer(l12);
// config->addLayer(l13);
// config->addLayer(l14);
// config->addLayer(l15);
// config->addLayer(l16);
// config->addLayer(l17);
// config->addLayer(l18);
// config->addLayer(l19);
// config->addLayer(l20);
// config->addLayer(l21);
// config->addLayer(l22);
// config->addLayer(l23);
// config->addLayer(l24);
// config->addLayer(l25);
// config->addLayer(l26);
// config->addLayer(l27);
// config->addLayer(l28);
// config->addLayer(l29);
// config->addLayer(l30);
// config->addLayer(l31);
// config->addLayer(l32);
// config->addLayer(l33);
// config->addLayer(l34);
// config->addLayer(l35);
// config->addLayer(l36);
