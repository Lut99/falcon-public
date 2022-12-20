
#pragma once
#include "CNNConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;


class CNNLayer : public Layer
{
private:
	CNNConfig conf;
	RSSVectorMyType activations;
	RSSVectorMyType deltas;
	RSSVectorMyType weights;
	RSSVectorMyType biases;

public:
	//Constructor and initializer
	CNNLayer(CNNConfig* conf, int _layerNum);
	void initialize();

	//Functions
	void printLayer() override;
	void forward(const RSSVectorMyType& inputActivation) override;
	void computeDelta(RSSVectorMyType& prevDelta) override;
	void updateEquations(const RSSVectorMyType& prevActivations) override;

	//Setters
	void setInputRows(size_t rows) override {
		this->conf.batchSize = rows;
		this->activations    = RSSVectorMyType(this->conf.batchSize * this->conf.filters * 
											(((this->conf.imageWidth - this->conf.filterSize + 2*this->conf.padding)/this->conf.stride) + 1) * 
											(((this->conf.imageHeight - this->conf.filterSize + 2*this->conf.padding)/this->conf.stride) + 1));
		this->deltas         = RSSVectorMyType(this->conf.batchSize * this->conf.filters * 
											(((this->conf.imageWidth - this->conf.filterSize + 2*this->conf.padding)/this->conf.stride) + 1) * 
											(((this->conf.imageHeight - this->conf.filterSize + 2*this->conf.padding)/this->conf.stride) + 1));
	};

	//Getters
	virtual const char* getName() { return "CNNLayer"; };
	RSSVectorMyType* getActivation() {return &activations;};
	RSSVectorMyType* getDelta() {return &deltas;};
	RSSVectorMyType* getWeights() {return &weights;};
	RSSVectorMyType* getBias() {return &biases;};
};