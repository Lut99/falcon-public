
#pragma once
#include "FCConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;

extern int partyNum;


class FCLayer : public Layer
{
private:
	FCConfig conf;
	RSSVectorMyType activations;
	RSSVectorMyType deltas;
	RSSVectorMyType weights;
	RSSVectorMyType biases;


public:
	//Constructor and initializer
	FCLayer(FCConfig* conf, int _layerNum);
	void initialize();

	//Functions
	void printLayer() override;
	void forward(const RSSVectorMyType& inputActivation) override;
	void computeDelta(RSSVectorMyType& prevDelta) override;
	void updateEquations(const RSSVectorMyType& prevActivations) override;

	//Setters
	void setInputRows(size_t rows) override {
		this->conf.batchSize = rows;
		this->activations    = RSSVectorMyType(this->conf.batchSize * this->conf.outputDim);
		this->deltas         = RSSVectorMyType(this->conf.batchSize * this->conf.outputDim);
	};

	//Getters
	virtual const char* getName() { return "FCLayer"; };
	RSSVectorMyType* getActivation() {return &activations;};
	RSSVectorMyType* getDelta() {return &deltas;};
	RSSVectorMyType* getWeights() {return &weights;};
	RSSVectorMyType* getBias() {return &biases;};
};