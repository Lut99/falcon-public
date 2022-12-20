
#pragma once
#include "BNConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;


class BNLayer : public Layer
{
private:
	BNConfig conf;
	RSSVectorMyType activations;
	RSSVectorMyType deltas;
	RSSVectorMyType gamma;
	RSSVectorMyType beta;
	RSSVectorMyType xhat;
	RSSVectorMyType sigma;

public:
	//Constructor and initializer
	BNLayer(BNConfig* conf, int _layerNum);
	void initialize();

	//Functions
	void printLayer() override;
	void forward(const RSSVectorMyType& inputActivation) override;
	void computeDelta(RSSVectorMyType& prevDelta) override;
	void updateEquations(const RSSVectorMyType& prevActivations) override;

	//Setters
	void setInputRows(size_t rows) override {
		this->conf.numBatches = rows;
		this->activations     = RSSVectorMyType(this->conf.numBatches * this->conf.inputSize);
		this->deltas          = RSSVectorMyType(this->conf.numBatches * this->conf.inputSize);
	};

	//Getters
	virtual const char* getName() { return "BCLayer"; };
	RSSVectorMyType* getActivation() {return &activations;};
	RSSVectorMyType* getDelta() {return &deltas;};
};