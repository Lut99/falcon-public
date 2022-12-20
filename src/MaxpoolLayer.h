
#pragma once
#include "MaxpoolConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;


class MaxpoolLayer : public Layer
{
private:
	MaxpoolConfig conf;
	RSSVectorMyType activations;
	RSSVectorMyType deltas;
	RSSVectorSmallType maxPrime;

public:
	//Constructor and initializer
	MaxpoolLayer(MaxpoolConfig* conf, int _layerNum);

	//Functions
	void printLayer() override;
	void forward(const RSSVectorMyType& inputActivation) override;
	void computeDelta(RSSVectorMyType& prevDelta) override;
	void updateEquations(const RSSVectorMyType& prevActivations) override;

	//Setters
	void setInputRows(size_t rows) override {
		this->conf.batchSize = rows;
		this->activations    = RSSVectorMyType(this->conf.batchSize * this->conf.features * 
											(((this->conf.imageWidth - this->conf.poolSize)/this->conf.stride) + 1) * 
											(((this->conf.imageHeight - this->conf.poolSize)/this->conf.stride) + 1));
		this->deltas         = RSSVectorMyType(this->conf.batchSize * this->conf.features * 
											(((this->conf.imageWidth - this->conf.poolSize)/this->conf.stride) + 1) * 
											(((this->conf.imageHeight - this->conf.poolSize)/this->conf.stride) + 1));
		this->maxPrime       = RSSVectorSmallType((((this->conf.imageWidth - this->conf.poolSize)/this->conf.stride) + 1) * 
													(((this->conf.imageHeight - this->conf.poolSize)/this->conf.stride) + 1) * 
													this->conf.features * this->conf.batchSize * this->conf.poolSize * this->conf.poolSize);
	};

	//Getters
	virtual const char* getName() { return "MaxpoolLayer"; };
	RSSVectorMyType* getActivation() {return &activations;};
	RSSVectorMyType* getDelta() {return &deltas;};
};