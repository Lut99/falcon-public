
#pragma once
#include <stdexcept>
#include <string>
#include "globals.h"

class Layer
{
public: 
	int layerNum = 0;
	Layer(int _layerNum): layerNum(_layerNum) {};
	virtual ~Layer() {};

//Virtual functions	
	virtual void printLayer() { throw std::runtime_error(std::string("No implementation of 'Layer::printLayer()' provided for '") + this->getName() + "'"); };
	virtual void forward(const RSSVectorMyType& inputActivation) { throw std::runtime_error(std::string("No implementation of 'Layer::forward()' provided for '") + this->getName() + "'"); };
	virtual void computeDelta(RSSVectorMyType& prevDelta) { throw std::runtime_error(std::string("No implementation of 'Layer::computeDelta()' provided for '") + this->getName() + "'"); };
	virtual void updateEquations(const RSSVectorMyType& prevActivations) { throw std::runtime_error(std::string("No implementation of 'Layer::updateEquations()' provided for '") + this->getName() + "'"); };

//Setters
	virtual void setInputRows(size_t rows) { throw std::runtime_error(std::string("No implementation of 'Layer::setInputRows()' provided for '") + this->getName() + "'"); };

//Getters
	virtual const char* getName() { return "<Layer>"; };
	virtual RSSVectorMyType* getActivation() { throw std::runtime_error(std::string("No implementation of 'Layer::getActivation()' provided for '") + this->getName() + "'"); };
	virtual RSSVectorMyType* getDelta() { throw std::runtime_error(std::string("No implementation of 'Layer::getDelta()' provided for '") + this->getName() + "'"); };
};