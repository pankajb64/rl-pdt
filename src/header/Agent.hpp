#pragma once

#include <stdafx.hpp>

// Pure abstract base class for all agents
class Agent
{
public:
	virtual ~Agent() {};
    virtual void init(const int &numActions, ...) = 0;
	virtual vector<int> getAction(string features, std::mt19937_64 & generator, const bool& training) = 0;
	virtual void train(string features, const vector<int> & action, const double & reward, string newFeatures, const bool & training) = 0;
};