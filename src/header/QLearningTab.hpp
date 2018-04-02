//
// Created by pankaj on 12/13/17.
//

#pragma once

#include <stdafx.hpp>

using namespace std;
using namespace Eigen;

//QLearning using Tabular Representation - based on HW4 code
class QLearningTab : public Agent
{
public:
    QLearningTab () {}
    ~QLearningTab () {}

    virtual void init(const int &numActions, ...);
    void init(const int &numActions, const double & alpha, const double & epsilon, const int& numAgents);
    virtual vector<int> getAction(string features, std::mt19937_64 &generator, const bool& training);
    virtual void train(string features, const vector<int> & action, const double & reward, string newFeatures, const bool & training);
    virtual double q(string state, const vector<int> & action);
    virtual void q_update(string state, const vector<int> & action, const double & update);
    string get_key(string state, const vector<int> & action);
    map<string, double> getMap();
    double q_get(string key);
    double get_rho();
    double get_alpha_rho();

    int numAgents;
    int numActions;
    double alpha;
    double lambda;
    double gamma;
    double epsilon;

private:
    int numFeatures;

    double rho;
    double alpha_rho;
    bool greedy;

    map<string, double> q_table; //table of q values

    std::bernoulli_distribution d; // The distribution to decide whether or not to explore
    std::uniform_int_distribution<int> explorationDistribution; // If exploring, this is the uniform distribution over actions

};