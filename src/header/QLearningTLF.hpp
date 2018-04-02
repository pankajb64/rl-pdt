//
// Created by pankaj on 12/20/17.
//

#pragma once

#include <stdafx.hpp>

class QLearningTLF : public  QLearningTab
{
public:
    QLearningTLF() {}
    ~QLearningTLF() {}

    virtual void init(const int &numActions, ...);
    void init(const int &numActions, const double & alpha, const double & epsilon, const int& numAgents, int numLocations, int numShops, int maxLoad, int maxInventory, const vector<int>& shopLocations);
    virtual double q(string state, const vector<int> & action);
    virtual void q_update(string state, const vector<int> & action, const double & update);
    double weight(const int& agent, const int& shop, const int &position, const int& inventory, const int& load, const vector<int> & action);
    void weight_update(const int& agent, const int& shop, const int &position, const int& inventory, const int& load, const vector<int> & action, double update);


private:
    VectorXd position_w;
    VectorXd load_w;
    VectorXd inventory_w;

    int numLocations;
    int numShops;
    int maxLoad;
    int maxInventory;

    vector<int> shopLocations;
};