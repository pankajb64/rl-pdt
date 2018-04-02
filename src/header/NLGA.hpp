//
// Created by pankaj on 12/17/17.
//

#pragma once

#include <stdafx.hpp>

using namespace std;
using namespace Eigen;

//Non-Learning Greedy Agent
class NLGA : public Agent
{
public:
    NLGA () {}
    virtual ~NLGA () {}

    void init(const int &numActions, ...);
    void init1(const int& numActions, const int & numLocations, const vector<int> & shopLocations, const vector<vector<int>> & locationMap, const int  & depotLocation, const int & numAgents, const int& maxInventory);
    vector<int> getAction(string features, std::mt19937_64 &generator, const bool& training);
    void train(string features, const vector<int> & action, const double & reward, string newFeatures, const bool & training);
    int get_action_direction(int currentLoc, int destLoc);

private:
    int numLocations;
    int numActions;
    int numAgents;
    int maxInventory;
    int depotLocation;

    vector<int> shopLocations;
    vector<vector<int>> locationMap;
    vector<vector<int>> distances;

    void initDistances();
    int distance(int i, int j);
    vector<int> split(const string& str, const string& delim);
    int greedy_select(int agent, const vector<int>& agentLocations, map<int, int>& inventories, const vector<int>& truckLoads, const set<int>& candidates );
    int get_action(int agent, int location, const vector<int>& agentLocations, const vector<int>& truckLoads);
    int minDistance(int location, int agent);
};