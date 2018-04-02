//
// Created by pankaj on 12/12/17.
//

//
// Created by pankaj on 12/12/17.
//
#pragma  once

#include <stdafx.hpp>

using namespace std;
using namespace Eigen;

//Product Delivery Domain Environment
class Env
{
public:
    Env(const int & numLocations, const vector<int> & shopLocations, const vector<vector<int>> & locationMap, const int  & depotLocation, const int & numTrucks, int T );
    ~Env();

    VectorXd getState();
    double update(const vector<int> & actions);
    bool inTerminalState();
    int getNumActions();
    void reset();
    int getStateDim();
    string getStateString();
    VectorXd getAfterState();
    string getAfterStateString();
    int getMaxInventory();
    int getMaxLoad();


private:
    vector<int> truckLocations;
    vector<int> truckLoads;
    vector<vector<int>> locationMap;
    map<int,int> inventories;

    int depotLocation;
    int numLocations;
    int numTrucks;
    int counter;
    int maxInventory = 4;
    int maxLoad = 4;
    int T = 20;

    VectorXd afterState;
    string afterStateString;

    void initInventory(const vector<int> & shopLocations);
    void resetInventory();
    void initTruckLocationsAndLoads();
    void resetTruckLocationsAndLoads();
    int getNextLocation(const int & truck, const int & action);

};