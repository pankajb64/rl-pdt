//
// Created by pankaj on 12/12/17.
//

#include <stdafx.hpp>

using namespace std;
using namespace Eigen;

Env::Env(const int & numLocations, const vector<int> & shopLocations, const vector<vector<int>> & locationMap, const int & depotLocation,
         const int & numTrucks, int T)
{
    this->numLocations = numLocations;
    this->locationMap = locationMap;
    this->depotLocation = depotLocation;
    this->numTrucks = numTrucks;

    this->T = T;
    counter = 0;
    afterState = VectorXd::Zero(getStateDim());

    initTruckLocationsAndLoads();
    initInventory(shopLocations);
}

Env::~Env() {}

void Env::initInventory(const vector<int> & shopLocations)
{
    for(int s = 0; s < shopLocations.size(); s++)
    {
        inventories[shopLocations[s]] = maxInventory;
    }
}

void Env::resetInventory()
{
    for(auto& x : inventories)
    {
        x.second = maxInventory;
    }
}

void Env::initTruckLocationsAndLoads()
{
    truckLocations.resize(numTrucks);
    truckLoads.resize(numTrucks);

    resetTruckLocationsAndLoads();
}

void Env::resetTruckLocationsAndLoads()
{
    for(int i = 0; i < numTrucks; i++)
    {
        truckLocations[i] = depotLocation;
        truckLoads[i] = maxLoad;
    }
}

VectorXd Env::getState() {

    //VectorXd state = VectorXd::Zero(numLocations * numTrucks + numLocations + numTrucks);
    VectorXd state = VectorXd::Zero(numLocations * numTrucks + inventories.size()*(maxInventory+1) + numTrucks);

    for (int i = 0; i < numTrucks; i++) {
        state[(numLocations * i) + truckLocations[i]] = 1;
    }

    int count = 0;
    for (auto &x : inventories) {
        if(x.second >= 0)
        {
            state[numLocations * numTrucks + count*(maxInventory+1) + x.second] = 1;
        }
        count++;
    }

    for (int i = 0; i < numTrucks; i++) {
        state[numLocations * numTrucks +  inventories.size()*(maxInventory+1) + i] = truckLoads[i]/maxLoad;
    }

    /*IOFormat CleanFmt(4, 0, ", ", " ", "[", "]");
    cout << "getState " << state.format(CleanFmt) << endl;*/
    return state;
}

string Env::getStateString()
{
    string state = "";
    for(int i = 0; i < numTrucks; i++)
    {
        state += to_string(truckLocations[i]) + ";";
    }

    for(auto& x : inventories)
    {
        state += to_string(x.second) + ";";
    }

    for(int i = 0; i < numTrucks; i++)
    {
        state += to_string(truckLoads[i]) + ";";
    }
    state.pop_back();

    return state;
}

double Env::update(const vector<int> & actions)
{
    double reward = 0;

    for(int i = 0; i < actions.size(); i++)
    {
        int action = actions[i];
        int location = truckLocations[i];

        if(action < 8)
        {
            reward -= 0.1;
        }

        if(action > 3 and action < 8) //unloads
        {
            if(truckLoads[i] > 0 && inventories.count(location) > 0)
            {
                int unload = action - 3;
                unload = min(truckLoads[i], unload);
                unload = min(maxInventory - inventories[location], unload);
                truckLoads[i] = truckLoads[i] - unload ;
                inventories[location] = min(maxInventory, inventories[location] + unload);
                //reward += 1;
            }
        }
        else if(action <= 3)
        {
            int nextLocation = getNextLocation(i, action);
            truckLocations[i] = nextLocation;
            if(nextLocation == depotLocation && truckLoads[i] == 0)
            {
                truckLoads[i] = maxLoad;
                //reward += 5;
            }
        }
    }

    afterState = getState();
    afterStateString = getStateString();

    counter++;

    for(auto& x : inventories )
    {
        if(counter > 0 && counter%(T*(x.first+1)) == 0 && x.second > 0)
        {
            x.second -= min(x.second, 1);
            //cout << "inventory for " << x.first << " is now " << x.second << endl;
        }
    }

    for(auto& x : inventories)
    {
        if(x.second == 0)
        {
            reward -= 5;
        }
    }

    return reward;
}

int Env::getNextLocation(const int & truck, const int & action)
{
    return locationMap[truckLocations[truck]][action];
}

bool Env::inTerminalState()
{
    return false; //counter >= 1000000;
}

int Env::getNumActions()
{
    return 9;
}

void Env::reset()
{
    counter = 0;

    resetTruckLocationsAndLoads();
    resetInventory();
}

int Env::getStateDim()
{
    return numLocations * numTrucks + inventories.size()*(maxInventory+1) + numTrucks;
}

VectorXd Env::getAfterState()
{
    return afterState;
}

string Env::getAfterStateString()
{
    return afterStateString;
}

int Env::getMaxInventory()
{
    return maxInventory;
}

int Env::getMaxLoad()
{
    return maxLoad;
}