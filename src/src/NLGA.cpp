//
// Created by pankaj on 12/17/17.
//
#include <stdafx.hpp>

void NLGA::initDistances()
{
    distances.resize(numLocations);
    for(int i = 0; i < numLocations; i++)
    {
        vector<int> dist(numLocations);

        for(int j = 0; j < numLocations; j++)
        {
            dist[j] = -1;
        }
        distances[i] = dist;
    }
    for(int i = 0; i < numLocations; i++)
    {
        for(int j = 0; j < numLocations; j++)
        {
            //cout << i << "," << j << endl;
            distances[i][j] = distances[i][j] == -1 ? distance(i, j) : distances[i][j];
            //cout <<  i << "," << j << ", Distance " << distances[i][j] << endl;
        }
    }
    //cout << "distances initialized" << endl;
}

int NLGA::distance(int i, int j)
{
    //cout << "Calling distance on " << i << " " << j << endl;
    if(j < i)
    {
        return distance(j, i);
    }

    if(j == i)
    {
        return 0;
    }

    if(distances[i][j] >= 0 )
    {
        //cout << "Returning " << distances[i][j] << endl;
        return distances[i][j];
    }

    set<int> nextLocations(locationMap[i].begin(), locationMap[i].end());
    int minDistance = -1;
    for(auto x : nextLocations)
    {
        //cout << i << " neighbour " << x << " " << j << endl;
        if(x == j)
        {
            //cout << "Found " << j << " returning 1" << endl;
            return 1;
        }
    }

    for(auto x : nextLocations)
    {
        if(x <= i)
        {
            //cout << "Skipping x =" << x << " for i = " << i <<endl;
            continue;
        }

        int dist = distance(x, j);
        minDistance = minDistance < 0 ? dist + 1 : min(minDistance, dist + 1);
        if(minDistance == 2) //not gonna get shorter than this.
        {
            //cout << "Found minDistance " << minDistance << " breaking loop" << endl;
            break;
        }
    }

    return minDistance;
}

vector<int> NLGA::getAction(string features, std::mt19937_64 &generator, const bool &training)
{
    //cout << features << endl;
    vector<int> tokens = Util::split(features, ";");
    vector<int> agentLocations(numAgents);

    for(int i = 0; i < numAgents; i++)
    {
        agentLocations[i] = tokens[i];
    }

    map<int, int> inventories;
    for(int i = 0; i < shopLocations.size(); i++)
    {
        inventories[shopLocations[i]] = tokens[numAgents + i];
    }

    vector<int> truckLoads(numAgents);
    for(int i = 0; i < numAgents; i++)
    {
        truckLoads[i] = tokens[numAgents + inventories.size() + i];
    }

    set<int> candidates;
    for(auto x : inventories)
    {
        if(x.second < maxInventory)
        {
            //cout << "Inserting candidate " << x.first << " with inventory " << x.second << endl;
            candidates.insert(x.first);
        }
    }

    vector<int> actions(numAgents);
    for(int i = 0; i < numAgents; i++)
    {
        int shop = greedy_select(i, agentLocations, inventories, truckLoads, candidates );
        //cout << "shop " << shop << endl;
        candidates.erase(shop);
        actions[i] = get_action(i, shop, agentLocations, truckLoads);
        //cout << i << " " << actions[i] << endl;
    }

    return  actions;
}


int NLGA::greedy_select(int agent, const vector<int> &agentLocations, map<int, int> &inventories,
                        const vector<int> &truckLoads, const set<int>& candidates)
{
    if(truckLoads[agent] == 0 || candidates.size() == 0)
    {
        return -1;
    }

    if(candidates.count(agentLocations[agent]) > 0) //agent already on a candidate location
    {
        //cout << "Already on " << agentLocations[agent] << endl;
        return agentLocations[agent];
    }

    //cout << "p" << endl;
    //lambda to compare entries in priority queue
    auto compare = [agentLocations, agent, inventories, this](int left, int right) {
        int agentLoc = agentLocations[agent];
        int dist1 = distances[agentLoc][left];
        int mDist1 = minDistance(left, agent);
        int dist2 = distances[agentLoc][right];
        int mDist2 = minDistance(right, agent);

        int diff1 = dist1 - mDist1;
        int diff2 = dist2 - mDist2;

        /*cout << dist1 << "," << mDist1 << "," << diff1 << "," << dist2 << "," << mDist2 << "," << diff2 << endl;
        cout << "inventories - { ";
        for(auto x : inventories)
        {
            cout << x.first << "=" << x.second << " ";
        }
        //cout << "}" << endl;
        //cout << left << "," << right << endl;*/
        return diff1 == diff2 ? inventories.at(left) > inventories.at(right) : diff1 > diff2;
    };

    priority_queue<int, vector<int>, decltype(compare)> q(compare);

    for(auto x :  candidates)
    {
        q.push(x);
    }
    //cout << "q" << q.size() <<  endl;

    int top = q.top();
    //cout << top << endl;
    int dist = distances[agentLocations[agent]][top];
    int mDist = minDistance(top, agent);

    if(dist - mDist < 0 && agent < numAgents - 1) //its closer to some other agent, so wait
    {
        return -1;
    }

    return top;
}

int NLGA::minDistance(int location, int agent)
{
    int minD = -1;
    for(int i = 0; i < numAgents; i++)
    {
        if(i != agent)
        {
            int d = distances[location][i];
            minD = minD < 0 ? d : min(minD, d);
        }
    }

    return minD;
}

int NLGA::get_action(int agent, int location, const vector<int> &agentLocations, const vector<int>& truckLoads)
{
    int action = -1;
    if(location == -1)
    {
        if(truckLoads[agent] == 0)
        {
            //go to depot
            action = get_action_direction(agentLocations[agent], depotLocation);
        }
        else
        {
            //wait
            action = 8;
        }
    }
    else
    {
        if(agentLocations[agent] == location)
        {
            //we are at shop, unload some goods!
            //always try to unload max, env automatically takes whatever possible/needed
            action = 7;
        }
        else
        {
            //go to location;
            action  = get_action_direction(agentLocations[agent], location);
        }
    }

    return action;
}

int NLGA::get_action_direction(int currentLoc, int destLoc)
{
    //cout << "current Loc " << currentLoc << " destLoc " << destLoc << endl;
    vector<int> nextLocations = locationMap[currentLoc];
    int minDistance = -1, next_action = -1;

    if(currentLoc == destLoc)
    {
        //wait there!
        next_action = 8;
    }

    for(int i = 0; i < nextLocations.size(); i++)
    {
        int neighbour = nextLocations[i];
        //cout << "Trying neighbour " << neighbour << " currentLoc " << currentLoc << endl;
        if(nextLocations[i] == destLoc)
        {
            next_action = i;
            minDistance = 1;
            break;
        }

        if(neighbour == currentLoc)
        {
            //cout << "Skipping " << neighbour << endl;
            continue;
        }

        if(minDistance < 0 || distances[nextLocations[i]][destLoc] < minDistance)
        {
            minDistance = distances[nextLocations[i]][destLoc];
            next_action = i;
            //cout << minDistance << " " << next_action << endl;
        }
    }

    return  next_action;
}

void NLGA::train(string features, const vector<int> &action, const double &reward, string newFeatures,
                 const bool &training)
{
    //No training, do nothing
    return;
}

void NLGA::init1(const int &numActions, const int &numLocations, const vector<int> &shopLocations, const vector<vector<int>> &locationMap,
                const int &depotLocation, const int &numAgents, const int &maxInventory)
{
    //cout << "initializing" << endl;
    this->numLocations = numLocations;
    this->shopLocations = shopLocations;
    this->locationMap = locationMap;
    this->depotLocation = depotLocation;
    this->numAgents = numAgents;
    this->maxInventory = maxInventory;
    this->numActions = numActions;

    //cout << "g" << endl;
    initDistances();
}

void NLGA::init(const int &numActions, ...)
{
    va_list args;
    va_start(args, numActions);
    int numLocations = va_arg(args, int);
    //cout << "NLGA Agent" << endl;
    vector<int> shopLocations = va_arg(args, vector<int>);
    //cout << "shopLocations" << endl;
    vector<vector<int>> locationMap = va_arg(args, vector<vector<int>>);
    //cout << "locationMap" << endl;
    int depotLocation = va_arg(args, int);
    //cout << depotLocation << endl;
    int numAgents = va_arg(args, int);
    //cout << numAgents << endl;
    int maxInventory = va_arg(args, int);
    //cout << maxInventory << endl;
    init1(numActions, numLocations, shopLocations, locationMap, depotLocation, numAgents, maxInventory);
    //cout << "?l" << endl;
}