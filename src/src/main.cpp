#include <stdafx.hpp>
#include <sys/stat.h>
#include <unistd.h>
#include <cublas_v2.h>
#include <iterator>
#include <chrono>

using namespace std;
using namespace Eigen;
using namespace std::chrono;

// Compute the sample variance of the provided vector
double var(const Eigen::VectorXd & v)
{
	double mu = v.mean();
	double result = 0;
	for (int i = 0; i < (int)v.size(); i++)
		result += (v[i] - mu)*(v[i] - mu);
	return result / (v.size() - 1.0);
}

// Run one agent on one environment for some number of episodes, and return the
// resulting discounted sums of rewards
//template <class T>
VectorXd runAgentEnvironment(shared_ptr<Agent> a, shared_ptr<Env> e, int timeSteps, mt19937_64 & generator, ofstream& out, int trainingStepsPerPhase, int testStepsPerPhase, int agentType, bool useAfterState)
{
    //IOFormat CleanFmt(2, 0, ", ", " ", "[", "]");
	VectorXd state, newFeatures, afterState, result(timeSteps);
    string stateString, newStateString, afterStateString;
	double reward;
	vector<int> action(1);
    int phaseStep = 0;
    bool mode = true; //true for training, false for testing
    //state = e->getState();
    stateString = e->getStateString();
    for (int t = 0; t < timeSteps; t++)
    {
        action = a->getAction(stateString, generator, mode);
        reward = e->update(action);

        //VectorXd newFeatures;
        //newFeatures = e->getState();
        newStateString = e->getStateString();
        //afterState = e->getAfterState();
        afterStateString = e->getAfterStateString();
        if((agentType == 1 || agentType == 3) && useAfterState)
        {
            newStateString = afterStateString;
        }

        result[t] = reward;

        a->train(stateString, action, reward, newStateString, mode);

        //state = afterState;
        stateString = newStateString;

        if(mode)
        {
            if(phaseStep < trainingStepsPerPhase)
            {
                phaseStep++;
            }
            else
            {
                phaseStep = 0;
                mode = false; // start evaluation
            }
        }
        else
        {
            if(phaseStep < testStepsPerPhase)
            {
                phaseStep++;
            }
            else
            {
                phaseStep = 0;
                mode = true; // start training again!
            }
        }
    }
    //cout << "done ? " << endl;
	return result;
}

// Entry point - runs the menu system
int main(int argc, char * argv[])
{
    //start the timer to measure execution time
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    int numTrials = 30;
    string agentName, envName = "pd";
    bool afterState = true;
    //1 for Learning Agent (QLearning), 2 for Non-Learning Greedy Agent (agent 3 not working)
    int agentType = 2;
    if(agentType == 1)
    {
        agentName = afterState ? "QLearningTabASH" : "QLearningTab";
    }
    else if(agentType == 3)
    {
        agentName = afterState ? "QLearningTLFASH" : "QLearningTLF";
    }
    else
    {
        agentName = "NLGA";
        numTrials = 1; // non-learning agent ain't need no multiple trials
    }
    int timeSteps = 1000000; //1000000;

    vector<VectorXd> results(numTrials);
    vector<std::mt19937_64> generators(numTrials);
#pragma omp parallel for
    for(int t = 0; t < numTrials; t++)
    {
        int numPhases = 20;
        double trainPercent = 0.96;

        int stepsPerPhase = timeSteps/numPhases;
        int trainingStepsPerPhase = (int) (stepsPerPhase*trainPercent);
        int testStepsPerPhase = stepsPerPhase - trainingStepsPerPhase;

        /*int numLocations = 10;
        int numShops = 5;
        int depotLocation = 4;
        int numTrucks = 2;

        vector<int> shoplocations = {0, 2, 6, 8, 9};
        vector<vector<int>> locationMap = {{0, 1, 3, 0},
                                           {0, 2, 4, 1},
                                           {1, 2, 5, 2},
                                           {3, 4, 6, 0},
                                           {3, 5, 7, 1},
                                           {4, 5, 8, 2},
                                           {6, 7, 6, 3},
                                           {6, 8, 9, 4},
                                           {7, 8, 9, 5},
                                           {7, 8, 9, 9}};*/

        int numLocations = 4;
        int numShops = 3;
        int depotLocation = 1;
        int numTrucks = 1;

        vector<int> shoplocations = {0, 2, 3};
        vector<vector<int>> locationMap = {{0, 1, 2, 0},
                                           {0, 1, 3, 1},
                                           {2, 3, 2, 0},
                                           {2, 3, 3, 1}};

        double alpha = 0.1;
        //double lambda = 0.99;
        //double gamma = 0.99;
        double epsilon = 0.1;

        shared_ptr<Env> e = make_shared<Env>(numLocations, shoplocations, locationMap, depotLocation, numTrucks, 20);
        int numActions = e->getNumActions();
        int numFeaturesPerAgent = e->getStateDim();
        int maxLoad = e->getMaxLoad();
        int maxInventory = e->getMaxInventory();

        shared_ptr<Agent> agent;

        //std::mt19937_64 generator;
        generators[t].seed(t);

        if(agentType == 1)
        {
            agent = make_shared<QLearningTab>();
            agent->init(numActions, alpha, epsilon, numTrucks);
        }
        else if(agentType == 3)
        {
            agent = make_shared<QLearningTLF>();
            //cout << "Passed " << numActions << "," << numLocations << ", " << numShops << "," << maxLoad << "," << maxInventory << endl;
            agent->init(numActions, alpha, epsilon, numTrucks, numLocations, numShops, maxLoad, maxInventory, shoplocations);
        }
        else
        {
            agent = make_shared<NLGA>();
            agent->init(numActions, numLocations, shoplocations, locationMap, depotLocation, numTrucks, maxInventory, shoplocations );
            //NLGA* a = (NLGA*) agent;
            //cout << a->getAction("0;4;3;3;3", generator, false)[0] << endl;
        }

        // Print results to a file
        string fileName = "out_" + agentName + "_" + envName +"_tr" + to_string(t) + ".csv";
        ofstream out(fileName.c_str());
        //out << "Timestep Reward,State,Action,Next State,Old Q,New Q,rho,alpha_rho" << endl;
        results[t] = runAgentEnvironment(agent, e, timeSteps , generators[t], out, trainingStepsPerPhase, testStepsPerPhase, agentType, afterState);

        out.close();
    }

    cout << "h" << endl;
    string fileName = "out_" + agentName + "_" + envName + ".csv";
    ofstream out(fileName.c_str());
    out << "Mean,Standard Deviation" << endl;

    for (int t = 0; t < timeSteps; t++)
    {
        VectorXd v(numTrials);
        for(int i = 0; i < numTrials; i++)
        {
            v[i] = results[i][t];
        }

        // Print the mean and sample standard deviation
        out << v.mean() << "," << sqrt(var(v)) << endl;
    }
    // Close and print a line saying that we're done with this experiment.
    out.close();
    cout << "Results printed to file. " << fileName << endl << endl;

    //end the timer
    high_resolution_clock ::time_point t2 = high_resolution_clock::now();

    //measure the duration. Ref https://stackoverflow.com/a/22387757/1026535
    auto duration = duration_cast<microseconds>(t2 - t1).count();
    cout << "Execution finished in " << duration << " microseconds" << endl;
}