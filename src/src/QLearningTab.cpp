#include <stdafx.hpp>

void QLearningTab::init( const int &numActions, ...)
{
    //cout << "ye kyu called ? " <<endl;
    va_list args;
    va_start(args, numActions);

    init(numActions, va_arg(args, double), va_arg(args, double), va_arg(args, int));
}
void QLearningTab::init(const int &numActions, const double & alpha, const double & epsilon, const int& numAgents)
{
    //cout << "ye callied" <<endl;
    this->numActions = numActions;
    this->numAgents = numAgents;

    this->alpha = alpha;
    this->lambda = 1;
    this->gamma = 1;
    this->epsilon = epsilon;

    rho = 0;
    alpha_rho = 1;
    greedy = false;

    //init table of q values
    //initMap();

    // Create the two random number distributions that we will use during action selection
    d = bernoulli_distribution(epsilon);
    explorationDistribution = uniform_int_distribution<int>(0, numActions - 1);
}

vector<int> QLearningTab::getAction(string features, std::mt19937_64 &generator, const bool& training)
{
    vector<int> actions(numAgents);
    // Check if we should explore
    if (d(generator)) // This returns true with probability epsilon, due to the way that d was initialized in the constructor (up top)
    {
        greedy = false;
        for(int i = 0; i < numAgents; i++)
        {
            actions[i] = explorationDistribution(generator);	// Explore by selecting an action uniformly randomly (this is what the explorationDistribution was initialized to do)
        }
        return actions;
    }

    greedy = true;
    set<vector<int>> bestActions;
    double bQ = 0;

    if(training)
    {
        //Hill Climbing

        for(int i = 0; i < numAgents; i++)
        {
            actions[i] = 8;
        }
        //vector<int> e(actions);
        bestActions.insert(actions);

        double qValue = q(features, actions);
        double prev = 0;

        do
        {
            prev = qValue;

            for(int i = 0; i < numAgents; i++)
            {
                for(int a = 0; a < numActions; a++)
                {
                    if(actions[i] != a)
                    {
                        int temp = actions[i];
                        actions[i] = a;
                        double newQ = q(features, actions);

                        if(newQ >= qValue)
                        {
                            if(newQ > qValue )
                            {
                                bestActions.clear();
                                qValue = newQ;
                            }

                            //vector<int> entry(actions);
                            bestActions.insert(actions);
                        }
                        else
                        {
                            actions[i] = temp;
                        }
                    }
                }
            }
        } while( prev != qValue);
        bQ = prev;
    }
    else
    {
        //Do a complete search - currently only works for numAgents <= 2
        double bestQ = 0;
        for(int a1 = 0; a1 < numActions; a1++)
        {
            for(int a2 = 0; a2 < numActions; a2++)
            {
                vector<int> acs = numAgents == 1 ? (vector<int>) {a2} : (vector<int>) {a1, a2};
                double tempQ = q(features, acs);
                if(isnan(tempQ))
                    cout << "features " << features << " acs " << acs[0] << " tempQ " << tempQ << endl;
                if( ((numAgents == 1 || a1 == 0) && a2 == 0) || tempQ > bestQ)
                {
                    bestActions.clear();
                    bestQ = tempQ;
                }

                if(tempQ >= bestQ)
                {
                    bestActions.insert(acs);
                }
            }

            if(numAgents == 1)
            {
                break;
            }
        }

        if(isnan(bestQ))
            cout << "bestQ " << bestQ << endl;
        bQ = bestQ;
    }

    vector<vector<int>> bestActionsV(bestActions.begin(), bestActions.end());

    /*cout << "qValue " << qValue << endl;
    for(int i = 0; i < bestActionsV.size(); i++)
    {
        cout << "[" << bestActionsV[i][0] << "," << bestActionsV[i][1] << "] ";
    }
    cout << endl;*/

    // If there's only one best action, return it
    if ((int)bestActionsV.size() == 1)
        return bestActionsV[0];

    // There are many best actions - select uniformly among them
    uniform_int_distribution<int> greedyIndexDistribution(0, (int)bestActionsV.size()-1);
    int idx = greedyIndexDistribution(generator);

    if(idx < 0 || idx >= bestActionsV.size())
    {
        string s = training ? "training" : "test";
        cout << "idx " << idx <<  " " << bestActions.size() << " " << isnan(bQ)  << " " << s<< endl;
    }
    return bestActionsV[idx];
}

void QLearningTab::train(string features, const vector<int> &action, const double &reward,
                    string newFeatures, const bool & training)
{
    if(!training)
    {
        return;
    }
    // Compute q(s,a), which is q(features, action) using our variables here
    double curQ = q(features, action);
    // Compute max_a', q(s',a'), where s' is described by newFeatures
    double newQ = 0;

    if(training)
    {
        //Hill Climbing

        vector<int> actions(numAgents);
        for(int i = 0; i < numAgents; i++)
        {
            actions[i] = 8;
        }

        double qValue = q(newFeatures, actions);
        double prev = 0;

        do
        {
            prev = qValue;

            for(int i = 0; i < numAgents; i++)
            {
                for(int a = 0; a < numActions; a++)
                {
                    if(actions[i] != a)
                    {
                        int temp = actions[i];
                        actions[i] = a;
                        double tempQ = q(newFeatures, actions);

                        if(tempQ > qValue)
                        {
                            qValue = tempQ;

                        }
                        else if(tempQ < qValue)
                        {
                            actions[i] = temp;
                        }
                    }
                }
            }
        } while( prev != qValue);

        newQ = qValue;
    }
    /*else
    {
        //Do a complete search - currently only works with numAgents = 2
        double bestQ = 0;
        for(int a1 = 0; a1 < numActions; a1++)
        {
            for(int a2 = 0; a2 < numActions; a2++)
            {
                vector<int> acs = {a1, a2};
                double tempQ = q(features, acs);

                bestQ = (a1 == 0 && a2 == 0) ? tempQ : max(tempQ, bestQ);
            }
        }
        newQ = bestQ;
    }*/

    if(greedy)
    {
        double delta = reward + newQ - curQ - rho;
        rho += alpha_rho*delta;
        alpha_rho = alpha_rho/(alpha_rho+10);
    }

    // Compute the TD error using reward, newQ, curQ, and gamma
    double delta = reward + newQ - curQ - rho;


    // Update the q-estimate (w)
    q_update(features, action, delta);
}

double QLearningTab::q(string features, const vector<int> & action)
{
    string key = get_key(features, action);
    return q_get(key);
}

double QLearningTab::q_get(string key)
{
    return q_table.count(key) > 0 ? q_table[key] : 0.0;
}

void QLearningTab::q_update(string features, const vector<int> & action, const double & update)
{
    string key = get_key(features, action);
    q_table[key] = q_get(key) + alpha*update;
}

string QLearningTab::get_key(string features, const vector<int>& action)
{
    string key = features;

    for(int a = 0; a < action.size(); a++)
    {
        key += "_" + to_string(action[a]);
    }
    return key;

}

/*void QLearningTab::initMap()
{
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 6; j++ )
        {
            for(int k = 0; k < 6; k++)
            {
                for(int m = 0; m < 6; m++)
                {
                    for(int l = 0; l < 2; l++)
                    {
                        for(int a = 0; a < 5; a++)
                        {
                            string key = to_string(i) + "_" + to_string(j) + "_" + to_string(k) + "_" + to_string(m) + "_0_" + to_string(l) + "_" + to_string(a);
                            q_table[key] = 0.0;
                        }
                    }

                }
            }
        }
    }
}*/

map<string, double> QLearningTab::getMap()
{
    return q_table;
}

double QLearningTab::get_rho()
{
    return rho;
}

double QLearningTab::get_alpha_rho()
{
    return alpha_rho;
}