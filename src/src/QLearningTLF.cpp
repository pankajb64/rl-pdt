#include <stdafx.hpp>

void QLearningTLF::init( const int &numActions, ...)
{
    va_list args;
    va_start(args, numActions);

    double alpha = va_arg(args, double);
    double epsilon = va_arg(args, double);
    int numAgents = va_arg(args, int);
    int numLocations = va_arg(args, int);
    int numShops = va_arg(args, int);
    int maxLoad = va_arg(args, int);
    int maxInventory = va_arg(args, int);
    vector<int> shoplocations = va_arg(args, vector<int>);

    init(numActions, alpha , epsilon, numAgents, numLocations, numShops, maxLoad, maxInventory, shoplocations);
}

void QLearningTLF::init(const int &numActions, const double & alpha, const double & epsilon, const int& numAgents, int numLocations, int numShops, int maxLoad, int maxInventory, const vector<int>& shopLocations)
{
    //cout << "Initializing TLF Agent" << endl;
    QLearningTab::init(numActions, alpha, epsilon, numAgents);
    // cout << "Base Initialized" << endl;

    this->numLocations = numLocations;
    this->numShops = numShops;
    this->maxLoad = maxLoad;
    this->maxInventory = maxInventory;
    this->shopLocations = shopLocations;

    /*cout << "Received " << numLocations << ", " << numShops << "," << maxLoad << "," << maxInventory << endl;
    cout << this->numLocations << endl;
    cout << "," << this->numShops << endl;
    cout <<"," << this->maxLoad << "," << this->maxInventory << ","  << endl;*/

    int joinActions =  pow((double)numActions, (int)numAgents);
    position_w = VectorXd::Zero(joinActions*numAgents*numShops*numLocations);
    load_w = VectorXd::Zero(joinActions*numAgents*numShops*(maxLoad+1));
    inventory_w = VectorXd::Zero(joinActions*numAgents*numShops*(maxInventory+1));

    //cout << joinActions << "," << numAgents << "," << numShops << "," << numLocations << endl;
    //cout << "Vector sizes " << position_w.size() << "," << inventory_w.size() << "," << load_w.size() << endl;
    //cout << "TLF Agent initialized" << endl;
}

double QLearningTLF::q(string state, const vector<int> &action)
{
    vector<int> tokens = Util::split(state, ";");
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

    double q = 0;

    for(int i = 0; i < numAgents; i++)
    {
        for(auto x : inventories)
        {
            double w = weight(i, x.first, agentLocations[i], x.second, truckLoads[i], action);
            if(isnan(w))
                cout << " agent " << i << " shop " << x.first << " " << agentLocations[i] << " " << x.second << " " << truckLoads[i] << " " << action[0] << " " << w << endl;
            q += w;
        }
    }

    return q;
}

void QLearningTLF::q_update(string state, const vector<int> &action, const double &update)
{
    vector<int> tokens = Util::split(state, ";");
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

    double q = 0;

    for(int i = 0; i < numAgents; i++)
    {
        for(auto x : inventories)
        {
            weight_update(i, x.first, agentLocations[i], x.second, truckLoads[i], action, update);
        }
    }
}

double QLearningTLF::weight(const int &agent, const int &shop, const int &position, const int &inventory,
                            const int &load, const vector<int> &action)
{
    //cout << "Geting TLF weights action[0] " <<action[0] << " agent " << agent << " shop " << shop << " position " << position << " inventory " << inventory <<  " load " << load << endl;
    int shop_pos = (int) distance(shopLocations.begin(), find(shopLocations.begin(), shopLocations.end(), shop));
    double idx_p = 0, idx_i = 0, idx_l = 0;
    for(int i = 0; i < numAgents; i++)
    {
        //cout << "action[" << i << "] " << action[i] << endl;
        idx_p = idx_p * numActions + action[i]*(numAgents*(numShops)*(numLocations));
        idx_i = idx_i * numActions + action[i]*(numAgents*(numShops)*(maxInventory+1));
        idx_l = idx_l * numActions + action[i]*(numAgents*(numShops)*(maxLoad+1));
    }

    //cout << idx_p << "," << idx_i << "," << idx_l << endl;
    idx_p = idx_p + agent*(numShops*numLocations) + shop_pos*numLocations + position;
    idx_i = idx_i + agent*(numShops*(maxInventory+1)) + shop_pos*(maxInventory+1) + inventory;
    idx_l = idx_l + agent*(numShops*(maxLoad+1)) + shop_pos*(maxLoad+1) + load;

    //cout << idx_p << "," << idx_i << "," << idx_l << endl;
    //cout << "Vector sizes " << position_w.size() << "," << inventory_w.size() << "," << load_w.size() << endl;
    if(idx_p < 0 || idx_p >= position_w.size() || isnan(position_w[idx_p]))
    {
        cout << "idx_p " << idx_p << endl;
    }

    if(idx_l < 0 || idx_l >= load_w.size() || isnan(load_w[idx_l]))
    {
        cout << "idx_l " << idx_l << endl;
    }

    if(idx_i < 0 || idx_i >= inventory_w.size() || isnan(inventory_w[idx_i]))
    {
        cout << "idx_i " << idx_i << endl;
    }

    return position_w[idx_p] + inventory_w[idx_i] + load_w[idx_l];
}

void QLearningTLF::weight_update(const int &agent, const int &shop, const int &position, const int &inventory,
                                 const int &load, const vector<int> &action, double update)
{
    //cout << "Updating TLF weights " << endl;
    int shop_pos = (int) distance(shopLocations.begin(), find(shopLocations.begin(), shopLocations.end(), shop));
    double idx_p = 0, idx_i = 0, idx_l = 0;
    for(int i = 0; i < numAgents; i++)
    {
        idx_p = idx_p * numActions*(numAgents*(numShops)*(numLocations)) + action[i];
        idx_i = idx_i * numActions*(numAgents*(numShops)*(maxInventory+1)) + action[i];
        idx_l = idx_l * numActions*(numAgents*(numShops)*(maxLoad+1)) + action[i];
    }

    idx_p = idx_p + agent*(numShops*numLocations) + shop_pos*numLocations + position;
    idx_i = idx_i + agent*(numShops*(maxInventory+1)) + shop_pos*(maxInventory+1) + inventory;
    idx_l = idx_l + agent*(numShops*(maxLoad+1)) + shop_pos*(maxLoad+1) + load;

    if(idx_p < 0 || idx_p >= position_w.size())
    {
        cout << "idx_p " << idx_p << endl;
    }

    if(idx_l < 0 || idx_l >= load_w.size())
    {
        cout << "idx_l " << idx_l << endl;
    }

    if(idx_i < 0 || idx_i >= inventory_w.size())
    {
        cout << "idx_i " << idx_i << endl;
    }

    //cout << idx_p << "," << idx_i << "," << idx_l << endl;
    double res = position_w[idx_p] + alpha*update;
    if(isnan(position_w[idx_p]) || isnan(res))
        cout << "idx_p " << idx_p << " position_w " << position_w[idx_p] << "alpha " << alpha << " update " << update << " alpha*update" << alpha*update << " res " << res << endl;
    position_w[idx_p] += alpha*update;

    res = inventory_w[idx_i] + alpha*update;
    if(isnan(inventory_w[idx_i]) || isnan(res))
        cout << "idx_i " << idx_i << " inventory_w " << inventory_w[idx_i] <<  alpha << " update " << update << " alpha*update" << alpha*update << " res " << res << endl;
    inventory_w[idx_i] += alpha*update;

    res = load_w[idx_l] + alpha*update;
    if(isnan(load_w[idx_l]) || isnan(res))
        cout << "idx_l " << idx_l << " load_w " << load_w[idx_l] <<  alpha << " update " << update << " alpha*update" << alpha*update << " res " << res << endl;
    load_w[idx_l] += alpha*update;
}