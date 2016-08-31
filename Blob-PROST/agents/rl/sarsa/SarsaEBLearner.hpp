/****************************************************************************************
 ** Implementation of Sarsa(lambda) with Exploration Bonus. It implements Fig. 8.8 (Linear, gradient-descent
 ** Sarsa(lambda)) from the book "R. Sutton and A. Barto; Reinforcement Learning: An
 ** Introduction. 1st edition. 1988."
 ** Some updates are made to make it more efficient, as not iterating over all features.
 **
 **
 ** Author:
 ***************************************************************************************/

#ifndef SARSAEBLEARNER_H
#define SARSAEBLEARNER_H
#include "SarsaLearner.hpp"
#endif
#include <vector>
#include <unordered_map>

using namespace std;

class SarsaEBLearner : public SarsaLearner{
private:

    double beta,sigma;

    /**
     * Constructor declared as private to force the user to instantiate SarsaLearner
     * informing the parameters to learning/execution.
     */
    SarsaEBLearner();

public:

    /**
    *   Initialize everything required for SarsaLearner.
    *   Additional params for EB:
    *   - beta : Exploration rate.
    *   - sigma: Generalization factor.
    */
    SarsaEBLearner(ALEInterface& ale, Features *features, Parameters *param,int seed);

    void learnPolicy(ALEInterface& ale, Features *features);

    ~SarsaEBLearner();
};
