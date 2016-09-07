/*******************************************************************************
 ** Implementation of Sarsa(lambda). It implements Fig. 8.8 (Linear,
 ** gradient-descent Sarsa(lambda)) from the book "R. Sutton and A. Barto;
 ** Reinforcement Learning: An Introduction. 1st edition. 1988." Some updates
 ** are made to make it more efficient, as not iterating over all features.
 **
 ** TODO: Make it as efficient as possible.
 **
 ** Author: Marlos C. Machado
 ******************************************************************************/

#ifndef TIMER_H
#define TIMER_H
#include "../../../common/Timer.hpp"
#include "../../../common/Mathematics.hpp"
#endif
#include "SarsaEBLearner.hpp"
#include <stdio.h>
#include <math.h>
#include <set>

#include <algorithm>

using namespace std;

SarsaEBLearner::SarsaEBLearner(ALEInterface& ale,
                               Features* features,
                               Parameters* param,
                               int seed)
    : SarsaLearner(ale, features, param, seed) {
  printf("SarsaEBLearner is Running the show!!!\n");
  beta = param->getBeta();
  sigma = param->getSigma();

  actionProbs.clear();
  featureProbs.clear();
  featureProbs.reserve(60000);

  NUM_PHI_OFFSET = ACTION_OFFSET + numActions;
}

void SarsaEBLearner::update_action_marginals() {}

void SarsaEBLearner::update_phi(vector<long long & features>,
                                int action,
                                long time_step) {
  // Updating the p(phi) and p(a/phi)
  // p(phi)  : rho_{t+1} = ((rho_t * (t + 1)) + phi_{t + 1}) / (t + 2)

  // Partial Update for all the seen features.
  // rho_{t+1}^' = rho_t * (t + 1) / (t + 2)
  for (auto it = featureProbs.begin(); it != featureProbs.end(); ++it) {
    it->second[0] *= ((time_step + 1.0) / (time_step + 2));
  }

  // Partial Update only for all the currently active features.
  // rho_{t + 1} = rho_{t + 1}^'  + (phi_{t + 1} / (t + 2))
  for (long long featIdx : features) {
    featureProbs[featIdx][0] += (1.0 / (time_step + 2));
  }
}

void SarsaEBLearner::update_action_given_phi(
    unordered_map<long long, vector<double>>& tmp_featureProbs,
    vector<long long>& features,
    int action,
    long time_step) {
  // p(a/phi): p_{t + 1}(a / phi) =
  //    p_t * (n_{phi} + 1) / (n_{phi} + 2) + I[a = cur_act] / (n_{phi} + 2)
  double n_phi = 0;
  for (long long featIdx : features) {
    n_phi = tmp_featureProbs[featIdx][NUM_PHI_OFFSET];
    for (int a = 0; a < numActions; a++) {
      tmp_featureProbs[featIdx][a + ACTION_OFFSET] *=
          (n_phi + 1.0) / (n_phi + 2);
      if (a == action) {
        tmp_featureProbs[featIdx][a + ACTION_OFFSET] += (1.0 / n_phi + 2);
      }
    }
  }
}

bool SarsaEBLearner::add_new_feature_to_map(long long featIdx, int time_step) {
  // Creating new vector to store needed data for active feature.
  vector<double> v(ACTION_OFFSET + numActions + 1);
  v.push_back(0.5 / (time_step + 1);
  v.push_back(0);

  // p(a=cur_act/phi_i=1) = \frac{n_{cur_act} + (1/numActions)}{n_phi + 1}
  // Here, n_{a} = 0
  for (int action = 0; action < numActions; action++) {
    v.push_back(1.0 / (numActions * 2));
  }
  v.push_back(0);
  featureProbs.insert(std::make_pair(featIdx, v));
}

double SarsaEBLearner::get_sum_log_phi(vector<long long>& features,
                                       long time_step) {
  double sum_log_phi = 0;

  // Iterating over the features to calculate the joint
  // TODO: Don't store the last p(a/phi_i),
  //       calculate it as 1 - \sum_{j=1}^{numActions-1} p(a_j/phi_i)
  for (long long featIdx : features) {
    if (featureProbs.find(featIdx) == featureProbs.end()) {
      add_new_feature_to_map(featIdx, time_step)
    }

    // p(phi_i=1)
    sum_log_phi += log(featureProbs[featIdx][0]);

    // Increment n_{phi} as the feature is active.
    featureProbs[featIdx][NUM_PHI_OFFSET] += 1;

    // Set the feature as seen.
    featureProbs[featIdx][1] = 1;
  }

  // p(phi_i=0)
  for (auto it = featureProbs.begin(); it != featureProbs.end(); ++it) {
    if (it->second[1] == 0) {
      sum_log_phi += log(1 - it->second[0]);
    } else {
      // Reset the seen features to unseen as its update has already been
      // done.
      it->second[1] = 0;
    }
  }

  return sum_log_phi;
}

double SarsaEBLearner::get_sum_log_action_given_phi(vector<long long>& features,
                                                    int action,
                                                    long time_step) {
  double sum_log_action_given_phi = 0;

  // ASSUMPTION: New features has been added to featureProbs in function
  //  get_sum_log_phi

  for (long long featIdx : features) {
    // p(a=cur_act/phi_i=1)
    sum_log_action_given_phi +=
        log(featureProbs[featIdx][action + ACTION_OFFSET]);
    // Set the feature as seen.
    featureProbs[featIdx][1] = 1;
  }

  // p(a=cur_act/phi_i=0) =
  //        (p(a=cur_act) - p(a=cur_act/phi_i=1)*p(phi_i=1))/(1-phi_i=1)
  for (auto it = featureProbs.begin(); it != featureProbs.end(); ++it) {
    // Update the probabilities for the inactive features.
    if (it->second[1] == 0) {
      // TODO: fix underfow issue.
      sum_log_action_given_phi +=
          log(actionMarginals[action] -
              (it->second[0] * it->second[action + ACTION_OFFSET])) -
          log(1 - it->second[0]);
    } else {
      // Reset the seen features to unseen as its update has already been done.
      it->second[1] = 0;
    }
  }

  return sum_log_action_given_phi;
}

void SarsaEBLearner::exploration_bonus(vector<long long>& features,
                                       long time_step,
                                       vector<double>& act_exp) {
  // Calculate p(phi)
  double sum_log_rho_phi = get_sum_log_phi(features, time_step);

  // Calculate for all action in Actions p(action,phi)
  vector<double> log_joint_phi_action(numActions);
  for (int action = 0; action < numActions; action++) {
    log_joint_phi_action.push_back(
        sum_log_rho_phi +
        get_sum_log_action_given_phi(features, action, time_step));
  }

  // Update the phi values with the currently seen features.
  update_phi(features, time_step);

  // Calculate p'(phi)
  double sum_log_rho_phi_prime = get_sum_log_phi(features, time_step);
  unordered_map<long long, vector<double>> tmp_featureProbs;
  double pseudo_count = 0;
  double sum_log_a_given_phi_prime = 0;
  for (int action = 0; action < numActions; action++) {
    // Copy original Map to tmp Map
    tmp_featureProbs = featureProbs;

    // Update the action given phi values for the current action and features.
    update_action_given_phi(tmp_featureProbs, features, action, time_step);

    // Calculate p'(a,phi)
    log_joint_phi_action_prime =
        sum_log_rho_phi_prime +
        get_sum_log_action_given_phi(features, action, time_step);

    // calculate pseudo count as (1/(exp(p(phi' - p()))-1))
    pseudo_count =
        1.0 /
        (exp(log_joint_phi_action_prime - log_joint_phi_action[action]) - 1);

    // push the exploration bonus to the output vector.
    act_exp.push_back(beta / sqrt(pseudo_count + 0.01));
  }
}

void SarsaEBLearner::learnPolicy(ALEInterface& ale, Features* features) {
  struct timeval tvBegin, tvEnd, tvDiff;
  vector<float> reward;
  double elapsedTime;
  double cumReward = 0, prevCumReward = 0;
  sawFirstReward = 0;
  firstReward = 1.0;
  vector<float> episodeResults;
  vector<int> episodeFrames;
  vector<double> episodeFps;

  long time_step = 1;

  long long trueFeatureSize = 0;
  long long trueFnextSize = 0;

  // Repeat (for each episode):
  // This is going to be interrupted by the ALE code since I set max_num_frames
  // beforehand
  for (int episode = episodePassed + 1;
       totalNumberFrames < totalNumberOfFramesToLearn; episode++) {
    // random no-op
    unsigned int noOpNum = 0;
    if (randomNoOp) {
      noOpNum = (*agentRand)() % (noOpMax) + 1;
      for (int i = 0; i < noOpNum; ++i) {
        ale.act(actions[0]);
      }
    }

    // We have to clean the traces every episode:
    for (unsigned int a = 0; a < nonZeroElig.size(); a++) {
      for (unsigned long long i = 0; i < nonZeroElig[a].size(); i++) {
        long long idx = nonZeroElig[a][i];
        e[a][idx] = 0.0;
      }
      nonZeroElig[a].clear();
    }

    F.clear();
    features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(), F);
    trueFeatureSize = F.size();
    groupFeatures(F);
    updateQValues(F, Q);

    currentAction = epsilonGreedy(Q, episode);
    gettimeofday(&tvBegin, NULL);
    vector<float> tmp_Q;
    vector<double> act_exp(numActions);
    int lives = ale.lives();
    // Repeat(for each step of episode) until game is over:
    // This also stops when the maximum number of steps per episode is reached
    while (!ale.game_over()) {
      reward.clear();
      reward.push_back(0.0);
      reward.push_back(0.0);
      updateQValues(F, Q);
      updateReplTrace(currentAction, F);

      sanityCheck();
      // Take action, observe reward and next state:
      act(ale, currentAction, reward);
      cumReward += reward[1];
      if (!ale.game_over()) {
        // Obtain active features in the new state:
        Fnext.clear();
        features->getActiveFeaturesIndices(ale.getScreen(), ale.getRAM(),
                                           Fnext);
        exploration_bonus(Fnext, time_step, act_exp);
        trueFnextSize = Fnext.size();
        groupFeatures(Fnext);

        // Update Q-values for the new active features
        updateQValues(Fnext, Qnext);
        // nextAction = epsilonGreedy(Qnext, episode);
        tmp_Q = Qnext;
        for (int action = 0; action < numActions; action++) {
          tmp_Q[action] += act_exp[action];
        }
        nextAction = Mathematics::argmax(tmp_Q, agentRand);
        printf("nextAction: %d\n", nextAction);
      } else {
        nextAction = 0;
        for (unsigned int i = 0; i < Qnext.size(); i++) {
          Qnext[i] = 0;
        }
      }
      // To ensure the learning rate will never increase along
      // the time, Marc used such approach in his JAIR paper
      if (trueFeatureSize > maxFeatVectorNorm) {
        maxFeatVectorNorm = trueFeatureSize;
        learningRate = alpha / maxFeatVectorNorm;
      }
      delta = reward[0] + act_exp[nextAction] + gamma * Qnext[nextAction] -
              Q[currentAction];
      // Update weights vector:
      for (unsigned int a = 0; a < nonZeroElig.size(); a++) {
        for (unsigned int i = 0; i < nonZeroElig[a].size(); i++) {
          long long idx = nonZeroElig[a][i];
          w[a][idx] = w[a][idx] + learningRate * delta * e[a][idx];
        }
      }
      F = Fnext;
      trueFeatureSize = trueFnextSize;
      currentAction = nextAction;
      time_step++;
    }

    // for (auto it = featureProbs.begin(); it != featureProbs.end(); ++it) {
    //   std::cout << "Prob for feature: " << it->first << ":" << it->second[0]
    //             << "," << it->second[1] << endl;
    // }

    gettimeofday(&tvEnd, NULL);
    timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
    elapsedTime = double(tvDiff.tv_sec) + double(tvDiff.tv_usec) / 1000000.0;

    double fps = double(ale.getEpisodeFrameNumber()) / elapsedTime;
    printf(
        "episode: %d,\t%.0f points,\tavg. return: %.1f,\t%d frames,\t%.0f "
        "fps\n",
        episode, cumReward - prevCumReward, (double)cumReward / (episode),
        ale.getEpisodeFrameNumber(), fps);
    episodeResults.push_back(cumReward - prevCumReward);
    episodeFrames.push_back(ale.getEpisodeFrameNumber());
    episodeFps.push_back(fps);
    totalNumberFrames +=
        ale.getEpisodeFrameNumber() - noOpNum * numStepsPerAction;
    prevCumReward = cumReward;
    features->clearCash();
    ale.reset_game();
    if (toSaveCheckPoint && totalNumberFrames > saveThreshold) {
      saveCheckPoint(episode, totalNumberFrames, episodeResults,
                     saveWeightsEveryXFrames, episodeFrames, episodeFps);
      saveThreshold += saveWeightsEveryXFrames;
    }
  }
}

SarsaEBLearner::~SarsaEBLearner() {}