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

  featureProbs.clear();
  featureProbs.reserve(60000);
}

void SarsaEBLearner::update_prob_feature(vector<long long>& features,
                                         long time_step) {
  // Update Formula: rho_{t+1} = ((rho_t * (t + 1)) + phi_{t + 1}) / (t + 2)

  // Update p1: rho_{t+1}^' = rho_t * (t + 1) / (t + 2)
  for (auto it = featureProbs.begin(); it != featureProbs.end(); ++it) {
    featureProbs[it->first][0] =
        it->second[0] * ((time_step + 1.0) / (time_step + 2));
    it->second[1] = 0;
  }

  // Update p2: rho_{t + 1} = rho_{t + 1}^'  + (phi_{t + 1} / (t + 2))
  for (long long featIdx : features) {
    featureProbs[featIdx][0] =
        featureProbs[featIdx][0] + (1.0 / (time_step + 2));
  }
}

double SarsaEBLearner::feature_log_joint_prob(vector<long long>& features,
                                              long time_step) {
  double log_joint = 0;
  for (long long featIdx : features) {
    if (featureProbs.find(featIdx) == featureProbs.end()) {
      vector<double> v = {0.5 / (time_step + 1), 0};
      featureProbs.insert(std::make_pair(featIdx, v));
    }
    log_joint += log(featureProbs[featIdx][0]);
    featureProbs[featIdx][1] = 1;
  }

  for (auto it = featureProbs.begin(); it != featureProbs.end(); ++it) {
    if (it->second[1] == 0) {
      log_joint += log(1 - it->second[0]);
    } else {
      it->second[1] = 0;
    }
  }

  return log_joint;
}

double SarsaEBLearner::exploration_bonus(vector<long long>& features,
                                         long time_step) {
  double sum_log_rho = feature_log_joint_prob(features, time_step);
  update_prob_feature(features, time_step);
  double sum_log_rho_prime = feature_log_joint_prob(features, time_step);
  double pseudo_count = 1 / (exp(sum_log_rho_prime - sum_log_rho) - 1);
  return beta / sqrt(pseudo_count + 0.01);
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
    double cur_exp_bonus = 0;
    vector<float> tmp_Q;
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
        cur_exp_bonus = exploration_bonus(Fnext, time_step);
        trueFnextSize = Fnext.size();
        groupFeatures(Fnext);

        // Update Q-values for the new active features
        updateQValues(Fnext, Qnext);
        // nextAction = epsilonGreedy(Qnext, episode);
        tmp_Q = Qnext;
        std::transform(tmp_Q.begin(), tmp_Q.end(), tmp_Q.begin(),
                       bind2nd(std::plus<double>(), cur_exp_bonus));
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
      delta = reward[0] + cur_exp_bonus + gamma * Qnext[nextAction] -
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