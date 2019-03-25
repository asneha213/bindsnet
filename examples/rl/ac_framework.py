#-----------------------------------
# Author: Sneha Reddy Aenugu
# Course: COMPSCI 687
# Description: HW2 Programming task
#----------------------------------

import numpy as np
import math
import argparse
import random
import itertools
import pdb
import pickle
from gridworld import GridWorld687
from simple_grid import SimpleGrid
from spike_rl import SpikingAgent

action_set = {0:"AU", 1:"AD", 2:"AL", 3:"AR"}


class ActorCritic():
    def __init__(self, epsilon, alpha, lda):
        self.epsilon = 0.01
        self.alpha = alpha
        self.gamma = 0.9
        self.lda = lda
        self.sigma = 1
        self.q = 10*np.ones((24, 4)) + np.random.normal(0,1, size=(24,4))
        self.gw = GridWorld687()
        #self.gw = SimpleGrid()
        self.actors = [SpikingAgent() for i in range(1)]

    def tabular_softmax(self, policy):
        softmax_policy = [np.exp(policy[i,:])/sum(np.exp(policy[i])) for i in range(policy.shape[0])]
        return softmax_policy

    def e_greedy_policy(self, action_ind):
        prob = (self.epsilon/4)*np.ones(4)
        prob[action_ind] = (1 - self.epsilon) + (self.epsilon/4)
        return prob

    def softmax_selection(self, qvalues, sigma):
        eps = 1e-5
        qvalues = qvalues + eps
        prob = np.exp(sigma*qvalues)/sum(np.exp(sigma*qvalues))
        return prob

    def run_actor_critic(self, num_episodes):
        returns = []
        theta = np.random.rand(24, 4)
        v_f = np.random.rand(24)
        alpha = 0.1
        for i in range(num_episodes):
            rewards = []
            states = []
            states.append(0)
            state = 1
            e_theta = np.zeros_like(theta)
            e_v = np.zeros_like(v_f)
            gamma = 0.9
            count = 0
            while state != 23 and count < 300:
                #print(count)
                # Act using actor
                o_rates = []
                action_rates = self.actors[0].forward(state)
                action_index = np.argmax(action_rates)
                #prob = self.e_greedy_policy(action_index)
                prob = self.softmax_selection(action_rates, self.sigma)
                attempt = int(np.random.choice(4, 1, p=prob))

                action = self.gw.action_from_attempt(action_set[attempt])
                if action != 'N':
                    action_ind = list(action_set.keys())[list(action_set.values()).index('A'+action)]
                else:
                    action_ind = attempt

                new_state = self.gw.transition_function(state, action)
                reward = self.gw.reward_function(new_state)
                #print(state, action, new_state)

                # Critic update
                e_v *= gamma*self.lda
                e_v[state-1] += 1
                delta_t = reward + gamma*v_f[new_state-1] - v_f[state-1]
                v_f += alpha*delta_t*e_v

                beta = 0.01
                # Actor update
                self.actors[0].update_weights(delta_t, state, action_ind, beta)

                state = new_state
                count += 1
                rewards.append(reward)
                states.append(new_state)
            #print("After %s episodes: %s " %(i, count))

            discounted_return = self.gw.get_discounted_returns(rewards)
            returns.append(discounted_return)
            if i%20 == 0 or i==99:
                print("Discounted Return after %s episodes: %s" %(i, discounted_return))
        return returns

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', dest='algorithm', default='sarsa')
    parser.add_argument('--selection', dest='selection', default='egreedy')
    parser.add_argument('--num_trials', dest='num_trials', default=100)
    parser.add_argument('--num_episodes', dest='num_episodes', default=200)
    parser.add_argument('--plot', dest='plot', action='store_true')

    args = parser.parse_args()


    rewards_trials = []


    if args.selection == 'egreedy':
        step_size = 0.1
        epsilon = 0.01
        lda = 0.5
    else:
        step_size = 0.1
        epsilon = 0.1
        #epsilon = 0.01

    step_size = 0.1

    for i in range(int(args.num_trials)):
        print('Trial:', i)
        td_cp = ActorCritic(epsilon=epsilon, alpha=step_size, lda=lda)
        rewards = td_cp.run_actor_critic(int(args.num_episodes))

        rewards_trials.append(rewards)
    print("Maximum reward reached at the end of 100 episodes : ", np.mean(rewards_trials, axis=0)[-1] )

    f = open('rewards.pkl', 'wb')
    pickle.dump(rewards_trials, f)
                                         
