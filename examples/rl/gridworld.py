#-----------------------------------
# Author: Sneha Reddy Aenugu
# Course: COMPSCI 687
# Description: HW2 Programming task
#----------------------------------

import numpy as np
import math
import argparse
from multiprocessing import Pool
from functools import partial
import random
import pickle
from matplotlib import pyplot as plt

action_set = {0:"AU", 1:"AD", 2:"AL", 3:"AR"}
improvement_array = []


class GridWorld687:
    def __init__(self):
        current_state = None;
        current_action = None
        self.array = np.concatenate((np.ones(80), np.ones(5)*2, np.ones(5)*3, np.ones(10)*4))


    def transition_function(self, state, action):
        new_state = None
        reward = None
        if action == "U":
            if state <= 5 or state == 23 or state == 21:
                new_state = state
            elif (state > 5 and state < 13) or state == 22:
                new_state = state - 5
            else:
                new_state = state - 4

        elif action == "D":
            if state >= 19 or state == 8:
                new_state = state
            elif state <= 7 or state == 17 or state == 18:
                new_state = state + 5
            else:
                new_state = state + 4

        elif action == "L":
            if state == 13 or state == 17 or state == 23 \
                    or state == 1 or state == 6 or state == 11\
                    or state == 15 or state == 19:
                new_state = state
            else:
                new_state = state - 1

        elif action == "R":
            if state == 12 or state == 16 or state == 5 \
                    or state == 10 or state == 14 or state == 18 \
                    or state == 23:
                new_state = state
            else:
                new_state = state + 1

        else:
            new_state = state
        
        return new_state

    def reward_function(self, state):
        if state == 21:
            reward = -10
        elif state == 23:
            reward = 10
        else:
            reward = 0
        return reward

    def action_from_attempt(self, attempt):
        #res = np.random.choice(np.arange(1, 5), p=[0.8, 0.05, 0.05, 0.1])
        res = self.array[random.randint(0, len(self.array)-1)]
        if attempt == "AU":
            if res == 1:
                action = "U"
            elif res == 2:
                action = "L"
            elif res == 3:
                action = "R"
            else:
                action = "N"

        elif attempt == "AD":
            if res == 1:
                action = "D"
            elif res == 2:
                action = "R"
            elif res == 3:
                action = "L"
            else:
                action = "N"

        elif attempt == "AL":
            if res == 1:
                action = "L"
            elif res == 2:
                action = "D"
            elif res == 3:
                action = "U"
            else:
                action = "N"

        elif attempt == "AR":
            if res == 1:
                action = "R"
            elif res == 2:
                action = "U"
            elif res == 3:
                action = "D"
            else:
                action = "N"

        return action


    def get_discounted_returns(self, rewards, gamma=0.9):
        if len(rewards) >= 300:
            return 0
        discounted_reward = 0
        for i in range(len(rewards)):
            discounted_reward += math.pow(gamma, i)*rewards[i]
        return discounted_reward

    def run_one_episode_random_policy(self, initial_state):
        states = []
        rewards = []
        actions = []
        state = initial_state
        while state != 23:
            attempt_action = action_set[np.random.choice(np.arange(0,4), 1)[0]]
            action = self.action_from_attempt(attempt_action)
            new_state = self.transition_function(state, action)
            reward = self.reward_function(new_state)
            states.append(new_state)
            rewards.append(reward)
            actions.append(action)
            state = new_state
        return states, rewards, actions

    def run_given_policy(self, initial_state, policy):
        states = []
        rewards = []
        actions = []
        state = initial_state
        count = 0
        while state != 23 and count < 300:
            pr = policy[state-1]
            array = np.concatenate((np.zeros(int(pr[0]*100)), np.ones(int(pr[1]*100)), 2*np.ones(int(pr[2]*100)), 3*np.ones(int(pr[3]*100))))
            attempt_action = action_set[array[random.randint(0, len(array)-1)]] 
            #attempt_action = action_set[np.random.choice(np.arange(0, 4), p=policy[state-1])]
            action = self.action_from_attempt(attempt_action)
            new_state = self.transition_function(state, action)
            reward = self.reward_function(new_state)
            states.append(new_state)
            rewards.append(reward)
            actions.append(action)
            state = new_state
            count += 1
        return states, rewards, actions


    def run_epsilon_greedy_policy(self, epsilon, q_func):
        states = []
        rewards = []
        actions = []
        state = 1
        count = 0
        action_ind = np.argmax(q_func[state])
        while state != 23 and count < 300:
            prob = (epsilon/4)*np.ones(4)
            prob[action_ind] = (1 - epsilon) + (epsilon/4)
            attempt = action_set[int(np.random.choice(4, 1, p=prob))]
            action = self.action_from_attempt(attempt)
            new_state = self.transition_function(state, action)
            new_action_ind = np.argmax(q_func[new_state])
            reward = self.reward_function(new_state)
            rewards.append(reward)
            state = new_state
            action_ind = new_action_ind
            count += 1
        return rewards



class SarsaGW():
    def __init__(self, epsilon, alpha, lda):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = 0.9
        self.lda = lda
        #self.sigma = 10 #selection
        self.sigma = 10
        self.q = 10*np.ones((24, 4)) + np.random.normal(0,1, size=(24,4))
        self.gw = GridWorld687()

    def e_greedy_policy(self, action_ind):
        prob = (self.epsilon/4)*np.ones(4)
        prob[action_ind] = (1 - self.epsilon) + (self.epsilon/4)
        return prob

    def softmax_selection(self, qvalues, sigma):
        eps = 1e-5
        qvalues = qvalues + eps
        prob = np.exp(sigma*qvalues)/sum(np.exp(sigma*qvalues))
        return prob


    def run_sarsa(self, num_episodes, selection='egreedy'):
        returns = []
        for i in range(num_episodes):
            rewards = []
            state = 1
            etraces = np.zeros(24)
            if selection == 'egreedy':
                action_ind = np.argmax(self.q[state])
                prob = self.e_greedy_policy(action_ind)
            else:
                action_ind = np.argmax(self.q[state])
                prob1 = self.e_greedy_policy(action_ind)
                prob = self.softmax_selection(self.q[state], self.sigma)

            attempt = int(np.random.choice(4, 1, p=prob))
            while(state != 23):
                action = self.gw.action_from_attempt(action_set[attempt])
                new_state = self.gw.transition_function(state, action)
                reward = self.gw.reward_function(new_state)

                if selection == 'egreedy':
                    new_action_ind = np.argmax(self.q[new_state]) 
                    prob = self.e_greedy_policy(new_action_ind)
                else:
                    prob = self.softmax_selection(self.q[state], self.sigma)

                new_attempt = int(np.random.choice(4, 1, p=prob))
                delta_t = reward + self.gamma*self.q[new_state][new_attempt] - self.q[state][attempt]

                etraces *= self.gamma*self.lda
                etraces[state] += 1
                #self.q[state][attempt] += self.alpha*delta_t
                self.q[:,attempt] += self.alpha*delta_t*etraces
                state = new_state
                attempt = new_attempt
                rewards.append(reward)
            discounted_return = self.gw.get_discounted_returns(rewards)
            if i%20 == 0 or i==99:
                print("Discounted Return after %s episodes: %s" %(i, discounted_return))

            returns.append(discounted_return)
        return returns


    def run_q_learning(self, num_episodes, selection='egreedy'):
        returns = []
        for i in range(num_episodes):
            rewards = []
            state = 1
            etraces = np.zeros(24)
            while(state != 23):
                if selection == 'egreedy':
                    action_ind = np.argmax(self.q[state])
                    prob = self.e_greedy_policy(action_ind)
                else:
                    prob = self.softmax_selection(self.q[state], self.sigma)

                attempt = int(np.random.choice(4, 1, p=prob))
                action = self.gw.action_from_attempt(action_set[attempt])
                new_state = self.gw.transition_function(state, action)
                reward = self.gw.reward_function(new_state)
                new_action_ind = np.argmax(self.q[new_state]) 
                new_attempt = new_action_ind
                etraces *= self.gamma*self.lda
                etraces[state] = 1
                self.q[:,attempt] += self.alpha*(reward + self.gamma*self.q[new_state][new_attempt] - self.q[state][attempt])*etraces
                state = new_state
                attempt = new_attempt
                rewards.append(reward)
            discounted_return = self.gw.get_discounted_returns(rewards)
            returns.append(discounted_return)
            if i%20 == 0 or i==99:
                print("Mean Discounted Return after %s episodes: %s" %(i, discounted_return))
        return returns

    def tabular_softmax(self, policy):
        softmax_policy = [np.exp(policy[i,:])/sum(np.exp(policy[i])) for i in range(policy.shape[0])]
        return softmax_policy

    def run_actor_critic(self, num_episodes):
        returns = []
        theta = np.random.rand(24, 4)
        v_f = np.random.rand(24)
        alpha = 1
        beta = 1
        for i in range(num_episodes):
            rewards = []
            state = 1
            e_theta = np.zeros_like(theta)
            e_v = np.zeros_like(v_f)
            gamma = 0.9
            count = 0
            while state != 23 and count < 300:
                #print(count)
                # Act using actor
                policy = self.tabular_softmax(theta)
                pr = policy[state-1]
                array = np.concatenate((np.zeros(int(pr[0]*100)), np.ones(int(pr[1]*100)), 2*np.ones(int(pr[2]*100)), 3*np.ones(int(pr[3]*100))))
                attempt = array[random.randint(0, len(array)-1)].astype(int)
                action = self.gw.action_from_attempt(action_set[attempt])
                new_state = self.gw.transition_function(state, action)
                reward = self.gw.reward_function(new_state)

                # Critic update
                e_v *= gamma*self.lda
                e_v[state-1] += 1
                delta_t = reward + gamma*v_f[new_state-1] - v_f[state-1]
                v_f += alpha*delta_t*e_v

                # Actor update
                state_param = theta[state-1]
                dlnpi = -1*np.exp(state_param[attempt])*np.exp(state_param)/math.pow(sum(np.exp(state_param)),2)
                dlnpi[attempt] += np.exp(state_param[attempt])/math.pow(sum(np.exp(state_param)),1)
                e_theta *= gamma*self.lda
                e_theta[state-1] += dlnpi
                #print(state, action)
                alpha1 = self.alpha
                theta += beta*delta_t*e_theta
                state = new_state
                count += 1
                rewards.append(reward)
            #print("After %s episodes: %s " %(i, count))

            discounted_return = self.gw.get_discounted_returns(rewards)
            returns.append(discounted_return)
            if i%20 == 0 or i==99:
                print("Discounted Return after %s episodes: %s" %(i, discounted_return))
        return returns

    def run_reinforce(self, num_episodes):
        returns = []
        #theta = np.random.rand(24, 4)
        theta = 5*np.ones((24, 4))
        gamma = 0.9
        for i in range(num_episodes):
            dlnpis = []
            states = []
            count = 0
            state = 1
            e_theta = np.zeros_like(theta)
            rewards = []
            while state != 23 and count < 300:
                #print(count)
                # Act using actor
                policy = self.tabular_softmax(theta)
                pr = policy[state-1]
                array = np.concatenate((np.zeros(int(pr[0]*100)), np.ones(int(pr[1]*100)), 2*np.ones(int(pr[2]*100)), 3*np.ones(int(pr[3]*100))))
                attempt = array[random.randint(0, len(array)-1)].astype(int)
                action = self.gw.action_from_attempt(action_set[attempt])
                new_state = self.gw.transition_function(state, action)
                reward = self.gw.reward_function(new_state)

                state_param = theta[state-1]
                dlnpi = -1*np.exp(state_param[attempt])*np.exp(state_param)/math.pow(sum(np.exp(state_param)),2)
                dlnpi[attempt] += np.exp(state_param[attempt])/sum(np.exp(state_param))
                #print(state, new_state, reward, dlnpi)
                dlnpis.append(dlnpi)
                states.append(state)
                rewards.append(reward)
                state = new_state
                count += 1

            g_t_array = [rewards[i:] for i in range(len(rewards))]
            g_t = np.array([*map(self.gw.get_discounted_returns, g_t_array)])

            del_j = np.zeros_like(theta)

            for j in range(len(states)):
                del_j[states[j]-1] += (gamma**j)*g_t[j]*dlnpis[j] 
            theta += self.alpha*del_j
        
            discounted_return = self.gw.get_discounted_returns(rewards)
            returns.append(discounted_return)
            if i%20 == 0 or i==99:
                print("Discounted Return after %s episodes: %s" %(i, discounted_return))
        return returns



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', dest='algorithm', default='sarsa')
    parser.add_argument('--selection', dest='selection', default='egreedy')
    parser.add_argument('--num_trials', dest='num_trials', default=10)
    parser.add_argument('--num_episodes', dest='num_episodes', default=100)
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
        td_cp = SarsaGW(epsilon=epsilon, alpha=step_size, lda=lda)
        if args.algorithm == 'sarsa':
            print('Sarsa')
            if args.selection == 'egreedy':
                print('egreedy')
                rewards = td_cp.run_sarsa(int(args.num_episodes))
            else:
                rewards = td_cp.run_sarsa(int(args.num_episodes), selection='softmax')
        elif args.algorithm == 'qlearning':
            print('QLearning')
            if args.selection == 'egreedy':
                print('egreedy')
                rewards = td_cp.run_q_learning(int(args.num_episodes))
            else:
                rewards = td_cp.run_q_learning(int(args.num_episodes), selection='softmax')

        elif args.algorithm == 'ac':
            rewards = td_cp.run_actor_critic(int(args.num_episodes))

        elif args.algorithm == 'reinforce':
            rewards = td_cp.run_reinforce(int(args.num_episodes))
        rewards_trials.append(rewards)
    
    f = open('rewards_ac.pkl', 'wb')
    pickle.dump(rewards_trials, f)

    print("Maximum reward reached at the end of 100 episodes : ", np.mean(rewards_trials, axis=0)[-1] )
    if args.plot:
        episodes = np.linspace(0,int(args.num_episodes)-1,int(args.num_episodes))
        rewards_mean = np.mean(rewards_trials, axis=0)
        rewards_std = np.std(rewards_trials, axis=0)
        plt.errorbar(episodes, rewards_mean, rewards_std)
        plt.ylabel('Mean reward')
        plt.xlabel('Number of episodes')
        plt.show()

