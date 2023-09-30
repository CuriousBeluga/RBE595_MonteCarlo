# Monte Carlo coding homework for RBE 595
import random

import numpy as np
import matplotlib.pyplot as plt

class stochasticCleaningSim:
    def __init__(self):
        #World Size and start from problem statement
        self.maxState=5
        self.minState=0
        self.curState=3
        #probabilities to do non determ movement from problem statement
        self.pForward=0.8
        self.pStuck=0.15
        self.pReverse=0.05
        #rewards
        self.r5=5
        self.r0=1
        self.rMiddle=0
        # epsilon soft parameters
        self.N = 100
        self.mean = 0
        # on policy-first visit parameters
        self.agent_state_action = []    #container list for state action pairs
        self.first_visit_policy= []
    def takeAction(self, u,index):
        #Changes state by action u, +1 or -1, and returns the reward associated with the state
        # (This can easily be changed to reward new state and reward if that is preferable, but new state is stored in simulator class) 
        if self.curState==self.minState or self.curState==self.maxState:
            return 0 #Returns 0 reward if starting at end state, and no move
        action=u
        p=random.random()
        if p<=self.pForward:#Chance to go forward
            action=action*1
        elif p<=self.pForward+self.pStuck:#Chance to stay stuck instead
            action=action*0
        else:#Chance to go backwards instead
            action=action*-1

        if index == "sim":   # if using takeAction for simulation, update position
            self.curState=self.curState+action
        else:
            pass

        if(self.curState==5):
            return self.r5
        elif(self.curState==0):
            return self.r0
        else:
            return self.rMiddle


    def firstVisitMonteCarlo(self, episodes, epsilon):
        valueLists=[[(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]] #list of lists, [G Value, # of Entries for each state, i where state is index], jth list is the 
        #values at episode j (not 0 indexed, 0 episode is the assumption at start), this entry is the G values for 0th trial
        for i in range(episodes):
            episodeValues=self.runEpisode(epsilon, valueLists[i-1]) #Returns the values of the states at first visits from an episode with the given Gs
            valueLists[i]=self.updateValueList(valueLists[i-1], episodeValues)#updates the list of values according to G equations
        return valueLists


    def on_policy_monte_carlo(self,episodes,epsilon):

        #row = state, column is action reward
        soft_e_policy = []
        #
        local_state = 3
        #Iterations for soft E policy
        N = 50
        # generate a soft epsilon policy
        for i in range(N):

            if np.random.random() < epsilon:  # pick a number and compare against epsilon to determine random action
                action = self.takeAction(np.random.choice([-1,1]),"e-soft")
            else:  # choose the action from highest action-value pair
                rewards = [self.takeAction(-1,"e-soft"),self.takeAction(1,"e-soft")]
                action = np.argmax(rewards)
                if action == 0:
                    action = -1
            local_state = action + local_state
            # add chosen action to the policy
            soft_e_policy.append((local_state, action))
            if local_state == 0 or local_state == 5:
                break


        print(soft_e_policy)
        value_function = np.zeros((6,2))
        returns = np.zeros((6,2))
        G = 0
        y = .5
        iterations = 40
        state_value = np.zeros((6,episodes))
        num_states = 6
        num_actions = 2
        sa_count = np.zeros((num_states,num_actions))
        # main algorithm body

        # agent's state
        G = 0
        # length of soft_e_policy
        for episode in range(episodes):

            # generate episode based on soft_e policy
            for t in range(0,len(soft_e_policy)):
                state,action = soft_e_policy[t]

                if (state, action) not in self.agent_state_action:  #check if state and action have already happened
                    G = G*y+self.takeAction(action,"sim")
                self.agent_state_action.append((state,action)) # each state action index contains a list, i think

                for state,action in self.agent_state_action:
                    returns[state,action]+=(G)
                    sa_count[state,action] +=1
                    value_function[state,action] = returns[state,action]/sa_count[state,action]
                    # self.first_visit_policy = np.zeros((num_states,num_actions))

                    for state in range(num_states):
                        action_star = np.argmax(value_function[state,:])
                        for action in range(num_actions):
                            if action == action_star:
                                self.first_visit_policy[state,action] = 1 -epsilon + epsilon/num_actions
                            else:
                                self.first_visit_policy[state,action] = epsilon/num_actions

            #     # q-value = average of returns at that state and action
            #     average_returns = np.divide(returns, np.maximum(1, np.sum(returns != 0, axis=1)[:, np.newaxis]))
            # for state in range(num_states):
            #     state_returns = [returns[s, a] for s, a in self.agent_state_action if s == state]
            #     if state_returns:
            #         state_value[state, episode] = np.mean(state_returns)

        # Plot state values versus number of episodes

        for state in range(num_states):
            plt.plot(range(episodes), value_function[state, :], label=f'State {state}')

        plt.xlabel('Number of Episodes')
        plt.ylabel('State Value')
        plt.legend()
        plt.title('State Value vs. Number of Episodes')
        plt.show()
        # return state_value


if __name__ == '__main__':
    # simulator=stochasticCleaningSim()
    # print(simulator.curState)
    # print(simulator.takeAction(1),"sim")
    # print(simulator.curState)

    # on_site_monte_carlo_procedure

    on_policy_mc_agent=stochasticCleaningSim()
    sim_state_value = on_policy_mc_agent.on_policy_monte_carlo(100,.5)
