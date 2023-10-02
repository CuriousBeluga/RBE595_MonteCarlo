# Monte Carlo coding homework for RBE 595
# Authors: Sean Tseng, Jonathan Landay
import random
import numpy as np
import matplotlib.pyplot as plt

class stochasticCleaningSim:
    def __init__(self,epsilon):
        #World Size and start from problem statement
        self.maxState=5
        self.minState=0
        self.curState=3
        # number of states and actions
        self.num_state = 6
        self.num_actions = 2
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
        self.policy=np.ones((self.num_state,self.num_actions))*epsilon/self.num_actions
        self.epsilon= epsilon
        #s
        self.action_history = []
        self.reward_history = []
        self.state_history = [self.curState]  # initialize with first state

    def resetSimulation(self):
        self.curState=3
        self.action_history = []
        self.reward_history = []
        self.state_history = []


    def takeAction(self, u,index):
        #Changes state by action u, +1 or -1, and returns the reward associated with the state
        # (This can easily be changed to reward new state and reward if that is preferable, but new state is stored in simulator class) 
        if self.curState==self.minState or self.curState==self.maxState:
            return 0 #Returns 0 reward if starting at end state, and no move
        p=random.random()
        if p<=self.pForward:#Chance to go forward
            action=u*1
        elif p<=self.pForward+self.pStuck:#Chance to stay stuck instead
            action=u*0
        else:#Chance to go backwards instead
            action=u*-1


        if index == "sim":   # if using takeAction for simulation, append position, action and reward
            self.curState=self.curState+action
            self.action_history.append(u)
            self.state_history.append(self.curState)


        # return reward
        if(self.curState==5):
            return self.r5
        elif(self.curState==0):
            return self.r0
        else:
            return self.rMiddle

    def generate_policy(self):
        for state in range(self.num_state):
            possible_rewards = [self.takeAction(-1, "none"), self.takeAction(1, "none")]
            best_action = np.argmax(possible_rewards)
            if best_action ==0:
                other_action = 1
            else:
                other_action = 0
            epsilon = .9
            # index 0 is movement of -1, index 1 is movement of +1
            self.policy[state,best_action] = 1 -epsilon+epsilon/self.num_actions
            self.policy[state,other_action] = 1 - self.policy[state,best_action]
        print(f'This is the generated initial policy: {self.policy}')



    def generate_episode(self):
        self.curState = random.randint(1,4)
        self.state_history =[self.curState]
        while True:     # loop until terminal state is reached

            action = np.random.choice(self.num_actions,p=self.policy[self.curState])
            # highest_prob_item = np.argmax(self.policy[self.curState])

            # print(f'Current state: {self.curState}')
            if action == 0:  # index 0 is action -1, index 1 is action +1
                action = -1
            # print(f'Action taken: {action}')
            reward = self.takeAction(action,"sim")
            self.reward_history.append(reward)
            # break clause
            if self.curState == 0 or self.curState ==5:
                # last action has nothing
                self.action_history.append(0)
                self.reward_history.append(0)
                break

        # print(f'This is the full state history: {self.state_history}')


    def on_policy_monte_carlo(self,episodes,epsilon):


        # generate a soft epsilon policy

        value_function = np.zeros((6,2))
        # state visited container
        state_counter = np.zeros((6,2))
        returns = np.zeros((6,2))
        G = 0
        y = .5
        iterations = 40
        state_value = np.zeros((6,episodes))
        num_states = 6
        num_actions = 2
        sa_count = np.zeros((num_states,num_actions))
        # episode to state value container, each row is an episode, each column of the row is the value of each state
        episodic_values = np.zeros((episodes,num_states))
        # main algorithm body

        G = 0
        # initialize a policy using E-greedy
        self.generate_policy()

        for episode in range(episodes):
            # generate episode using policy
            self.generate_episode()


            state_action_pairs = []
            for i in range(len(self.state_history)):
                state_action_pairs.append([self.state_history[i],self.action_history[i]])
            # print(f'Episode {episode+1} state action pairs:{state_action_pairs}')
            for t in reversed(range(0,len(self.state_history))):

                current_reward = self.reward_history[t]
                # print(f'This is the reward for state {self.state_history[t]}: {current_reward}')
                G = G * y + current_reward
                current_state = self.state_history[t]
                current_action = self.action_history[t]
                current_pair = [current_state,current_action]

                if current_pair not in state_action_pairs[:t]:
                    state_counter[current_state,current_action] +=1
                    value_function[current_state,current_action] += (G-value_function[current_state,current_action]/state_counter[current_state][current_action])


                episodic_values[episode,current_state] = value_function[current_state,current_action]
                # # Carry over to the next episode
                # if episode < episodes+1:
                #     episodic_values[episode+1, current_state] = value_function[current_state, current_action]

            # #
            # for state in range(self.num_state):
            #     episodic_values[episode+1,state] = value_function[state,]

            for state in range(num_states): # updating policy
                action_star = np.argmax(value_function[state]) # find best action for each state
                if action_star == 0:
                    other_action = 1
                else:
                    other_action = 0
                # if the action is the optimal one, then use one equation
                self.policy[state,action_star] = 1 - self.epsilon + self.epsilon/self.num_actions
                # if action isn't the optimal one, use another equation
                self.policy[state,other_action] = 1 - self.policy[state][action_star]

            # print(f'Episode {episode + 1} policy: {self.policy}')
            print(f'Episode {episode+1} complete')
            # print(episodic_values)
        self.resetSimulation()
        return episodic_values,self.policy

    def graphEpisodes(self, episodic_values):
        for i,state in enumerate(episodic_values.T): #Iterate over the columnic states for figure i
            Y=state
            X=np.linspace(1,len(Y),len(Y)) #creates X  linspace equal to number of episodes from length of states

            plt.figure(i)
            plt.title(f'State {i} Value per episode')
            plt.plot(X,Y)
        plt.show()



if __name__ == '__main__':
    # simulator=stochasticCleaningSim()
    # print(simulator.curState)
    # print(simulator.takeAction(1),"sim")
    # print(simulator.curState)

    # on_site_monte_carlo_procedure
    epsilon = .5
    episodes = 1000
    on_policy_mc_agent=stochasticCleaningSim(epsilon)
    sim_state_values,policy = on_policy_mc_agent.on_policy_monte_carlo(episodes,epsilon)
    print(f'Optimal Values, (index = state): {sim_state_values[-1]}')
    print(f'Here is the optimal policy:\n {policy}')
    on_policy_mc_agent.graphEpisodes(sim_state_values)
