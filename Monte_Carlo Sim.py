# Monte Carlo coding homework for RBE 595
import random
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

    def takeAction(self, u):
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
        self.curState=self.curState+action
        if(self.curState==5):
            return self.r5
        elif(self.curState==0):
            return self.r0
        else:
            return self.rMiddle

if __name__ == '__main__':
    simulator=stochasticCleaningSim()
    print(simulator.curState)
    print(simulator.takeAction(1))
    print(simulator.curState)
