

import random as rand

import numpy as np


class QLearner(object):
    """  		  	   		 	   			  		 			     			  	 
    This is a Q learner object.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    :param num_states: The number of states to consider.  		  	   		 	   			  		 			     			  	 
    :type num_states: int  		  	   		 	   			  		 			     			  	 
    :param num_actions: The number of actions available..  		  	   		 	   			  		 			     			  	 
    :type num_actions: int  		  	   		 	   			  		 			     			  	 
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.  		  	   		 	   			  		 			     			  	 
    :type alpha: float  		  	   		 	   			  		 			     			  	 
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.  		  	   		 	   			  		 			     			  	 
    :type gamma: float  		  	   		 	   			  		 			     			  	 
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.  		  	   		 	   			  		 			     			  	 
    :type rar: float  		  	   		 	   			  		 			     			  	 
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.  		  	   		 	   			  		 			     			  	 
    :type radr: float  		  	   		 	   			  		 			     			  	 
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.  		  	   		 	   			  		 			     			  	 
    :type dyna: int  		  	   		 	   			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	   			  		 			     			  	 
    :type verbose: bool  		  	   		 	   			  		 			     			  	 
    """
    def author(self):
        return 'xz'

    def __init__(
        self,
        num_states=100,
        num_actions=4,
        alpha=0.2,
        gamma=0.9,
        rar=0.5,
        radr=0.99,
        dyna=0,
        verbose=False,
    ):
        """  		  	   		 	   			  		 			     			  	 
        Constructor method  		  	   		 	   			  		 			     			  	 
        """
        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.s = 0
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.q_table = np.zeros((num_states, num_actions))
        if dyna > 0:
            self.tc = np.full((num_states, num_actions, num_states), 0.00001)
            self.t = self.tc / self.tc.sum(axis=2, keepdims=True)
            self.r = np.zeros((num_states, num_actions))

    def querysetstate(self, s):
        """  		  	   		 	   			  		 			     			  	 
        Update the state without updating the Q-table  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        :param s: The new state  		  	   		 	   			  		 			     			  	 
        :type s: int  		  	   		 	   			  		 			     			  	 
        :return: The selected action  		  	   		 	   			  		 			     			  	 
        :rtype: int  		  	   		 	   			  		 			     			  	 
        """
        self.s = s

        random = rand.random()
        if random < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.q_table[s, :])

        if self.verbose:
            print(f"s = {s}, a = {action}")

        return action

    def query(self, s_prime, r):
        """  		  	   		 	   			  		 			     			  	 
        Update the Q table and return an action  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        :param s_prime: The new state  		  	   		 	   			  		 			     			  	 
        :type s_prime: int  		  	   		 	   			  		 			     			  	 
        :param r: The immediate reward  		  	   		 	   			  		 			     			  	 
        :type r: float  		  	   		 	   			  		 			     			  	 
        :return: The selected action  		  	   		 	   			  		 			     			  	 
        :rtype: int  		  	   		 	   			  		 			     			  	 
        """
        self.q_table[self.s, self.a] = (1 - self.alpha) * self.q_table[self.s, self.a] + \
                                 self.alpha * (r + self.gamma * np.max(self.q_table[s_prime]))

        random = rand.random()
        if random < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.q_table[s_prime, :])

        self.rar *= self.radr

        if self.dyna > 0:
            self.tc[self.s, self.a, s_prime] += 1
            self.t = self.tc / self.tc.sum(axis=2, keepdims=True)
            self.r[self.s, self.a] = (1 - self.alpha) * self.r[self.s, self.a] + (self.alpha * r)

            for _ in range(self.dyna):
                a_dyna = np.random.randint(self.num_actions)
                s_dyna = np.random.randint(self.num_states)

                s_prime_dyna = np.argmax(self.t[s_dyna,a_dyna])

                r = self.r[s_dyna, a_dyna]

                self.q_table[s_dyna, a_dyna] = (1 - self.alpha) * self.q_table[s_dyna, a_dyna] + \
                                         self.alpha * (r + self.gamma * np.max(self.q_table[s_prime_dyna,]))

        self.s = s_prime
        self.a = action

        if self.verbose:
            print(f"s = {s_prime}, a = {action}, r={r}")

        return action


if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
