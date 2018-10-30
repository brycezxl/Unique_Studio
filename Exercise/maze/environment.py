#!/usr/bin/env python
# coding=utf-8
import numpy as np
class environment():
    """
    Function:
        The environment is a maze which has treasure and trap, 
        your task is finding a shortest way to treasure. The 
        followings are some apis you should use.     
    """
    def __init__(self):
        self._reward = np.array([[0,0,0,0],
                       [0,1,-5,1],
                       [0,-5,5,2],
                       [0,1,2,1]])
        self._end_state =[(1,2), (2,1), (2,2)]
        self._start_state = (0,0)

    def start(self):
        """
        Function:
            Return the startial state
        Args:
            None
        Rerurn:
            start_state (tuple): (0, 0) 
        """
        return self._start_state


    def get_action(self, state):
        """
        Function:
            according to the input state, give the optional actions
        Args:
            state (tuple): (x, y)
        Return:
            a list of optional actions
        """
        udlr_set = [(1,1), (1,2), (2,1), (2,2)]
        udr_set = [(1,0), (2,0)]
        udl_set = [(1,3), (2,3)]
        dlr_set = [(0,1), (0,2)]
        ulr_set = [(3,1), (3,2)]
        if state in udlr_set:
            return ['up', 'down', 'left', 'right']
        elif state in udr_set:
            return ['up', 'down', 'right']
        elif state in udl_set:
            return ['up', 'down', 'left']
        elif state in dlr_set:
            return ['down', 'left', 'right']
        elif state in ulr_set:
            return ['up', 'left', 'right']
        elif state == (0,0):
            return ['down', 'right']
        elif state == (0,3):
            return ['down', 'left']
        elif state == (3,0):
            return ['up', 'right']
        else:
            return ['up', 'left']
            

    def get_reward(self, state, action):
        """
        Function: 
            Given current state and chosen action, get the 
            reward and next reward. If the next state is trop or 
            treasure, the process come to end.
        Args:
            state (tuple): (x, y)
            action (string): right or lelt or up or down
        Returns:
            next_state (tuple): next state (x', y')
            reward (int): the reward
            end (bool): end or not  
        """

        end = False
        if action == 'up':
            next_state = (state[0]-1, state[1])
        elif action == 'down':
            next_state = (state[0]+1, state[1])
        elif action == 'left':
            next_state = (state[0], state[1]-1)
        else:
            next_state = (state[0], state[1]+1)
        if next_state in self._end_state:
            end = True
        return  next_state, self._reward[next_state[0]][next_state[1]]-2, end




