## 라이브러리 임포트
import math
from typing import Optional, Union

import numpy as np

import gym
from gym import logger, spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled
from embedData import embedFlightData
#from functions import checkConnection, get_reward, update_state
from utils import checkConnection, get_reward, update_state


class CrewPairingEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
   
    def __init__(self, V_f_list, flight_list):
        # number of flights
        self.V_f_list = V_f_list
        self.flight_list = flight_list
        self.N_flight = len(V_f_list)
        self.flight_cnt = 0
        self.V_p_cnt = 0
        self.V_p_list = [[0,0,0,[0],[0],[0]] for i in range(self.N_flight)]
        self.output = [[] for i in range(self.N_flight)]
        self.state = self.V_f_list[0]
        self.terminated = False
        
        self.action_space = spaces.Discrete(self.N_flight)
        self.steps_beyond_terminated = None # step() 함수가 호출되었을 때, terminated가 True인 경우를 의미함.
    

    def step(self, action):
        V_f = self.V_f_list[self.flight_cnt]
        
        if action == 1:            
            reward = get_reward(self.V_p_list, V_f, self.V_p_cnt)
            update_state(self.V_p_list, V_f, self.V_p_cnt)
            self.output[self.V_p_cnt].append(self.flight_list[self.flight_cnt].id)

            self.flight_cnt += 1
            self.V_p_cnt = 0

            if self.flight_cnt == self.N_flight:
                self.terminated = True
                return self.state, reward, self.terminated, False, {}, self.output
        else:
            reward = 0
            self.V_p_cnt += 1

        V_f = self.auto_insert(self.V_f_list[self.flight_cnt])
        self.state = self.V_p_list[self.V_p_cnt][3] + V_f[4]

        if self.flight_cnt == self.N_flight:
            self.terminated = True

        return self.state, reward, self.terminated, False, {}, self.output

    def reset(self):
        # number of flights
        self.action_space = spaces.Discrete(self.N_flight)
        self.steps_beyond_terminated = None  # step() 함수가 호출되었을 때, terminated가 True인 경우를 의미함.

        self.steps_beyond_terminated = None
        self.V_p_list = [[0, 0, 0, [0], [0], [0]] for i in range(self.N_flight)]
        self.output = [[] for i in range(self.N_flight)]
        self.flight_cnt = 0
        self.terminated = False

        V_f = self.auto_insert(self.V_f_list[0])
        self.state = self.V_p_list[self.V_p_cnt][3] + V_f[4]  # V_p 출발공항 + V_f 도착공항

        return self.state, {}
    
    def auto_insert(self, V_f):
        while True:
            if self.V_p_list[self.V_p_cnt] == [0, 0, 0, [0], [0], [0]]:
                self.V_p_list[self.V_p_cnt] = V_f
                self.output[self.V_p_cnt].append(self.flight_list[self.flight_cnt].id)

                self.flight_cnt += 1

                if self.flight_cnt == self.N_flight:
                    break
                V_f = self.V_f_list[self.flight_cnt]

                self.V_p_cnt = 0

            elif not checkConnection(self.V_p_list[self.V_p_cnt], V_f):
                self.V_p_cnt += 1
            else:
                break
        return V_f