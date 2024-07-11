import numpy as np
import gym
from gym import spaces
from utils import checkConnection, get_reward, update_state

class CrewPairingEnv(gym.Env):
    def __init__(self, V_f_list, flight_list, airport_total):
        self.V_f_list = V_f_list
        self.flight_list = flight_list
        self.N_flight = len(V_f_list)
        self.flight_cnt = 0
        self.V_p_cnt = 0
        self.V_p_list = [[0, 0, 0, [0], [0], [0]] for _ in range(self.N_flight)]
        self.output = [[] for _ in range(self.N_flight)]
        self.state = self.V_f_list[0]
        self.terminated = False

        self.action_space = spaces.Discrete(self.N_flight)
        self.steps_beyond_terminated = None

        # Homebase 인덱스 설정
        self.homebase_index = airport_total.index("HB1")

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

        V_f = self.auto_insert(self.V_f_list[self.flight_cnt], is_first_flight=False)
        self.state = self.V_p_list[self.V_p_cnt][3] + V_f[4]

        if self.flight_cnt == self.N_flight:
            self.terminated = True

        return self.state, reward, self.terminated, False, {}, self.output

    def reset(self):
        self.action_space = spaces.Discrete(self.N_flight)
        self.steps_beyond_terminated = None

        self.V_p_list = [[0, 0, 0, [0], [0], [0]] for _ in range(self.N_flight)]
        self.output = [[] for _ in range(self.N_flight)]
        self.flight_cnt = 0
        self.terminated = False

        V_f = self.auto_insert(self.V_f_list[0], is_first_flight=True)
        self.state = self.V_p_list[self.V_p_cnt][3] + V_f[4]

        return self.state, {}

    def auto_insert(self, V_f, is_first_flight):
        while True:
            if self.V_p_list[self.V_p_cnt] == [0, 0, 0, [0], [0], [0]]:
                self.V_p_list[self.V_p_cnt] = V_f
                self.output[self.V_p_cnt].append(self.flight_list[self.flight_cnt].id)

                self.flight_cnt += 1

                if self.flight_cnt == self.N_flight:
                    break
                V_f = self.V_f_list[self.flight_cnt]

                self.V_p_cnt = 0

            elif not checkConnection(self.V_p_list[self.V_p_cnt], V_f,self.homebase_index):
                self.V_p_cnt += 1
            else:
                break
        return V_f