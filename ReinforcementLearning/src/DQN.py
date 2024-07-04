import os
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.distributions import Categorical
from embedData import *
from utils import checkConnection, get_reward, update_state
from CrewPairingEnv import CrewPairingEnv
import random
import openpyxl
from datetime import datetime
import csv

# Hyperparameters
learning_rate = 0.005
gamma = 0.98
buffer_limit = 50000
batch_size = 32
INF = 999999999999999


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self, NN_Size):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(NN_Size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        if random.random() < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()
            
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    # 사용자로부터 실행할 데이터의 년월일, 에피소드 수, 실행 ID 입력 받기
    month = input("Enter the month of the input file (e.g., 201406): ")
    episodes = int(input("Enter the number of episodes to run: "))
    excutionId = input("Enter the excution ID (eg., 20240704-1)")

    current_directory = os.path.dirname(__file__)
    
    # 요구 디렉토리
    output_directory = os.path.join(current_directory, '../output')
    logs_directory = os.path.join(current_directory, '../logs')
    models_directory = os.path.join(current_directory, '../models')
    # 디렉토리가 없으면 만들기
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if not os.path.exists(logs_directory):
        os.makedirs(logs_directory)
    if not os.path.exists(models_directory):
        os.makedirs(models_directory)

    # 데이터 입력 받기
    path = os.path.abspath(os.path.join(current_directory, '../input'))
    readXlsx(path, f'/ASCP_Data_Input_{month}.xlsx')

    # 데이터 임베딩
    flight_list, V_f_list, NN_size = embedFlightData(path)
    print('Data Imported')
    #flight_list, V_f_list, NN_size = embedFlightData_Stratified(path)

    # Load Crew Pairing Environment
    N_flight = len(flight_list)
    print("Number of Flights :", N_flight)
    env = CrewPairingEnv(V_f_list, flight_list)
    q = Qnet(NN_size)

    q_target = Qnet(NN_size)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    score = -INF
    best_score = -INF
    output = [[] for i in range(N_flight)]

    # 에피소드 별 리워드와 소요 시간을 기록하는 로그 파일(csv)을 logs에 저장
    logs_filename = os.path.join(logs_directory, f'episode_rewards_{month}_{episodes}_{excutionId}.csv')
    with open(logs_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Episode", "Reward", "Best Score", "Time Elapsed"])
        csvwriter.writerow(["------------------------------------------"])
        time = datetime.now()
    
        for n_epi in range(episodes):
            print(f"Episode {n_epi}, Time Elapsed: {datetime.now() - time}")
            epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
            s, _ = env.reset()  # V_p departure airport, V_f arrival airport
            done = False
            output_tmp = []
            
            while not done:
                a = q.sample_action(torch.from_numpy(np.array(s)).float(), epsilon)
                s_prime, r, done, truncated, info, output_tmp = env.step(action=a)

                done_mask = 0.0 if done else 1.0
                memory.put((s, a, r / 100.0, s_prime, done_mask))

                s = s_prime  # flight changed by action
                score += r
            
            if memory.size() > 2000:
                train(q, q_target, memory, optimizer)

            if best_score < score:
                best_score = score
                output = output_tmp
                train(q, q_target, memory, optimizer)
                # Best Model이 갱신될 때 마다 models 디렉토리에 저장
                torch.save(q.state_dict(), os.path.join(models_directory, f'dqn_model_{month}_{episodes}_{excutionId}.pth'))
                print('Model Saved')
                
            csvwriter.writerow([n_epi, f"{score:.2f}", f"{best_score:.2f}", f"{datetime.now() - time}"])
            print(f"Current Score: {score:.2f}, Best Score: {best_score:.2f}")
                
            score = 0

    env.close()
    
    # 최종 생성된 페어링을 csv로 ouput 디렉토리에 저장
    print_xlsx(output, os.path.join(output_directory, f'output_pairing_{month}_{episodes}_{excutionId}.csv'))
    
if __name__ == '__main__':
    main()
