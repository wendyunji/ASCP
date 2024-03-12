# logs 폴더에서 파일 읽어서 파일 이름과 맨 마지막 줄의 best score을 리스트에 저장하기
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

class OptaVisualization:
    def __init__(self, grid, dir_name_list, filename):
        # 여러개의 디렉토리를 하나의 그래프로 그리기 위해 디렉토리 이름을 저장하기
        self.dir_name_list = dir_name_list
        self.filename = filename
        self.logs_path_list = [os.path.join(os.path.dirname(os.path.abspath(__file__)),'logs/'+dir_name) for dir_name in dir_name_list]
        self.logs_list = [os.listdir(logs_path) for logs_path in self.logs_path_list]
        
        self.dir_best_scores = {}
        # x축의 시간 간격(분)을 입력받아서 그 시간 간격마다의 best score를 저장하기
        self.grid_sec = grid * 60000
        
    def get_best_scores(self, logs, logs_path):
        """
        파일 line예시
        14:20:45.095     LS step (1), time spent (831), score (0hard/-54807478200soft),     best score (0hard/-54807478200soft), accepted/selected move count (1/2), picked move (F39336 {Pairing - 39336 { pair=[F39336] }[0]} <-> F32112 {Pairing - 32112 { pair=[F32112] }[0]}).
        best score를 갱신할 때마다 time spent와 best score를 저장하기
        """
        best_score = None
        best_scores = []
        # logs의 파일들 이름순 정렬
        logs.sort()
        for log in tqdm(logs):
            # print(log)
            with open(os.path.join(logs_path, log), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    score = re.findall(r'best score \(\d+hard/-(\d+)soft\)', line)
                    time_spent = re.findall(r'time spent \((\d+)\)', line)
                    if score:
                        try:
                            score = int(score[0])
                        except:
                            continue
                        if best_score is None or score < best_score:
                            best_score = score
                            #print('time spent:', time_spent[0], 'best score:', best_score)
                            best_scores.append((int(time_spent[0]), best_score))
                            
        return best_scores
    
    def get_best_scores_by_grid(self, logs, logs_path):
        """
        grid_sec마다의 best score를 저장하기
        """
        best_score_list = self.get_best_scores(logs, logs_path)
        # print(best_score_list)
        best_score_by_grid = []

        # print(best_score_list)
        # 제일 첫번째 시간은 0으로 만들고 값은 제일 첫번째 입력값으로 만들기
        best_score_by_grid.append((0, best_score_list[0][1]))

        for i in tqdm(range(1, best_score_list[-1][0], self.grid_sec)):
            best_score = None
            for time, score in best_score_list:
                if time <= i:
                    best_score = score
                else:
                    break
            # 시간을 다시 분으로 바꾸기
            i = i // 60000
            best_score_by_grid.append((i, best_score))
        best_score_by_grid.pop(1)
        print(best_score_by_grid)
        return best_score_by_grid
    
    def get_dir_best_scores(self):
        # print(self.logs_list)
        # print(self.logs_path_list)
        # print(self.dir_name_list)
        for i, logs in tqdm(enumerate(self.logs_list)):
            print('getting best scores...')
            # print('logs:', logs)
            best_score_list = self.get_best_scores_by_grid(logs, self.logs_path_list[i])
            self.dir_best_scores[self.dir_name_list[i]] = best_score_list
            #print(self.dir_best_scores)
        return self.dir_best_scores
    
    def filter_min_time(self):
        # 시간이 가장 짧은 그래프의 길이에 맞춰서 다른 그래프의 길이를 맞추기
        max_len = max([len(best_score_list) for best_score_list in self.dir_best_scores.values()])
        for dir_name, best_score_list in tqdm(self.dir_best_scores.items()):
            print('filtering max time...')
            if len(best_score_list) < max_len:
                self.dir_best_scores[dir_name] = best_score_list + [best_score_list[-1]] * (max_len - len(best_score_list))
            else:   
                self.dir_best_scores[dir_name] = best_score_list[:max_len]
            
        return self.dir_best_scores
    
    def export_graph_table(self):
        # dir_best_scores에 저장된 best score를 그래프로 그리고 표를 저장하기
        """
        표 예시
        |       | dir1 | dir2 | dir3 |
        |-------|------|------|------|
        | time1 |  100 |  200 |  300 |
        | time2 |  200 |  300 |  400 |
        """
        print('exporting graph and table...')
        
        self.get_dir_best_scores()
        self.filter_min_time()
        # print(self.dir_best_scores)
        plt.figure(figsize=(20, 10))
        # 여백 최대한 없애기
        plt.tight_layout()
        # y축 값들 log scale로 만들기
        plt.yscale('log')
        # pandas 표 만들기
        table = pd.DataFrame()
        # 디렉토리 이름을 generator와 solver로 나누기 ex) RL_great -> RL, great
        # generator와 solver 리스트 만들기
        generator_list = []
        solver_list = []
        for dir_name in self.dir_best_scores.keys():
            # generator, solver = dir_name.split('_')
            generator = dir_name
            # solver = dir_name
            generator_list.append(generator)
            # solver_list.append(solver)
            
        # generator와 solver 중복 제거
        generator_list = list(set(generator_list))
        solver_list = list(set(solver_list))
        
        solver_line_style = {}
        
        # generator : RL->'-', 1F1P->'--', KBRA->'-.', else->':'
        generator_color = {}
        for i, generator in enumerate(generator_list):
            if generator == 'RL':
                generator_color[generator] = 'r'
                solver_line_style[generator] = '-'
            elif generator == 'OptaPlanner':
                generator_color[generator] = 'g'
                solver_line_style[generator] = '--'
            elif generator == 'KBRA':
                generator_color[generator] = 'b'
                solver_line_style[generator] = '--'
            else:
                generator_color[generator] = 'b'
            
        print(generator_color)
        print(solver_line_style)
        
        # solver : GD ->'r', HC->'g', Tabu->'b'
        # solver_line_style = {}
        # for i, solver in enumerate(solver_list):
        #     if solver == 'GreatDeluge':
        #         generator_color[solver] = 'r'
        #         solver_line_style[solver] = '-'
        #     elif solver == 'HillClimbing':
        #         generator_color[solver] = 'g'
        #         solver_line_style[solver] = '--'
        #     else:
        #         generator_color[solver] = 'b'
        #         solver_line_style[solver] = '--'
        
        # 그래프 그리기
        for i, (dir_name, best_score_list) in enumerate(self.dir_best_scores.items()):
            print(best_score_list)
            # generator, solver = dir_name.split('_')
            generator = dir_name
            # generator = dir_name
            print(generator_color)
            print(solver_line_style)
            line_style = solver_line_style[generator]
            color = generator_color[generator]
            if dir_name == 'RL':
                label = 'With RL Great Deluge'
            else:
                label = 'With OPIS Great Deluge'
            plt.plot([score for time, score in best_score_list], label=label, linestyle=line_style, color=color)
            table[dir_name] = [score for time, score in best_score_list]
        
        table.index = [time for time, _ in best_score_list]
        table.to_csv(f'{self.filename} RL vs Without RL using Great Deluge Algorithm.csv')
        # 글씨 겹치지 않게 크기 조절
        plt.xticks(range(len(best_score_list)), [file for file, _ in best_score_list], rotation=60, fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(fontsize=20)
        plt.xlabel('Time(min)', fontsize=20)
        plt.ylabel('Score', fontsize=20)
        #plt.ylabel('Optimization Cost Score', fontsize=25)
        plt.title(f'{self.filename} RL vs Without RL using Great Deluge Algorithm', fontsize=30)
        # # 그래프가 꺾이는 부분(점수가 달라지는 부분)에 점 표시 및 점수, 날짜 출력, 모든 그래프에 적용
        # for dir_name, best_score_list in self.dir_best_scores.items():
        #     for i, (time, score) in enumerate(best_score_list):
        #         if i == 0:
        #             continue
        #         # 점수가 달라지는 부분에 점 표시
        #         if best_score_list[i-1][1] != score:
        #             plt.scatter(i, score, c='red')
        #             score_src = f'{score:,}'
        #             plt.text(i, score, f'{score_src}', rotation=30, verticalalignment='bottom', fontsize=7)
        # plt.show()
        plt.savefig(f'{self.filename}.png')
        
        