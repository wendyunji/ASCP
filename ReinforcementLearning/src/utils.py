from Components import Airport, Aircraft, Hotel
import copy
#[t_ori, t_des, dur, a_ori, a_des, type]
#[  0      1     2     3       4    5  ]

#Aircraft['[0,0,1]'] = [crewnum, layover, quickturn]

def checkConnection(V_p, V_f, homeindex):
    
    if V_p == [0,0,0,[0],[0],[0]] : 
        return True
    if V_p[3][homeindex] != 1:
        return False
    
    flight_gap = V_f[0] - V_p[1]
    
    if V_p[4] == V_p[3]:  # 완성된 페어링
        return False 
    if V_f[1] - V_p[0] > 12 * 60: # 페어링 길이 제약 -> 12시간으로 수정
        return False  
    if flight_gap < 30: # 시간의 선후관계 제약 + 30 제약
        return False  
    if V_p[4] != V_f[3]:  # 공간 제약
        return False 
    if V_p[5] != V_f[5]: # 항공기 기종 제약
        return False  
    if V_p[2] + V_f[2] > 4 * 60: 
        return False
    if flight_gap >= 4* 60: # 최대 연결 시간 제약
        return False

    return True


def update_state(V_p_list, V_f, idx) :
    V_p = copy.deepcopy(V_p_list[idx])
    V_f_tmp = copy.deepcopy(V_f)
    
    if V_p == [0,0,0,[0],[0],[0]] :
        V_p = V_f_tmp
    
    else :
        V_p[2] += V_f_tmp[2]    # 휴식시간이 없었다면 V_p_dur에 gap과 c_dur을 더함
        
        V_p[1] = V_f_tmp[1] # 도착시간
        V_p[4] = V_f_tmp[4] # 도착공항
    
    V_p_list[idx] = V_p


def get_reward(V_p_list, V_f, idx) :
    V_p = V_p_list[idx]

    if V_p[3] == V_f[4] :
        reward = 1
    else : reward = 0
        
    return reward