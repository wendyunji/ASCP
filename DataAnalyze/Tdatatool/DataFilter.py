import pandas as pd
from itertools import combinations
from tqdm import tqdm
import time
import os

def find_combinations(fleet_data, target_sum):
    valid_combinations = []
    for r in range(1, len(fleet_data) + 1):
        for combo in combinations(fleet_data, r):
            if sum(item[1] for item in combo) == target_sum:
                valid_combinations.append(combo)
    return valid_combinations

def process_flight_data(file_path, target_sum):
    start_time = time.time()
    data = pd.read_csv(file_path, header=None, sep='|')
    data.columns = [
        'Departure_Station', 'Arrival_Station', 'Fleet_Type', 'Aircraft_Number',
        'Departure_DateTime', 'Arrival_DateTime', 'Dep_Timezone_Diff', 'Arr_Timezone_Diff'
    ]
    load_time = time.time() - start_time
    print(f"Data loading time: {load_time / 60:.2f} minutes")

    fleet_data = data[data['Fleet_Type'].between(3000, 3999)]
    fleet_data = fleet_data.groupby('Fleet_Type').size().reset_index(name='Count')
    fleet_data = list(fleet_data.itertuples(index=False, name=None))
    
    print("Finding valid combinations...")
    start_time = time.time()
    valid_combinations = find_combinations(fleet_data, target_sum)
    combination_time = time.time() - start_time
    print(f"Combination finding time: {combination_time / 60:.2f} minutes")
    print(f"Number of valid combinations found: {len(valid_combinations)}")
    
    results = []
    for combo_index, combo in enumerate(tqdm(valid_combinations, desc="Processing combinations"), 1):
        start_time = time.time()
        fleet_types = [item[0] for item in combo]
        filtered_data = data[data['Fleet_Type'].isin(fleet_types)].copy()
        
        filtered_data.loc[:, 'Departure_DateTime'] = pd.to_datetime(filtered_data['Departure_DateTime']) - pd.to_timedelta(filtered_data['Dep_Timezone_Diff'], unit='m')
        filtered_data.loc[:, 'Arrival_DateTime'] = pd.to_datetime(filtered_data['Arrival_DateTime']) - pd.to_timedelta(filtered_data['Arr_Timezone_Diff'], unit='m')
        
        output_data = filtered_data[['Departure_Station', 'Departure_DateTime', 'Arrival_Station', 'Arrival_DateTime', 'Fleet_Type']]
        output_data.columns = ['ORIGIN', 'ORIGIN_DATE', 'DEST', 'DEST_DATE', 'AIRCRAFT_TYPE']
        results.append((combo, output_data))
        
        save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Dataset/Tdataset/format'))
        os.makedirs(save_dir, exist_ok=True)
        output_file_path = os.path.join(save_dir, f'filtered_flights_{month}_{"_".join(map(str, fleet_types))}.csv')
        output_data.to_csv(output_file_path, index=False)
        processing_time = time.time() - start_time
        print(f"Processed combination {combo_index} in {processing_time / 60:.2f} minutes")

    return results

month = input("Enter month (e.g., 01 for January): ")
target_sum = int(input("Enter target sum: "))

file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f'../../Dataset/Tdataset/raw/tt2014{month}.legs'))

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

filtered_flight_data = process_flight_data(file_path, target_sum)

print(f"Total number of combinations processed: {len(filtered_flight_data)}")
for combo, df in filtered_flight_data:
    combo_str = ', '.join([f"({item[0]}, {item[1]})" for item in combo])
    print(f"Combination used: {combo_str}")
