import pandas as pd
from itertools import permutations

import pandas as pd
from itertools import permutations
import glob
import os

# Specify the month and use a wildcard for the suffix
month = input("Enter the month (e.g., 06): ")
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f'format/filtered_flights_{month}_*.csv'))

# Load all matching CSV files
all_files = glob.glob(file_path)
df_list = [pd.read_csv(file) for file in all_files]
df = pd.concat(df_list, ignore_index=True)

# Extract unique airports from the relevant columns
airports = pd.concat([df['ORIGIN'], df['DEST']]).unique()

# Generate all combinations of the airports
combinations = list(permutations(airports, 2))

# Create a DataFrame from the combinations
combinations_df = pd.DataFrame(combinations, columns=['Departure', 'Arrival'])

# Add Deadhead column (0 if same airport, 1 if different)
combinations_df['Deadhead'] = (combinations_df['Departure'] != combinations_df['Arrival']).astype(int)

save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'deadhead'))
os.makedirs(save_dir, exist_ok=True)
output_file_path = os.path.join(save_dir, f'deadhead(airport_combinations)_{month}.csv')
combinations_df.to_csv(output_file_path, index=False)


print(f"Combinations saved to {output_file_path}")

