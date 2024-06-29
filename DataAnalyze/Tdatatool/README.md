# Flight Data Filter Tool

This tool processes flight data to find specific combinations of fleet types that match a target sum of aircraft counts.

## Prerequisites

- Python 3.8 or higher
- Required Python packages:
  - `pandas`
  - `tqdm`

You can install the required packages using pip:

```bash
pip install pandas tqdm
```

## Project Structure

```
.
├── DataAnalyze
│   ├── Tdatatool
│   │   └── DataFilter.py  # Main script
│
├── Dataset
│   └── Tdataset
│       ├── raw            # Input data files (e.g., tt201406.legs)
│       └── format         # Output directory for processed files
```

## Usage

1. **Place the Input Data**: Ensure your flight data files (e.g., `tt201406.legs`) are located in the `Dataset/Tdataset/raw` directory.

2. **Run the Script**: Execute the script from the terminal:

   ```bash
   /bin/python3.8 /path/to/DataFilter.py
   ```

   Replace `/path/to/` with the actual path to your `DataFilter.py` script.

3. **Input Parameters**:
   - **Month**: Enter the month in two-digit format (e.g., `06` for June).
   - **Target Sum**: Enter the target sum of aircraft counts you want to find combinations for.

4. **Output**: Processed files will be saved in the `Dataset/Tdataset/format` directory with filenames indicating the fleet types used in the combination.

## Example

```bash
/bin/python3.8 /home/ascp_opta_5/ASCP/DataAnalyze/Tdatatool/DataFilter.py
Enter month (e.g., 01 for January): 06
Enter target sum: 11684
```

This will process the data for June and find combinations that sum to `15656`. The results will be saved in the `format` directory.

## Notes

- Make sure your folder structure is correctly set up as shown above.
- Ensure that the data files are correctly formatted and located in the specified directories.
- Output CSV files will contain the filtered flight data.

## Troubleshooting

- **File Not Found**: Ensure that the input file exists in the `raw` directory and the path is correctly set in the script.
- **Python Environment**: Ensure you're using the correct Python version and that all dependencies are installed.
