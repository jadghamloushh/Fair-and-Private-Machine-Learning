import pandas as pd
import numpy as np

def load_adult_data(file_path):
    """Load and preprocess the Adult dataset."""
    try:
        # First try to read the data without specifying column names
        data = pd.read_csv(file_path, skipinitialspace=True)
        print(f"Successfully loaded file: {file_path}")
        print(f"Columns found: {data.columns.tolist()}")
        print(f"First few rows:\n{data.head(2)}")
        
        # Standardize column names
        data.columns = [col.lower().replace('-', '_') for col in data.columns]
        
    except Exception as e:
        print(f"Error reading file as standard CSV: {e}")
        try:
            # Define column names based on the dataset description
            column_names = [
                'age', 'workclass', 'fnlwgt', 'education', 'education_num', 
                'marital_status', 'occupation', 'relationship', 'race', 'gender',
                'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
            ]
            data = pd.read_csv(file_path, names=column_names, skipinitialspace=True)
            print(f"Successfully loaded file with custom column names: {file_path}")
            print(f"First few rows:\n{data.head(2)}")
        except Exception as e2:
            print(f"Error reading file with custom column names: {e2}")
            print("Trying to load data as space-separated...")
            try:
                # Try to read as space-separated values
                data = pd.read_csv(file_path, sep='\s+', names=column_names)
                print(f"Successfully loaded file as space-separated: {file_path}")
                print(f"First few rows:\n{data.head(2)}")
            except Exception as e3:
                print(f"All loading methods failed: {e3}")
                raise Exception("Could not load the dataset. Please check the file format.")
    
    # Print data types
    print("\nData types:")
    print(data.dtypes)
    
    # Check for any string columns that should be numeric
    for col in ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']:
        if col in data.columns and data[col].dtype == 'object':
            print(f"Column {col} is string type but should be numeric. Sample values: {data[col].head(5).tolist()}")
    
    # Print unique values for categorical columns
    for col in ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'gender', 'native_country', 'income']:
        if col in data.columns:
            print(f"\nUnique values for {col}: {data[col].unique()}")
    
    return data

# Try to load the data with different potential filenames
import os

print("Current working directory:", os.getcwd())
print("Files in directory:", [f for f in os.listdir() if os.path.isfile(f)])

# Try different potential filenames
potential_files = ['adult.data', 'adult.csv', 'adult_data.csv', 'adult-data.csv', 'adult.txt']

data = None
for file in potential_files:
    if os.path.exists(file):
        print(f"\nAttempting to load {file}...")
        try:
            data = load_adult_data(file)
            print(f"Successfully loaded {file}")
            break
        except Exception as e:
            print(f"Failed to load {file}: {e}")

if data is None:
    print("\nCould not find or load any adult dataset files.")
    print("Please provide the correct filename or format the data according to the expected structure.")
else:
    print("\nData loaded successfully!")