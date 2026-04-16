import os
import pandas as pd
def load_csv_data(file_name: str, header: int = 0, sep: str = ',', decimal: str = '.', skiprow: int = None):
    """
    Load a CSV file into a pandas DataFrame.
    Parameters:
    - file_name: str - The name of the CSV file to load.
    - header: int - The row number to use as the column names.
    Returns:
    - pd.DataFrame - The loaded data as a DataFrame.
    """
    # Path to the folder containing this script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Navigate to data directory
    data_path = os.path.join(BASE_DIR, "..", "data")
    data_path = os.path.abspath(data_path)
    
    df = pd.read_csv(data_path + '\\' + file_name, header=header, sep=sep, decimal=decimal, skiprows= skiprow)
    
    
    return df



# def load_json_data(file_name: str):
#     """
#     Load a JSON file into a pandas DataFrame.
#     Parameters:
#     - file_name: str - The name of the JSON file to load.
#     Returns:
#     - pd.DataFrame - The loaded data as a DataFrame.
#     """
#     # Path to the folder containing this script
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     # Navigate to data directory
#     data_path = os.path.join(BASE_DIR, "..", "data")
#     data_path = os.path.abspath(data_path)
    
#     df = pd.read_json(data_path + '\\' + file_name)
    
#     return df
