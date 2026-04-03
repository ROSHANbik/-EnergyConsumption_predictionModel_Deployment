import numpy as np
import pandas as pd
import os 
import sys 


data_path = os.path.join("data","data.csv")
def data_ingestion():


    df = pd.read_csv(data_path)
    df.columns
    return df