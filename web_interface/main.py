### Main python script for data generation
# TODO: relate to AI algorithm
import sys
import pandas as pd

def main(sequence):
  print(sequence)
  data = pd.read_csv("./fake_data/data.csv", sep=',')
  return data.to_json()