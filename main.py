### Main python script for data generation
# TODO: relate to AI algorithm
import sys
import pandas as pd
import json

from .model import generate_interacting_antibody

def main(sequence):
  for seq in generate_interacting_antibody(sequence, 10, 1):
    data = pd.DataFrame(dict(seq=[seq], score=[0]))
    return data.to_json()

  
if __name__ == "__main__":
  print(main("RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCS"))