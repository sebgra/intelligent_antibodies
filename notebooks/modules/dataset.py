import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import encoding

from sklearn.preprocessing import OrdinalEncoder

class DatasetFactory:

    def __init__(self, encoder):
        self.encoder = encoder

    def get_data(self):
        df_seq = pd.read_csv("../data/SAbDab/sequences.csv", sep=";")
        df_seq[["seq_rcpb", "seq_type"]] = df_seq["seq_id"].str.split('|',  n=1, expand=True)
        ordinal_encoder = OrdinalEncoder()
        enc_seq_type = ordinal_encoder.fit_transform(df_seq[["seq_type"]])
        seq = df_seq["sequence"]
        seq_encoded = self.encoder.encode(seq)
        X_data = seq_encoded
        y_data = enc_seq_type
        return X_data, y_data, X_data.shape[0], X_data.shape[1]
    
class OneHotProtDataset:
    def get_data(vector_size=1500):
        # df_match = pd.read_csv("../data/SAbDab/data.csv", sep=";")
        df_seq = pd.read_csv("../data/SAbDab/sequences.csv", sep=";")
        df_seq[["seq_rcpb", "seq_type"]] = df_seq["seq_id"].str.split('|',  n=1, expand=True)
        ordinal_encoder = OrdinalEncoder()
        enc_seq_type = ordinal_encoder.fit_transform(df_seq[["seq_type"]])
        seq = df_seq["sequence"]
        encoder = encoding.ProteinOneHotEncoder(encoding.AMINO_ACID_ALPHABET)
        seq_encoded = encoder.encode(seq, vector_size=vector_size)
        X_data = seq_encoded
        y_data = enc_seq_type
        alphabet_size = len(encoding.AMINO_ACID_ALPHABET)
        return X_data, y_data, vector_size, alphabet_size