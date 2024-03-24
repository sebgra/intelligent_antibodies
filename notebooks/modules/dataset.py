import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import encoding

from sklearn.preprocessing import OrdinalEncoder

class EquilibratedDataset:
    def __init__(self, encoder):
        self.encoder = encoder

    def getdata(self, vector_size):
        df_seq = pd.read_csv("../data/SAbDab/sequences.csv", sep=";")
        df_seq[["seq_rcpb", "seq_type"]] = df_seq["seq_id"].str.split('|',  n=1, expand=True)
        df_seq_ab = df_seq[df_seq["seq_type"] == "ab"]
        df_seq_ag = df_seq[df_seq["seq_type"] == "ag"]
        # n_ab = df_seq_ab.shape[0]
        # n_ag = df_seq_ag.shape[0]
        # print(n_ab, n_ag)
        # df_seq_ag_selected = df_seq_ag.sample(n_ab, replace=False)
        df_match = pd.read_csv("../data/SAbDab/data_filtered.csv", sep=";")
        # df_seq_equilibrated = pd.concat([df_seq_ab, df_seq_ag_selected])
        # df_filtered_equilibrated = df_match[df_match["ab"].isin(df_seq_equilibrated["seq_id"]) & df_match["ag"].isin(df_seq_equilibrated["seq_id"])]
        data = df_match
        data = data.merge(df_seq, left_on="ab", right_on="seq_id", how="inner")
        data.rename(columns={"sequence": "sequence_ab"}, inplace=True)
        data = data.merge(df_seq, left_on="ag", right_on="seq_id", how="inner")
        data.rename(columns={"sequence": "sequence_ag"}, inplace=True)
        data = data[["sequence_ab", "sequence_ag", "interaction"]]
        # Equilibrate interaction vs not interaction
        data_interaction = data[data["interaction"] == 1]
        data_non_interaction = data[data["interaction"] == 0]
        n_interaction = data_interaction.shape[0]
        # n_non_interaction = data_non_interaction.shape[0]
        data_non_interaction_selected = data_non_interaction.sample(n_interaction, replace=False)
        data_selected = pd.concat([data_interaction, data_non_interaction_selected])
        x_data_ag = self.encoder.encode(data_selected["sequence_ag"], vector_size)
        x_data_ab = self.encoder.encode(data_selected["sequence_ab"], vector_size)
        x_data_ab = np.array(x_data_ab, dtype=np.float32)
        x_data_ag = np.array(x_data_ag, dtype=np.float32)
        X_data = [x_data_ab, x_data_ag]
        y_data = data_selected["interaction"]
        y_data = np.array(y_data, dtype=np.float32)
        alphabet_size = x_data_ag[0].shape[1]
        return X_data, y_data, vector_size, alphabet_size

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
    

class AntibodyOneHotProtDataset:
    def get_data(vector_size):
        df_seq = pd.read_csv("../data/SAbDab/sequences.csv", sep=";")
        df_seq[["seq_rcpb", "seq_type"]] = df_seq["seq_id"].str.split('|',  n=1, expand=True)
        df_seq = df_seq[df_seq["seq_type"] == "ab"] 
        ordinal_encoder = OrdinalEncoder()
        enc_seq_type = ordinal_encoder.fit_transform(df_seq[["seq_type"]])
        seq = df_seq["sequence"]
        encoder = encoding.ProteinOneHotEncoder(encoding.AMINO_ACID_ALPHABET)
        seq_encoded = encoder.encode(seq, vector_size)
        X_data = seq_encoded
        y_data = enc_seq_type
        alphabet_size = len(encoding.AMINO_ACID_ALPHABET)
        return X_data, y_data, vector_size, alphabet_size


class AntibodyNLFProtDataset:
    def get_data(vector_size):
        df_seq = pd.read_csv("../data/SAbDab/sequences.csv", sep=";")
        df_seq[["seq_rcpb", "seq_type"]] = df_seq["seq_id"].str.split('|',  n=1, expand=True)
        df_seq = df_seq[df_seq["seq_type"] == "ab"] 
        ordinal_encoder = OrdinalEncoder()
        enc_seq_type = ordinal_encoder.fit_transform(df_seq[["seq_type"]])
        seq = df_seq["sequence"]
        encoder = encoding.NLFEncoder()
        seq_encoded = encoder.encode(seq, vector_size)
        X_data = seq_encoded
        y_data = enc_seq_type
        alphabet_size = len(encoding.AMINO_ACID_ALPHABET)
        return X_data, y_data, vector_size, encoder.nlf.shape[0]

