import os
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import encoding

from sklearn.preprocessing import OrdinalEncoder

class EquilibratedDataset:
    """
    Loads, filters, and equilibrates a dataset of protein sequences for
    machine learning tasks.

    This class handles the loading of antibody (ab) and antigen (ag)
    sequence data, merges it with interaction data, and balances the
    dataset to have an equal number of interacting and non-interacting pairs.

    Parameters
    ----------
    encoder : object
        An encoder object with `encode` and `encode_sequence` methods
        to convert protein sequences into numerical matrices.
    """

    def __init__(self, encoder: Any):
        self.encoder = encoder
        self.df_seq = pd.read_csv("../data/SAbDab/sequences.csv", sep=";")
        self.df_match = pd.read_csv("../data/SAbDab/data_filtered.csv", sep=";")

    def _prepare_dataframes(self) -> pd.DataFrame:
        """
        Prepares and merges the sequence and interaction dataframes.

        Returns
        -------
        pd.DataFrame
            A merged DataFrame containing antibody and antigen sequences
            and their interaction labels.
        """
        self.df_seq[["seq_rcpb", "seq_type"]] = self.df_seq["seq_id"].str.split('|',  n=1, expand=True)
        
        # Merge interaction data with antibody and antigen sequences
        data = self.df_match.merge(self.df_seq, left_on="ab", right_on="seq_id", how="inner")
        data.rename(columns={"sequence": "sequence_ab"}, inplace=True)
        
        data = data.merge(self.df_seq, left_on="ag", right_on="seq_id", how="inner")
        data.rename(columns={"sequence": "sequence_ag"}, inplace=True)
        
        return data[["sequence_ab", "sequence_ag", "interaction"]]

    def _equilibrate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Balances the dataset by sampling non-interacting pairs to match
        the number of interacting pairs.

        Parameters
        ----------
        data : pd.DataFrame
            The merged DataFrame of sequences and interactions.

        Returns
        -------
        pd.DataFrame
            The equilibrated DataFrame.
        """
        data_interaction = data[data["interaction"] == 1]
        data_non_interaction = data[data["interaction"] == 0]
        n_interaction = data_interaction.shape[0]
        
        data_non_interaction_selected = data_non_interaction.sample(n_interaction, replace=False)
        
        return pd.concat([data_interaction, data_non_interaction_selected])

    def getdata(self, vector_size: int) -> Tuple[List[np.ndarray], np.ndarray, int, int]:
        """
        Loads and prepares the full dataset for a machine learning model.

        This is the main public method that orchestrates the data loading,
        equilibration, and encoding process.

        Parameters
        ----------
        vector_size : int
            The desired length for each protein sequence vector. Sequences
            will be truncated or padded to this size during encoding.

        Returns
        -------
        X_data : list of np.ndarray
            A list containing two NumPy arrays: one for the antibody
            sequences and one for the antigen sequences.
        y_data : np.ndarray
            A NumPy array of interaction labels (0 or 1).
        vector_size : int
            The vector size used for encoding.
        alphabet_size : int
            The size of the encoding alphabet (e.g., number of unique amino acids).
        
        Examples
        --------
        >>> from modules.encoding import ProteinOneHotEncoder
        >>> encoder = ProteinOneHotEncoder()
        >>> dataset = EquilibratedDataset(encoder)
        >>> X, y, vector_size, alphabet_size = dataset.getdata(vector_size=200)
        >>> x_ab, x_ag = X
        """
        data = self._prepare_dataframes()
        data_selected = self._equilibrate_data(data)
        
        x_data_ab = self.encoder.encode(data_selected["sequence_ab"], vector_size)
        x_data_ag = self.encoder.encode(data_selected["sequence_ag"], vector_size)
        
        x_data_ab = np.array(x_data_ab, dtype=np.float32)
        x_data_ag = np.array(x_data_ag, dtype=np.float32)
        
        X_data = [x_data_ab, x_data_ag]
        y_data = np.array(data_selected["interaction"], dtype=np.float32)
        
        # Get alphabet size from the encoded data
        alphabet_size = x_data_ag.shape[2]
        
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

