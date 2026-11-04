import os
from typing import List, Tuple, Any

import numpy as np
import pandas as pd

from intelligent_antibodies.modules.utils import encoding
from intelligent_antibodies.modules.utils.paths import SABDAB_DIR

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
        self.df_seq = pd.read_csv(SABDAB_DIR / "sequences.csv", sep=";")
        self.df_match = pd.read_csv(SABDAB_DIR / "data_filtered.csv", sep=";")

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

        data = data[["sequence_ab", "sequence_ag", "interaction"]]

        # Both merges above are many-to-many: `seq_id` is one row per CHAIN, so a
        # single PDB id carries up to 9 of them. That turns 345k interaction rows
        # into ~1.17M, of which ~358k are byte-identical duplicates -- and once
        # train_test_split scatters copies of one pair across both sides, the
        # validation score is partly just memorised training rows.
        data = data.drop_duplicates()

        # Chain-level duplication also lets the same (ab, ag) sequence pair arrive
        # labelled both 1 and 0 (~955 pairs). No model can satisfy both, so drop
        # them rather than training against contradictory ground truth.
        conflicting = data.groupby(["sequence_ab", "sequence_ag"])["interaction"].transform("nunique") > 1
        return data[~conflicting]

    def _equilibrate_data(
        self, data: pd.DataFrame, negative_ratio: float = 1.0, random_state: int = 42
    ) -> pd.DataFrame:
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

        # `negative_ratio` negatives per positive. A strict 1:1 balance threw
        # away 99.5% of the available negatives (805,803 of them, against
        # 3,989 positives) -- and with only ~4k pairs left, a classifier of
        # any size overfits long before it learns anything that generalizes.
        # Keeping more negatives is the cheapest available increase in
        # training signal; the caller compensates with class weights so the
        # scores stay calibrated.
        # `random_state` matters more than it looks: without it this draw was
        # different on every call, so the train/test split could never be
        # reproduced. Re-deriving "the held-out set" later then silently mixed
        # in pairs the model had trained on -- which inflated a measured AUC
        # from 0.71 to 0.84 the first time I tried to plot it.
        n_negative = min(int(n_interaction * negative_ratio), len(data_non_interaction))
        data_non_interaction_selected = data_non_interaction.sample(
            n_negative, replace=False, random_state=random_state
        )

        return pd.concat([data_interaction, data_non_interaction_selected])

    def getdata(
        self,
        vector_size: int,
        negative_ratio: float = 1.0,
        return_groups: bool = False,
        random_state: int = 42,
    ) -> Tuple[List[np.ndarray], np.ndarray, int, int]:
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
        >>> from intelligent_antibodies.modules.utils.encoding import ProteinOneHotEncoder
        >>> encoder = ProteinOneHotEncoder()
        >>> dataset = EquilibratedDataset(encoder)
        >>> X, y, vector_size, alphabet_size = dataset.getdata(vector_size=200)
        >>> x_ab, x_ag = X
        """
        data = self._prepare_dataframes()
        data_selected = self._equilibrate_data(
            data, negative_ratio=negative_ratio, random_state=random_state
        )

        x_data_ab = self.encoder.encode(data_selected["sequence_ab"], vector_size)
        x_data_ag = self.encoder.encode(data_selected["sequence_ag"], vector_size)
        
        x_data_ab = np.array(x_data_ab, dtype=np.float32)
        x_data_ag = np.array(x_data_ag, dtype=np.float32)
        
        X_data = [x_data_ab, x_data_ag]
        y_data = np.array(data_selected["interaction"], dtype=np.float32)
        
        # Get alphabet size from the encoded data
        alphabet_size = x_data_ag.shape[2]

        if return_groups:
            # The antibody sequence, for a group-aware train/test split. The
            # same antibody appears in many pairs (1,406 distinct antibodies
            # across 3,989 positives), so a random split puts copies of one
            # antibody on both sides and the test score partly measures
            # memorization of that antibody rather than interaction.
            groups = data_selected["sequence_ab"].to_numpy()
            return X_data, y_data, vector_size, alphabet_size, groups

        return X_data, y_data, vector_size, alphabet_size

class DatasetFactory:
    def __init__(self, encoder):
        self.encoder = encoder

    def get_data(self):
        df_seq = pd.read_csv(SABDAB_DIR / "sequences.csv", sep=";")
        df_seq[["seq_rcpb", "seq_type"]] = df_seq["seq_id"].str.split('|',  n=1, expand=True)
        ordinal_encoder = OrdinalEncoder()
        enc_seq_type = ordinal_encoder.fit_transform(df_seq[["seq_type"]])
        seq = df_seq["sequence"]
        seq_encoded = self.encoder.encode(seq)
        X_data = seq_encoded
        y_data = enc_seq_type
        return X_data, y_data, X_data.shape[0], X_data.shape[1]


class OneHotProtDataset:
    @staticmethod
    def get_data(vector_size=1500):
        df_seq = pd.read_csv(SABDAB_DIR / "sequences.csv", sep=";")
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
    @staticmethod
    def get_data(vector_size):
        df_seq = pd.read_csv(SABDAB_DIR / "sequences.csv", sep=";")
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
    @staticmethod
    def get_data(vector_size):
        df_seq = pd.read_csv(SABDAB_DIR / "sequences.csv", sep=";")
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

