""""ref. https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c"""

import numpy as np
import pandas as pd
from keras.utils import Sequence

class SequenceEncodingDataGenerator(Sequence):
    """Generate data for antibody and antigen encoded sequence"""

    def __init__(self, x: pd.DataFrame, labels: pd.Series, seq_id_to_encoded_sequences: dict[str, np.ndarray], batch_size=32):
        self.seq_ids = list(seq_id_to_encoded_sequences.keys())
        self.x_df = x
        self.labels = labels 
        self.seq_id_to_encoded_sequences = seq_id_to_encoded_sequences

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        ag_seq_id = self.x_df.iloc[index]['ag']
        ab_seq_id = self.x_df.iloc[index]['ab']
        x_ag = self.seq_id_to_encoded_sequences[ag_seq_id]
        x_ab = self.seq_id_to_encoded_sequences[ab_seq_id]
        y = self.labels.iloc[index]['interaction']
        return ([x_ag, x_ab], y)