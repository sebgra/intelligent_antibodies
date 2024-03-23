"""
ref. Fidle
"""
import numpy as np
import pandas as pd

AMINO_ACID_ALPHABET = "ARNDCQEGHILKMFPSTWYVXU"

class ProteinOneHotEncoder:

    def __init__(self, alphabet=AMINO_ACID_ALPHABET):
        self.alphabet = alphabet
        self.encoder = self.alphabet_one_hot()

    def alphabet_one_hot(self):
        base = np.zeros((len(self.alphabet)))
        encoding = {}
        for index, letter in enumerate(self.alphabet):
            code = base.copy()
            code[index] = 1
            encoding[letter] = code
        return encoding

    def one_hot_encode_sequence(self, sequence: str, vector_size: int) -> np.ndarray:
        encoded = np.zeros((vector_size, len(self.encoder)))
        if len(sequence) > vector_size:
            sequence = sequence[:vector_size]
        for i, letter in enumerate(sequence):
            encoded[i] = self.encoder[letter]
        return encoded

    def encode(self, x: str, vector_size: int=1500):
        x_encode = []
        for sequence in x:
            x_encode.append(self.one_hot_encode_sequence(sequence, vector_size))
        return np.array(x_encode)

class NLFEncoder:
    def __init__(self, ):
        self.nlf = pd.read_csv('../data/NFL.csv', index_col=0)
        # The letter X will represent an empty position (added in case we might encounter empty spots somewhere)
        self.nlf['X'] = [0.0] * self.nlf.shape[0]
        # If we encounter Selenocysteine, we'll treat it as Cysteine
        self.nlf['U'] = self.nlf['C']
    
    def encode(self, sequence: str, vector_size: int=1500) -> np.ndarray:
        sequence = sequence[:vector_size]
        seq_tensor = np.zeros((vector_size, self.nlf.shape[0]), dtype=np.float16)
        for i, letter in enumerate(sequence):
            seq_tensor[i] = self.nlf[letter]
        return seq_tensor

    def call(self, x):
        tensors = []
        for sequence in x:
            tensors.append(self.encode(sequence))
        return np.array(tensors)