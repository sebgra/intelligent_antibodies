"""
ref. Fidle
"""
import numpy as np
import pandas as pd
import itertools

AMINO_ACID_ALPHABET = "ARNDCQEGHILKMFPSTWYVXU"

class ProteinOneHotEncoder:

    def __init__(self, alphabet=AMINO_ACID_ALPHABET):
        self.alphabet = alphabet
        self.encoder, self.decoder = self.alphabet_one_hot()

    def alphabet_one_hot(self):
        base = np.zeros((len(self.alphabet)))
        encoder = {}
        decoder = {}
        for index, letter in enumerate(self.alphabet):
            code = base.copy()
            code[index] = 1
            encoder[letter] = code
            decoder[code] = letter
        return encoder, decoder

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
    
    def decode(self, x_encode: np.ndarray):
        seq: list[str] = [''] * x_encode.shape[0]
        for i, code in enumerate(x_encode):
            seq[i] = self.decoder[code]
        return seq
    

class NLFEncoder:
    def __init__(self, ):
        self.nlf = pd.read_csv('../data/NLF.csv', index_col=0)
        # The letter X will represent an empty position (added in case we might encounter empty spots somewhere)
        self.nlf['X'] = [0.0] * self.nlf.shape[0]
        # If we encounter Selenocysteine, we'll treat it as Cysteine
        self.nlf['U'] = self.nlf['C']
        self.decoder = {
            self.nlf[letter]: letter for letter in self.nlf.columns
        }
    
    def encode_sequence(self, sequence: str, vector_size: int=1500) -> np.ndarray:
        sequence = sequence[:vector_size]
        seq_tensor = np.zeros((vector_size, self.nlf.shape[0]), dtype=np.float16)
        for i, letter in enumerate(sequence):
            seq_tensor[i] = self.nlf[letter]
        return seq_tensor

    def encode(self, x):
        tensors = []
        for sequence in x:
            tensors.append(self.encode_sequence(sequence))
        return np.array(tensors)
    
    def decode(self, x_encode: np.ndarray):
        seq: list[str] = [''] * x_encode.shape[0]
        for i, code in enumerate(x_encode):
            seq[i] = self.decoder[code]
        return seq

class BLOSUMEncoder:
    def __init__(self):
        blosum = pd.read_csv('../data/BLOSUM62.csv', index_col=0)
        # If we encounter Selenocysteine, we'll treat it as Cysteine
        blosum['U'] = blosum['C']
        # # We will not encounter B, Z or * so we can delete the corresponding vectors
        # blosum.drop(labels=['B', 'Z', 'X', '*'], axis=0, inplace=True)
        # blosum.drop(labels=['B', 'Z', 'X', '*'], axis=1, inplace=True)
        self.decoder = {
            self.blosum[letter]: letter for letter in self.blosum.columns
        }

    def encode_sequence(self, sequence: str, vector_size: int=1500) -> np.ndarray:
        sequence = sequence[:vector_size]
        seq_tensor = np.zeros((vector_size, self.blosum.shape[0]), dtype=np.float16)
        for i, letter in enumerate(sequence):
            seq_tensor[i] = self.blosum[letter]
        return seq_tensor
    
    def encode(self, x):
        tensors = []
        for sequence in x:
            tensors.append(self.encode_sequence(sequence))
        return np.array(tensors)

    def decode(self, x_encode: np.ndarray):
        seq: list[str] = [''] * x_encode.shape[0]
        for i, code in enumerate(x_encode):
            seq[i] = self.decoder[code]
        return seq


"""
Kmers Frequency
"""

def alphabet_kmer(alphabet=AMINO_ACID_ALPHABET, k: int=3):
    kmers_order = itertools.product(*itertools.repeat(alphabet, k))
    encoding = {}
    for kmer in kmers_order:
        encoding[kmer] = 0
    return encoding

def kmer_frequency_encode(sequence: str, encoder: dict[str, int], k: int=3):
    kmers = []
    for i in range(len(sequence)-k):
        kmers.append(sequence[i:i+k+1])
    for subseq in kmers:
        encoder[subseq] += 1
    frequency = []
    for key in encoder.key():
        frequency.append(encoder[key])
    return frequency
