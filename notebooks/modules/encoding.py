"""
ref. Fidle
"""
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import tensorflow as tf
import itertools

AMINO_ACID_ALPHABET = "ARNDCQEGHILKMFPSTWYVXU"

class ProteinOneHotEncoder:

    def __init__(self, alphabet: str = AMINO_ACID_ALPHABET):
        self.alphabet = alphabet
        self.encoder, self.decoder = self.alphabet_one_hot()

    def alphabet_one_hot(self)-> Tuple[Dict[str, np.array], Dict[Tuple, str]]:
        """
        Create one-hot encoder/decoder database. 

        Returns
        -------
        Tuple[Dict[str, np.array], Dict[Tuple, str]]
            Encoder and decoder dictionaries containing either amino acid or one hot encoding vector.
        """        
        base: np.array = np.zeros((len(self.alphabet)))
        encoder: dict = {}
        decoder: dict = {}
        for index, letter in enumerate(self.alphabet):
            code: np.array = base.copy()
            code[index] = 1
            encoder[letter] = code
            decoder[tuple(code)] = letter
            
        return encoder, decoder

    def one_hot_encode_sequence(self, sequence: str, vector_size: int) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        sequence : str
            _description_
        vector_size : int
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """        
        encoded: tf.Tensor = np.zeros((vector_size, len(self.encoder)))
        if len(sequence) > vector_size:
            sequence = sequence[:vector_size]
        for i, letter in enumerate(sequence):
            encoded[i] = self.encoder[letter]
        return encoded

    def encode(self, x: str, vector_size: int = 1500) -> np.array:
        """
        Encode protein sequence from amino acids to one hot 2D vector.

        Parameters
        ----------
        x : str
            _description_
        vector_size : int, optional
            _description_, by default 1500

        Returns
        -------
        np.array
            _description_
        """        
        x_encode = []
        for sequence in x:
            x_encode.append(self.one_hot_encode_sequence(sequence, vector_size))
        return np.array(x_encode)
    
    def decode(self, x_encode: np.ndarray) -> List[str]:
        """
        Decode protein sequence from one hot encoding seq to string sequence with amino acid aphabet.

        Parameters
        ----------
        x_encode : np.ndarray
            One hot encoded sequence to decode.
        Returns
        -------
        Dict[str]
            Decoded sequence as amino acids.
        """        
        seq: list[str] = [''] * x_encode.shape[0]
        for i, code in enumerate(x_encode):
            max_index: int = np.argmax(code)
            seq[i] = self.alphabet[max_index]
        return seq

    

class NLFEncoder:
    def __init__(self, ):
        self.nlf = pd.read_csv('../data/NLF.csv', index_col=0)
        # The letter X will represent an empty position (added in case we might encounter empty spots somewhere)
        self.nlf['X'] = [0.0] * self.nlf.shape[0]
        # If we encounter Selenocysteine, we'll treat it as Cysteine
        self.nlf['U'] = self.nlf['C']
        self.decoder = {
            tuple(self.nlf[letter]): letter for letter in self.nlf.columns
        }
    
    def encode_sequence(self, sequence: str|List[str], vector_size: int ) -> np.ndarray:
        """
        Encode protein sequence as amino acid alphabet to NLF encoded protein matrix.

        Parameters
        ----------
        sequence : str|List[str]
            Sequence f amino acids to encode.
        vector_size : int
            Size of the potentially truncaded protein to encode.

        Returns
        -------
        np.ndarray
            NLF encoded protein sequence
        """        
        sequence = sequence[:vector_size]
        seq_tensor = np.zeros((vector_size, self.nlf.shape[0]), dtype=np.float16)
        for i, letter in enumerate(sequence):
            seq_tensor[i] = self.nlf[letter]
        return seq_tensor

    def encode(self, x, vector_size=2000) -> np.array:
        tensors = []
        for sequence in x:
            tensors.append(self.encode_sequence(sequence, vector_size))
        return np.array(tensors)
    
    def decode(self, x_encode: np.ndarray):
        seq: list[str] = [''] * x_encode.shape[0]
        for i, code in enumerate(x_encode):
            seq[i] = self.decoder[tuple(code)]
        return seq

class BLOSUMEncoder:
    def __init__(self):
        self.blosum = pd.read_csv('../data/blosum62.csv', index_col=0)
        # If we encounter Selenocysteine, we'll treat it as Cysteine
        self.blosum['U'] = self.blosum['C']
        # # We will not encounter B, Z or * so we can delete the corresponding vectors
        # blosum.drop(labels=['B', 'Z', 'X', '*'], axis=0, inplace=True)
        # blosum.drop(labels=['B', 'Z', 'X', '*'], axis=1, inplace=True)
        self.decoder = {
            self.blosum[letter]: letter for letter in self.blosum.columns
        }

    def encode_sequence(self, sequence: str, vector_size: int) -> np.ndarray:
        sequence = sequence[:vector_size]
        seq_tensor = np.zeros((vector_size, self.blosum.shape[0]), dtype=np.float16)
        for i, letter in enumerate(sequence):
            seq_tensor[i] = self.blosum[letter]
        return seq_tensor
    
    def encode(self, x, vector_size: int=1500):
        tensors = []
        for sequence in x:
            tensors.append(self.encode_sequence(sequence, vector_size))
        return np.array(tensors)

    def decode(self, x_encode: np.ndarray):
        seq: list[str] = [''] * x_encode.shape[0]
        for i, code in enumerate(x_encode):
            seq[i] = self.decoder[tuple(code)]
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
