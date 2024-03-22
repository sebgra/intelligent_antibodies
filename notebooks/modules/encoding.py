"""
ref. Fidle
"""
import numpy as np
import itertools

AMINO_ACID_ALPHABET = "ARNDCQEGHILKMFPSTWYVXU"

def alphabet_one_hot(alphabet=AMINO_ACID_ALPHABET):
    base = np.zeros((len(alphabet)))
    encoding = {}
    for index, letter in enumerate(alphabet):
        code = base.copy()
        code[index] = 1
        encoding[letter] = code
    return encoding

def one_hot_encode_sequence(sequence: str, encoder: dict[str, np.ndarray], vector_size: int) -> np.ndarray:
    encoded = np.zeros((vector_size, len(encoder)))
    if len(sequence) > vector_size:
        sequence = sequence[:vector_size]
    for i, letter in enumerate(sequence):
        encoded[i] = encoder[letter]
    return encoded

def one_hot_encoder(x: str, encoder: dict[str, np.ndarray], vector_size: int=1500):
    x_encode = []
    for sequence in x:
        x_encode.append(one_hot_encode_sequence(sequence, encoder, vector_size))
    return np.array(x_encode)

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
    for i in range(len(sequence)):
        kmers.append(sequence[i:i+k+1])
    for subseq in kmers:
        encoder[subseq] += 1
    frequency = []
    for key in encoder.key():
        frequency.append(encoder[key])
    return frequency