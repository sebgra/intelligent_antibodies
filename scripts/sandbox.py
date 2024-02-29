import os
import sys

from Bio import SeqIO

# TODO: Add all the prefixes for the antibodies
ab_prefixes = ["IMMUNOGLOBULIN"]

fasta_sequences = SeqIO.parse(open("data/SAbDab/fasta/all_samples/1oaz.fasta"),'fasta')
for fasta in fasta_sequences:
    print(fasta)
    description, name, sequence = fasta.description, fasta.id, str(fasta.seq)
    print(name, sequence)


name, chains, molecule, specie = description.split('|')
print(f"Name: {name}")
print(f"Chains: {chains}")
print(f"Molecule: {molecule}")
print(f"Specie: {specie}")

for prefix in ab_prefixes:
    if prefix in molecule:
        print(f"Found Ab in {name}")
        break
