# app.py

from flask import Flask
from main import *
import pandas as pd
import numpy as np
from Bio import Align

DATABASEPATH = "../data/SAbDab/sequences.csv"

import os
from Bio.PDB import PDBList

app = Flask(__name__)

@app.route('/')

def hello():
    return 'Hello from Flask!' 

@app.route('/api/generate/<sequence>', methods=['GET'])

def generate_antibodies(sequence):
    print(f'Loading sequence {sequence}')
    resp = main(sequence)
    return resp, {"Access-Control-Allow-Origin": "*"}

@app.route('/api/structure/align/<gen_seq>', methods=['GET'])
def get_id(gen_seq):
    print(f'Received {gen_seq}, looking for ID')
    sequence_database = pd.read_csv(DATABASEPATH, sep=';', header=0)
    scores = []
    ids = []
    for _, sequence in sequence_database.iterrows():
        ids += [sequence['seq_id'].split('|')[0]]
        aligner = Align.PairwiseAligner()
        alignments = aligner.align(sequence['sequence'], gen_seq)
        scores += [ alignments[0].score]
    return ids[np.argmax(scores)], {"Access-Control-Allow-Origin": "*"}

@app.route('/api/structure/retrieve/<id>', methods=['GET'])
def get_pdb(id):
    print(f'Dowloading {id}\'s PDB file')
    pdb_list = PDBList()
    pdb_path = './intelligent-antibodies-view/src/components/_static'
    pdb_filename = pdb_list.retrieve_pdb_file(id, pdir= pdb_path, file_format='pdb')
    if os.path.exists(pdb_filename):
        pdb_file = f'{pdb_path}/{id}.pdb'
        os.rename(pdb_filename, pdb_file)
    return pdb_filename, {"Access-Control-Allow-Origin": "*"}