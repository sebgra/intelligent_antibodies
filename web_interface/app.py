# app.py

from flask import Flask
from main import *
import json

app = Flask(__name__)

@app.route('/')

def hello():
    return 'Hello from Flask!' 

@app.route('/api/generate/<sequence>', methods=['GET'])

def generate_antibodies(sequence):
    print(f'Loading sequence {sequence}')
    resp = main(sequence)
    return resp, {"Access-Control-Allow-Origin": "*"}