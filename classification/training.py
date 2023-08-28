from models.AST_bird import ast 
from models.resnet_bird import resnet
from models.hubert_bird import hubert

from models.AST import ast_possum
from models.resnet import resnet_possum
from models.hubert import hubert_possum

RMS = ['rms0', 'rms2', 'rms5']
STATE = ['denoised', 'original']

def train():
    for state in STATE:
        print(f'State: {state}')
        for rms in RMS:
            print(f'RMS: {rms}')
            resnet_possum(rms,state)
            hubert_possum(rms,state)
            ast_possum(rms,state)

train()