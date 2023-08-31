from models.AST_bird import ast 
from models.resnet_bird import resnet
from models.hubert_bird import hubert

from models.AST import ast_possum
from models.resnet import resnet_possum
from models.hubert import hubert_possum

import warnings
warnings.filterwarnings("ignore")

RMS = ['extreme']  
STATE = ['denoised']

def train():
    for state in STATE:
        for rms in RMS:
            print('__________________________')
            print(f'State: {state}, RMS: {rms}')
            # resnet_possum(rms,state)
            # hubert_possum(rms,state)
            ast_possum(rms,state)

train()