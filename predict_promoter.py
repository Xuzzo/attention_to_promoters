import sys
import os

sys.path.insert(0,'/Users/mmfp/opt/anaconda3/lib/python3.7/site-packages/')
sys.path.insert(0,'/usr/local/lib/python3.7/site-packages')

from fastai import *
from fastai.text import *
from Bio import Seq
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import FeatureLocation, CompoundLocation
import networkx as nx

import pandas as pd


path = Path('/Users/mmfp/Desktop/genomic_data/e_coli/')

classification_df = pd.read_csv(path/'e_coli_promoters_dataset.csv')
train_df = classification_df[classification_df.set == 'train']
valid_df = classification_df[classification_df.set == 'valid']
test_df = classification_df[classification_df.set == 'test']
