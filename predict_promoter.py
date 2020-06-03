import sys
import os

sys.path.insert(0, '/Users/mmfp/opt/anaconda3/lib/python3.7/site-packages/')
sys.path.insert(0, '/usr/local/lib/python3.7/site-packages')
cdir = os.path.abspath(__file__)
sys.path.insert(0, cdir)

from fastai import *
from fastai.text import *
from Bio import Seq
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import FeatureLocation, CompoundLocation
import networkx as nx
import tokenizer_utils as tokutils
import transformer_utils as tutils
import tensorflow as tf

import pandas as pd
import numpy as np


path = Path('/Users/mmfp/Desktop/genomic_data/e_coli/')
NUM_LAYERS = 4
D_MODEL = 128
BATCH_SIZE=200
DFF = 512
NUM_HEADS = 8
DROPOUT_RATE = 0.1
EPOCHS = 20


classification_df = pd.read_csv(path/'e_coli_promoters_dataset.csv')
train_df = classification_df[classification_df.set == 'train'].sample(frac=1)
valid_df = classification_df[classification_df.set == 'valid'].sample(frac=1)
test_df = classification_df[classification_df.set == 'test'].sample(frac=1)
# I will provisionally use fastai tokenizer and libraries, but the next part can be optimesed quite easily
tok = tokutils.Tokenizer(tokutils.GenomicTokenizer, n_cpus=1, pre_rules=[], post_rules=[], special_cases=['xxpad'])
data_clas = tokutils.GenomicTextClasDataBunch.from_df(path, train_df, valid_df, test_df=test_df, tokenizer=tok, 
                                            text_cols='Sequence', label_cols='Promoter', bs=300)


def create_batch(ds, target):
    """
    @ comment: returns a list of batches with input and target values
    """
    ds = np.array([d[0].data for d in ds])
    _data = np.array([(ds[i], tar) for i, tar in enumerate(target)])
    for i in range(0, len(_data), BATCH_SIZE):
        yield _data[i:i+BATCH_SIZE]


train_data_batch = create_batch(data_clas.train_ds, train_df['Promoter'].tolist())
validation_data_batch = create_batch(data_clas.valid_ds, valid_df['Promoter'].tolist())
test_data_batch = create_batch(data_clas.test_ds, test_df['Promoter'].tolist())
    

learning_rate = tutils.CustomSchedule(D_MODEL)
optimizer = tf.keras.optimizers.Adam(learning_rate,
                                     beta_1=0.9, 
                                     beta_2=0.98, 
                                     epsilon=1e-9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                                    from_logits=True, 
                                    reduction='none')


def loss_function(real, pred):
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  
  return tf.reduce_mean(loss_)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                                    name='train_accuracy')


transformer = tutils.Encoder(NUM_LAYERS, 
                            D_MODEL, 
                            NUM_HEADS, DFF,
                            rate=DROPOUT_RATE)

checkpoint_path = os.path.join(cdir,"./checkpoints")

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

def train_step(inp, tar):
    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar, True)
        loss = loss_function(tar, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    
    train_loss(loss)
    train_accuracy(tar, predictions)   

for epoch in range(EPOCHS):
    start = time.time()
    
    train_loss.reset_states()
    train_accuracy.reset_states()
    
    # inp -> portuguese, tar -> english
    for (batch, (inp, tar)) in enumerate(train_data_batch):
        train_step(inp, tar)
        
        if batch % 50 == 0:
            print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))
        
    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                            ckpt_save_path))
        
    print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                    train_loss.result(), 
                                                    train_accuracy.result()))

    print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))                                         
