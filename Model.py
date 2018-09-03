# -*- coding: utf-8 -*-
"""
Created on Wed Jun 06 23:12:52 2018

@author: moseli
"""

from numpy.random import seed
seed(1)

from sklearn.model_selection import train_test_split as tts
import logging
import numpy as np


import keras
from keras import backend as k
k.set_learning_phase(1)
from keras.preprocessing.text import Tokenizer
from keras import initializers
from keras.optimizers import RMSprop
from keras.models import Sequential,Model
from keras.layers import Dense,LSTM,Dropout,Input,Activation,Add,Concatenate
from keras.layers.advanced_activations import LeakyReLU,PReLU
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)
