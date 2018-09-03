# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 21:32:54 2018

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
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

"""______________________________________________________________________________"""

def loss_function(loss):
    if loss == 'mse':
        LF = 'mse'
        activation = 'relu'
    elif loss == 'log_loss':
        LF = 'categorical_crossentropy'
        activation = 'softmax'
    else:
        LF = 'mse'
        activation = 'linear'

    return LF, activation
    



"""__________________bidirectional encoder(2 layers)-decoder lstm________________"""

def Bi2EnDe(lossFunction):
    print('Bidirectional 2 layer encoder decoder LSTM...')
    
    LF,activation = loss_function(lossFunction)
   
    """__encoder___"""
    encoder_inputs = Input(shape=en_shape)
    
    encoder_LSTM = LSTM(hidden_units,return_sequences=True,return_state=True,go_backwards=True)
    encoder_LSTM2 = LSTM(hidden_units,return_sequences=True,return_state=True)
    
    encoder_outputs,_,_ = encoder_LSTM(encoder_inputs)
    
    encoder_outputs,state_h,state_c = encoder_LSTM2(encoder_outputs)
    
    encoder_states = [state_h, state_c]
    
    """____decoder___"""
    decoder_inputs = Input(shape=(None,de_shape[1]))
    decoder_LSTM = LSTM(hidden_units,dropout_U=0.5,dropout_W=0.5,return_sequences=True,return_state=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_inputs,initial_state=encoder_states)
    
    decoder_dense = Dense(de_shape[1],activation=activation)
    decoder_outputs = decoder_dense(decoder_outputs)
    
    model= Model(inputs=[encoder_inputs,decoder_inputs], outputs=decoder_outputs)
    print(model.summary())
    
    #rmsprop = RMSprop(lr=learning_rate,clipnorm=clip_norm)
    model.compile(loss = LF,optimizer='adam',metrics=['accuracy'])
    
    x_train,x_test,y_train,y_test=tts(x,y,test_size=0.2)
    history= model.fit(x=[x_train,y_train],
              y=y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=([x_test,y_test], y_test))
    """_________________inference mode__________________"""
    encoder_model_inf = Model(encoder_inputs,encoder_states)
    
    decoder_state_input_H = Input(shape=(hidden_units,))
    decoder_state_input_C = Input(shape=(hidden_units,)) 
    decoder_state_inputs = [decoder_state_input_H, decoder_state_input_C]
    decoder_outputs, decoder_state_h, decoder_state_c = decoder_LSTM(decoder_inputs,
                                                                     initial_state=decoder_state_inputs)
    decoder_states = [decoder_state_h, decoder_state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    
    decoder_model_inf= Model([decoder_inputs]+decoder_state_inputs,
                         [decoder_outputs]+decoder_states)
    
    scores = model.evaluate([x_test,y_test],y_test, verbose=1)
    
    print('LSTM test scores:', scores)
    print('\007')
    return model,encoder_model_inf,decoder_model_inf,history



"""________________________(1 layer) encoder-decoder lstm______________________"""

def UniEnDe(lossFunction):
    print(' 1 layer encoder decoder LSTM...')
    
    LF,activation = loss_function(lossFunction)
   
    """__encoder___"""
    encoder_inputs = Input(shape=en_shape)
    
    encoder_LSTM = LSTM(hidden_units,return_sequences=True,return_state=True,go_backwards=True)
    encoder_LSTM2 = LSTM(hidden_units,return_sequences=True,return_state=True)
    
    encoder_outputs,_,_ = encoder_LSTM(encoder_inputs)
    
    encoder_outputs,state_h,state_c = encoder_LSTM2(encoder_outputs)
    
    encoder_states = [state_h, state_c]
    
    """____decoder___"""
    decoder_inputs = Input(shape=(None,de_shape[1]))
    decoder_LSTM = LSTM(hidden_units,dropout_U=0.5,dropout_W=0.5,return_sequences=True,return_state=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_inputs,initial_state=encoder_states)
    
    decoder_dense = Dense(de_shape[1],activation=activation)
    decoder_outputs = decoder_dense(decoder_outputs)
    
    model= Model(inputs=[encoder_inputs,decoder_inputs], outputs=decoder_outputs)
    print(model.summary())
    
    #rmsprop = RMSprop(lr=learning_rate,clipnorm=clip_norm)
    model.compile(loss=LF,optimizer='adam',metrics=['accuracy'])
    
    x_train,x_test,y_train,y_test=tts(x,y,test_size=0.2)
    history= model.fit(x=[x_train,y_train],
              y=y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=([x_test,y_test], y_test))
    """_________________inference mode__________________"""
    encoder_model_inf = Model(encoder_inputs,encoder_states)
    
    decoder_state_input_H = Input(shape=(hidden_units,))
    decoder_state_input_C = Input(shape=(hidden_units,)) 
    decoder_state_inputs = [decoder_state_input_H, decoder_state_input_C]
    decoder_outputs, decoder_state_h, decoder_state_c = decoder_LSTM(decoder_inputs,
                                                                     initial_state=decoder_state_inputs)
    decoder_states = [decoder_state_h, decoder_state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    
    decoder_model_inf= Model([decoder_inputs]+decoder_state_inputs,
                         [decoder_outputs]+decoder_states)
    
    scores = model.evaluate([x_test,y_test],y_test, verbose=1)
    
    print('LSTM test scores:', scores)
    print('\007')
    return model,encoder_model_inf,decoder_model_inf,history


"""________________________(2 layer) encoder-decoder lstm + CNN______________________"""

