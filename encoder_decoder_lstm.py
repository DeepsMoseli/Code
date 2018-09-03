# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 21:10:41 2018

@author: moseli
"""

from numpy.random import seed
seed(1)

from sklearn.model_selection import train_test_split as tts
import logging
import numpy as np


import keras
from keras import backend as k
from sklearn.metrics import log_loss
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

"""________________________________________________________________________________"""
#######################model params####################################
batch_size = 4
num_classes = 1
epochs = 100
hidden_units = 256
learning_rate = 0.002
clip_norm = 1.0

#######################################################################
############################Helper Functions###########################
#######################################################################    
"""
manip = statematrixmanipulation(True)
training_data=manip.createTrainingData(dataset)

x=training_data[0]
y=training_data[1]

pickler=picklehandler()
pickler.pickleFickle(x,y,"stateMatrixData120")

"""
#######################if loaded from pickle###########################
pickler=picklehandler()
data = pickler.loadPickle(ProcData_location,"pianoroll38_3_9_2018")   
x= data['x']
y= data['y']

#######################################################################

en_shape=np.shape(x[0])
de_shape=np.shape(y[0])


######################################################################


def encoder_decoder():
    print('Encoder_Decoder LSTM...')
   
    """__encoder___"""
    encoder_inputs = Input(shape=en_shape)
    
    encoder_LSTM = LSTM(hidden_units,return_sequences=True,dropout_U=0.1,
                        dropout_W=0.1,recurrent_initializer='normal', bias_initializer='ones',return_state=True)
    encoder_LSTM2 = LSTM(hidden_units,return_sequences=True,dropout_U=0.1,
                         dropout_W=0.1,recurrent_initializer='normal', bias_initializer='ones',return_state=True)
    encoder_LSTM3 = LSTM(hidden_units,return_sequences=True,
                         recurrent_initializer='normal', bias_initializer='ones',return_state=True)
    
    encoder_outputs,_,_ = encoder_LSTM(encoder_inputs)
    encoder_outputs,_,_ = encoder_LSTM2(encoder_outputs)
    encoder_outputs,state_h,state_c = encoder_LSTM3(encoder_outputs)
    
    
    encoder_states = [state_h, state_c]
    
    """____decoder___"""
    decoder_inputs = Input(shape=(None,de_shape[1]))
    decoder_LSTM = LSTM(hidden_units,return_sequences=True,
                        recurrent_initializer='normal',bias_initializer='ones',return_state=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_inputs,initial_state=encoder_states)
    
    decoder_dense = Dense(de_shape[1],activation='sigmoid')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    
    model= Model(inputs=[encoder_inputs,decoder_inputs], outputs=decoder_outputs)
    print(model.summary())
    
    rmsprop = RMSprop(lr=learning_rate,clipnorm=clip_norm)
    #model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.compile(loss='binary_crossentropy',optimizer=rmsprop,metrics=['accuracy'])
    
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
    
    decoder_outputs, decoder_state_h, decoder_state_c= decoder_LSTM(decoder_inputs,initial_state=decoder_state_inputs)
    #decoder_outputs, decoder_state_h, decoder_state_c = decoder_LSTM2(decoder_outputs)
    
    decoder_states = [decoder_state_h, decoder_state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    
    decoder_model_inf= Model([decoder_inputs]+decoder_state_inputs,
                         [decoder_outputs]+decoder_states)
    
    scores = model.evaluate([x_test,y_test],y_test, verbose=1)
    
    print('LSTM test scores:', scores)
    print('\007')
    return model,encoder_model_inf,decoder_model_inf,history



"""________________generate song vectors___________"""

def generateSong(sample,indeX):
    stop_pred = False
    sample =  np.reshape(sample,(1,en_shape[0],en_shape[1]))
    #get initial h and c values from encoder
    init_state_val = encoder.predict(sample)
    #target_seq = np.zeros((1,1,de_shape[1]))
    #target_seq=np.random.normal(size=(1,1,de_shape[1]))
    #start with random note from training set
    target_seq =np.reshape(y[indeX][0],(1,1,de_shape[1]))
    generated_song=[]
    while not stop_pred:
        decoder_out,decoder_h,decoder_c= decoder.predict(x=[target_seq]+init_state_val)
        generated_song.append(decoder_out)
        init_state_val= [decoder_h,decoder_c]
        target_seq=np.reshape(decoder_out,(1,1,de_shape[1]))
        if len(generated_song)== de_shape[0]:
            stop_pred=True
            break
    return np.array(generated_song).reshape((1,de_shape[0],de_shape[1]))


"""___________________________________________________________________________________"""

trained_model,encoder,decoder,history = encoder_decoder()

"""___________________________________Sample___________________________________________"""

def getNotes(indeX):
    pp=y[indeX]
    sample_song = np.reshape(generateSong(x[indeX],indeX),pp.shape)
    for k in range(len(sample_song)):
        for p in range(len(sample_song[k])):
            if sample_song[k][p]<=np.random.randint(40,50)/100:
                #sample_song[k][p]=0
                pass
            else:
                #sample_song[k][p]=1
                pass
    return sample_song


pred2=getNotes(0)

pickler.SaveSongForGen(pred)



