# -*- coding: utf-8 -*-
"""
Created on Sat Jun 09 14:31:42 2018

@author: moseli
"""

from numpy.random import seed
seed(1)
import random
from sklearn.model_selection import train_test_split as tts
import logging


import numpy as np
import keras
from keras import backend as k
k.set_learning_phase(1)
from keras import initializers
from keras.optimizers import RMSprop,Adam
from keras.models import Model
from keras.layers import Dense,LSTM,Input,Activation,Add,TimeDistributed,\
Permute,Flatten,RepeatVector,merge,Lambda,Multiply
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

#######################model params###########################
batch_size = 4
num_classes = 1
epochs = 100
Gz_epochs=200
hidden_units = 512
learning_rate = 0.0001
clip_norm = 2.0

#######################if loaded from pickle###########################
pickler=picklehandler()
data = pickler.loadPickle(ProcData_location,"GAN1_4_9_2018")   
x= data['x']
y= data['y']
#######################################################################


#######################################################################
############################Helper Functions###########################
#######################################################################    

def Generator(data,z):
    print('-----Generator Network-----')
    Ge_shape=(None,z)
    realshape=np.shape(data)
    
    generator_inputs = Input(shape=Ge_shape)
    
    #Bidirectional LSTM CELL
    generator_LSTM = LSTM(hidden_units,return_sequences=True,return_state=True)
    generator_LSTM_rev=LSTM(hidden_units,return_state=True,return_sequences=True,go_backwards=True)
    generator_LSTM_last = LSTM(hidden_units,return_sequences=True,return_state=True)
    
    gen_outputs, _, _ = generator_LSTM(generator_inputs)
    gen_outputs_r, _, _ = generator_LSTM_rev(generator_inputs)
    generator_outputs_final= Add()([gen_outputs,gen_outputs_r])
    
    generator_outputs_final2,_,_ = generator_LSTM_last(generator_outputs_final)
    
    #dense layers
    generator_dense = Dense(hidden_units, activation = 'tanh')(generator_outputs_final2)
    generator_dense2 = Dense(np.shape(data)[-1], activation = 'relu')(generator_dense)   
    
    model= Model(inputs=generator_inputs, outputs=generator_dense2)
    print(model.summary())
    
    
    rmsprop = RMSprop(lr=learning_rate,clipnorm=clip_norm)

    model.compile(loss='binary_crossentropy',optimizer=rmsprop,metrics=["accuracy"])
    
    latent = np.random.normal(size=(realshape[0],realshape[1],z))
    
    x_train,x_test,y_train,y_test=tts(latent,data,test_size=0.20, random_state=40)
    history= model.fit(x=x_train,y=y_train,
              batch_size=batch_size,
              epochs=Gz_epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    
    scores = model.evaluate(x_test,y_test, verbose=1)
    print(scores)
    return model,history
    


def Discriminator(data,labels):
    print('------Discriminator Network-------')
    di_shape=np.shape(data[0])
    discriminator_inputs = Input(shape=di_shape)
    
    #Bidirectional LSTM CELL
    discriminator_LSTM = LSTM(hidden_units,return_sequences=False,return_state=True)
    discriminator_LSTM_rev=LSTM(hidden_units,return_state=True,return_sequences=False,go_backwards=True)
    
    discriminator_outputs,_,_= discriminator_LSTM(discriminator_inputs)
    discriminator_outputs_rev,_,_ = discriminator_LSTM_rev(discriminator_inputs)
    
    discriminator_outputs_final=Add()([discriminator_outputs,discriminator_outputs_rev])
    
    #First dense layer
    discriminator_dense = Dense(int(hidden_units),activation='relu')
    discriminator_outputs_final = discriminator_dense(discriminator_outputs_final)
    
    #last dense to output a probability
    discriminator_dense2 = Dense(1,activation='sigmoid')
    discriminator_outputs_final = discriminator_dense2(discriminator_outputs_final)
    
    model= Model(inputs=discriminator_inputs, outputs=discriminator_outputs_final)
    print(model.summary())
    
    model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
    
    x_train,x_test,y_train,y_test=tts(data,labels,test_size=0.20, random_state=42)
    history= model.fit(x=x_train,y=y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    
    scores = model.evaluate(x_test,y_test, verbose=1)
    
    print('LSTM test scores:', scores)
    print('\007')
    return model,history


def cleanupsamples(samples):
    for k in range(len(samples)):
        for p in range(len(samples[k])):
            for j in range(len(samples[k][p])):
                if samples[k][p][j]<0.05:
                    samples[k][p][j]=0
                else:
                    samples[k][p][j]=1
    return samples
                    

#############################################################################################
################################Test`
#############################################################################################

gendata=x[:40]
G_z,historyGz = Generator(gendata,hidden_units)

pp=G_z.predict(np.random.normal(size=(5,100,hidden_units)))
pp2=cleanupsamples(pp)
pickler.SaveSongForGen(pp2[7])


x2=list(x)
y2=list(y)
pp3=list(pp2)
for p in pp3:
    x2.append(p)
    y2.append(0)

x2=np.array(x2)
y2=np.array(y2)

D_x,historyDx= Discriminator(x2,y2)
D_x.predict(np.reshape(pp2[3],(1,100,176)))





test = np.random.randint(0,128,size = (5,np.shape(dataset)[1],np.shape(dataset)[2]))

for k in range(len(test[1])):
    test[1][k][0]=k
    
y_pred = model.predict(pp,verbose=1)


print(y_pred)


