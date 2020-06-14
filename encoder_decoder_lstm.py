# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 21:10:41 2018

@author: moseli
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from numpy.random import seed
seed(1)

from sklearn.model_selection import train_test_split as tts
import logging
import numpy as np
import pandas as pd
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

import keras
import keras_metrics
import pickle as pick
import tensorflow as tf
from keras import backend as k
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix as cm, accuracy_score
from sklearn.metrics import roc_curve
k.set_learning_phase(1)
from keras.preprocessing.text import Tokenizer
from keras import initializers
from keras.optimizers import RMSprop
from keras.models import Sequential,Model
from keras.layers import Dense,LSTM,Dropout,Input,Activation,Add,Concatenate
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.models import load_model
import seaborn as sb
import pandas as pd
from matplotlib import pyplot as plt
import time
from tqdm import tqdm

seed = 7
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

"""________________________________________________________________________________"""
#######################model params####################################
batch_size = 8
num_classes = 1
classes=2
epochs = 80
hidden_units = 256
learning_rate = 0.001
clip_norm = 0.8
kfolds=2

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

results_location="C:\\Users\\Deeps\\Documents\\School\\MIT807\\backup\\accuracy_crossValidation\\"
kfold_location = "C:\\Users\\Deeps\\Documents\\School\\MIT807\\backup\\accuracy_crossValidation\\kfold\\"
#######################if loaded from pickle###########################
pickler=picklehandler()
data = pickler.loadPickle(ProcData_location,"eco_deco2_3_10_2018")   
x= data['x'][:50]
y= data['y'][:50]



#######################################################################
#count imbalance#
zeros=0
ones=0
for kkk in range(len(y)):
    for p in range(len(y[kkk])):
        for t in range(len(y[kkk][p])):
            if y[kkk][p][t]==0:
                zeros+=1
            else:
                ones+=1

print(ones)
print(zeros)
print(ones+zeros)
print(1-(ones/zeros))


#######################################################################
en_shape=np.shape(x[0])
de_shape=np.shape(y[0])


######################################################################

# define 10-fold cross validation test harness
cv_scores=[]
cv_history={}


def TPR(y_true, y_pred):	
    TP = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))	
    possible_positives = k.sum(k.round(k.clip(y_true, 0, 1)))	
    recall = TP / (possible_positives + k.epsilon())	
    return recall

def TNR(y_true, y_pred):	
    TN =176- k.sum(k.round(k.clip(y_true, 0, 1)))	
    FP = 176 - k.sum(k.round(k.clip(y_pred, 0, 1)))
    TNRValue = TN / (TN + FP + k.epsilon())	
    return TNRValue

def BAcc(y_true,y_pred):
    TP = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))	
    possible_positives = k.sum(k.round(k.clip(y_true, 0, 1)))	
    recall = TP / (possible_positives + k.epsilon())	
    
    TN =176- k.sum(k.round(k.clip(y_true, 0, 1)))	
    FP = 176 - k.sum(k.round(k.clip(y_pred, 0, 1)))
    TNRValue = TN / (TN + FP + k.epsilon())	
    
    return (recall+TNRValue)/2
    

def customLoss(y_true,y_pred):
    TP=k.sum(k.clip(y_true * y_pred, 0, 1))
    possible_positives = k.sum(k.clip(y_true, 0, 1))
    recall = TP / (possible_positives + k.epsilon())	
    TN =176- k.sum(k.clip(y_true, 0, 1))	
    FP = 176 - k.sum(k.clip(y_pred, 0, 1))
    TNRValue = TN / (TN + FP + k.epsilon())
    return 1-((recall+TNRValue)/2)

def encoder_decoder(learning_rate,batchS):
    randstate = np.random.randint(1,100)
    print('Encoder_Decoder LSTM... ' + str(randstate))
   
    """__encoder___"""
    encoder_inputs = Input(shape=en_shape)
    
    encoder_LSTM = LSTM(hidden_units,return_sequences=True,return_state=True)
    encoder_LSTM2 = LSTM(hidden_units,return_sequences=True,return_state=True,go_backwards=True)
    encoder_LSTM3 = LSTM(hidden_units,return_sequences=True,return_state=True)
    
    encoder_outputs1,_,_ = encoder_LSTM(encoder_inputs)
    encoder_outputs2,_,_ = encoder_LSTM2(encoder_inputs)
    
    encoder_outputs_combined = Add()([encoder_outputs1,encoder_outputs2])
    
    encoder_outputs,state_h,state_c = encoder_LSTM3(encoder_outputs_combined)
    
    
    encoder_states = [state_h, state_c]
    
    """____decoder___"""
    decoder_inputs = Input(shape=(None,de_shape[1]))
    decoder_LSTM = LSTM(hidden_units,return_sequences=True,return_state=True)
    decoder_LSTM2 = LSTM(hidden_units,return_sequences=True,return_state=True)
    
    decoder_outputs1, _, _ = decoder_LSTM(decoder_inputs,initial_state=encoder_states)
    decoder_outputs, _, _ = decoder_LSTM2(decoder_outputs1)
    
    decoder_dense = Dense(de_shape[1],activation='relu')
    decoder_outputs = decoder_dense(decoder_outputs)
        
    
    model= Model(inputs=[encoder_inputs,decoder_inputs], outputs=decoder_outputs)
    print(model.summary())
    
    rmsprop = RMSprop(lr=learning_rate,clipnorm=clip_norm,decay=learning_rate*0.001)
    
    model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=[BAcc])
    
    x_train,x_test,y_train,y_test=tts(x,y,test_size=0.2,random_state=randstate)
    history= model.fit(x=[x_train,y_train],
              y=y_train,
              batch_size=batchS,
              epochs=epochs,
              verbose=1,
              validation_data=([x_test,y_test], y_test))
    """_________________inference mode__________________"""
    encoder_model_inf = Model(encoder_inputs,encoder_states)
    
    decoder_state_input_H = Input(shape=(hidden_units,))
    decoder_state_input_C = Input(shape=(hidden_units,)) 
    decoder_state_inputs = [decoder_state_input_H, decoder_state_input_C]
    
    decoder_outputs, decoder_state_h, decoder_state_c= decoder_LSTM(decoder_inputs,initial_state=decoder_state_inputs)
    decoder_outputs, decoder_state_h, decoder_state_c = decoder_LSTM2(decoder_outputs)
    
    decoder_states = [decoder_state_h, decoder_state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    
    decoder_model_inf= Model([decoder_inputs]+decoder_state_inputs,
                         [decoder_outputs]+decoder_states)
    
    scores = model.evaluate([x_test,y_test],y_test, verbose=1)
    cv_scores.append(scores)
    
    print('LSTM test scores:', scores)
    print('\007')
    return model,encoder_model_inf,decoder_model_inf,history



"""________________generate song vectors___________"""

def generateSong(sample,indeX):
    stop_pred = False
    sample =  np.reshape(sample,(1,en_shape[0],en_shape[1]))
    #get initial h and c values from encoder
    init_state_val = encoder.predict(sample)
    #target_seq = np.ones((1,1,de_shape[1]))
    target_seq=np.random.uniform(size=(1,1,de_shape[1]))
    #start with random note from training set
    #target_seq =np.reshape(y[indeX][0],(1,1,de_shape[1]))
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


batchsizes = [5,8,10]
lrates=[0.0005,0.001,0.005]

for batchsize in batchsizes:
    print("############### Batch Size_" + str(batchsize)+"_####################")
    for rate in lrates:
        print("############### Rate_" + str(rate)+"_####################")
        cv_history={}
        for kf in range(kfolds):
            _,_,_,history = encoder_decoder(rate,batchsize)
            cv_history["batSize_"+str(batchsize) + "_CV_" + str(kf) + "_rate_" + str(rate)]=history.history
            pd.DataFrame(cv_history["batSize_"+str(batchsize) + "_CV_" + str(kf) + "_rate_" + str(rate)]).to_csv("%s%s_%s_%s.csv"%(kfold_location,batchsize,rate,kf),index=False)
        


stop=1000000000
for kf in tqdm(range(kfolds)):
    trained_model,encoder,decoder,history = encoder_decoder(0.005,5)



for p in cv_history:
    with open('%s%s.pickle'%(results_location,"cv_history_"+p), 'wb') as f:
                    pick.dump(cv_history[p].history, f,protocol=2)
                    
   
for jj in range(5):
    with open('%s%s.pickle'%(results_location,"cv_history_"+str(jj)), 'rb') as f:
        x = pick.load(f)
    pd.DataFrame(x).to_csv("%s_cv%s.csv"%(results_location,jj),index=False)

"""
weights={}
total=0
for i in range(116):
    for o in range(300):
        for p in range(176):
            if i==0:
                weights[p]=0
            if y[i][o][p]==1:
                weights[p]+=1
                total+=1
            else:
                total+=1

class_weights = np.zeros((300, 176))
for j in weights:
    class_weights[:, j] = weights[j]

sum(class_weights,1)

"""
allplot={}
for p in range(3):
    for  j in range(3):
        k1=pd.read_csv ("%s%s_%s_%s.csv"%(kfold_location,batchsizes[p],lrates[j],0))
        k2=pd.read_csv ("%s%s_%s_%s.csv"%(kfold_location,batchsizes[p],lrates[j],1))
        mean_train_accuracy=list((k1["BAcc"]+k2["BAcc"])/2)
        mean_CV_accuracy=list((k1["val_BAcc"]+k2["val_BAcc"])/2)
        #allplot["%s_%s_BAcc"%(batchsizes[p],lrates[j])]=mean_train_accuracy
        allplot["%s_%s_val_BAcc"%(batchsizes[p],lrates[j])]=mean_CV_accuracy
allplotdf=pd.DataFrame(allplot)

#plot history

history2=pd.DataFrame(cv_history['batSize_10_CV_0_rate_0.005'])
history3=pd.DataFrame(cv_history['batSize_10_CV_1_rate_0.005'])
history4=pd.DataFrame(cv_history['2'].history)
history5=pd.DataFrame(cv_history['3'].history)
history6=pd.DataFrame(cv_history['4'].history)

mean_train_accuracy = np.mean([history2['loss'],history3['loss'],history4['loss'],history5['loss'],history6['loss']],axis=0)
mean_CV_accuracy = np.mean([history2['val_loss'],history3['val_loss'],history4['val_loss'],history5['val_loss'],history6['val_loss']],axis=0)

maxes={}
for p in allplotdf:
    plt.plot(allplotdf[p])
    maxes[p]=max(allplotdf[p])
plt.title("Model training and CV Balanced accuracy for diffferent parameters")
plt.ylabel('Balanced accuracy')
plt.xlabel('Epoch')
plt.grid('on')
plt.legend([p for p in allplotdf],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

print(maxes)


'-----loss----'
plt.plot(history2['loss'])
plt.plot(history2['acc'])
plt.title("Model Loss And Accuracy")
plt.ylabel('Loss')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid('on')
plt.legend(['Loss','Accuracy'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

"""___________________________________Sample___________________________________________"""

def getNotes(indeX):
    pp=y[indeX]
    sample_song = np.reshape(generateSong(x[indeX],indeX),pp.shape)
    for kk in range(len(sample_song)):
        for p in range(len(sample_song[kk])):
            if sample_song[kk][p]<=np.random.randint(2,7)/100:
                #sample_song[kk][p]=0
                pass
            else:
                #sample_song[kk][p]=1
                pass
    return sample_song


pred=getNotes(10)
pickler.SaveSongForGen(pred)

pred2=pred.reshape((176,300))
pred3=y[98]
sb.heatmap(y[65][:176])

plt.figure(figsize=(10,15))
plt.imshow(pred3[:200])
plt.title("Note Progression State Matrix")
plt.ylabel('Time')
plt.xlabel('2N Pitch Information')
plt.show()

