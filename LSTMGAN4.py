from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:29:22 2018

@author: moseli
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:04:52 2018

@author: moseli
"""

from scipy.io.wavfile import read
from scipy.io.wavfile import write
import numpy as np
from sklearn.model_selection import train_test_split as tts
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import librosa.display
import os
import seaborn
import math
import time
import IPython.display as ipd

"""----------------------------------declare inputs--------------------------------------"""

dataLocation="C:\\Users\\moseli\\Documents\\Masters of Information technology\\MIT809\\datasets\\VCTK-Corpus\\wav48"
outputlocation="C:/Users/moseli/Documents/Masters of Information technology/MIT809/code/myImplementation/output/"

sample_rate=48000
train=[]
trainsize=8000
samplelength=[1,4901]
"""---------------------------------Helper functions--------------------------------------"""

def readfile(folder,filename):
    file = read("%s\\%s"%(folder,filename))
    return file

def randomsong():
    return np.ndarray([sample_rate,3],dtype=int)
    
def writesong(filename,data):
    write("%s%s"%(outputlocation,filename),sample_rate,data)


def trainingdata():
    x=0
    for subdir, dirs, files in os.walk(dataLocation):
        for file in files:
            train.append(readfile(subdir,file)[1])
            x+=1          
            if x==trainsize:
                return "Complete: %s audio files loaded"%(trainsize)
    
def visualize(folder,track):
    loc=dataLocation+"\\"+folder+"\\"+"%s_%s.wav"%(folder,track)
    x,y=librosa.load(loc)
    plt.figure(figsize=(12, 4))
    librosa.display.waveplot(x, sr=y)

def playback(track):
    ipd.Audio(track,rate=sample_rate)
    
def sampleAudio(Xarray,start=samplelength[0], stop=samplelength[1]):
    return Xarray[start:stop]

def createSampleTargets(x):
    if int(x[samplelength[1]+5])<0:
        return 0
    elif int(x[samplelength[1]+5])<100:
        return 1
    elif int(x[samplelength[1]+5])<200:
        return 2
    else:
        return 3

def regularize(name):
    reg={'L1':tf.contrib.layers.l1_regularizer(scale=0.003, scope=None),
         'L2':tf.contrib.layers.l2_regularizer(scale=0.003,scope=None)
         }
    return reg[name]

"""----GAN functions-----"""
#discreminator net
Y_C_shape=243
def Discriminator(X):
    #convolution
    global Y_C_shape
    X_reshaped = tf.reshape(X, [-1, int(math.sqrt(inputsize)), int(math.sqrt(inputsize)), 1])   # TensorFlow's convolutional operation wants a "volume" 
    Y1C = tf.nn.relu(tf.nn.conv2d(X_reshaped, W1C, strides=[1, 1, 1, 1], padding='SAME') + B1C)
    pool = tf.nn.max_pool(Y1C, ksize=[1, 1, 1, 1],
                      strides=[1, 2, 2, 1], 
                      padding='SAME')
    Y2C = tf.nn.relu(tf.nn.conv2d(pool, W2C, strides=[1, 2, 2, 1], padding='SAME') + B2C)
    pool2 = tf.nn.max_pool(Y2C, ksize=[1, 1, 1, 1],
                      strides=[1, 2, 2, 1], 
                      padding='SAME')
    Y_C=tf.contrib.layers.flatten(pool2)
    Y_C_shape=int(Y_C.get_shape()[1])
    Y3C=tf.nn.relu(tf.matmul(Y_C,W3C)+B3C)
    #fully connected
    D_h1 = tf.nn.relu(tf.matmul(Y3C, D_W1) + D_b1)
    hidden1 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(hidden1,D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit

#Generator net
def Generator(Z):
    G_h1 = tf.nn.relu(tf.matmul(Z, G_W1) + G_b1)
    hidden2 = tf.nn.relu(tf.matmul(G_h1,G_W2) + G_b2)
    G_out_logit = tf.matmul(hidden2, G_W3) + G_b3
    G_out = tf.nn.tanh(G_out_logit)
    return G_out


def sample_Z(batch_size,latent_space_size):
    '''Uniform prior for G(Z)'''
    return np.random.normal(scale=0.2,size=[batch_size,latent_space_size])
    
#load data and make standard dataset
trainingdata()
data_sets = {}
data_sets['audio_train']=np.array(list(map(sampleAudio,train)))
print(np.shape(data_sets['audio_train']))


data_sets['audio_test']=np.array(list(map(createSampleTargets,train)))
print(np.shape(data_sets['audio_test']))

librosa.display.waveplot(data_sets["audio_train"][10], sr=sample_rate)

final_data={}

final_data["train_X"],final_data["test_X"],final_data["train_y"],final_data["test_y"] = tts(data_sets["audio_train"],data_sets["audio_test"],test_size=0.1)

del train
"""--------------------------------Neural net----------------------------------------"""
#model variables
learning_rate=0.008
lower_lr=0.002
init_sigma=0.2
epoches=100
batch_size=50
steps_epoc=60000/batch_size
num_steps =int(steps_epoc*epoches)
latent_space_size=200

inputsize=samplelength[1]-samplelength[0]
starttime=time.time()

##############################################build graph#############################################
tf.reset_default_graph()

#D(X)

layer_1_depth = 3   # How deep is layer 1?
layer_2_depth = 3   # How deep is layer 2? 
filter_size = 8   # What size of filters do we want?

X = tf.placeholder(tf.float32, shape=[None,inputsize], name='input')
#parameters for layer1
W1C = tf.Variable(tf.truncated_normal([filter_size, filter_size, 1, layer_1_depth] ,stddev=init_sigma),name='CONV_W1')
B1C = tf.Variable(tf.zeros([layer_1_depth]),name='CONV_b1')
#parameters for layer2
W2C = tf.Variable(tf.truncated_normal([filter_size, filter_size, layer_1_depth, layer_2_depth] ,stddev=init_sigma),name='CONV_W2')
B2C = tf.Variable(tf.zeros([layer_2_depth]),name='CONV_b2')

W3C=tf.Variable(tf.truncated_normal([Y_C_shape,inputsize],stddev=init_sigma))
B3C=tf.Variable(tf.zeros([inputsize]),name="CONV_b3")

D_W1 = tf.Variable(tf.truncated_normal([inputsize, 200]), name='D_W1')
D_b1 = tf.Variable(tf.zeros(shape=[200]), name='D_b1')
D_W2 = tf.Variable(tf.truncated_normal([200, 80]), name='D_W2')
D_b2 = tf.Variable(tf.zeros(shape=[80]), name='D_b2')
D_W3 = tf.Variable(tf.truncated_normal([80, 1]), name='D_W3')
D_b3 = tf.Variable(tf.zeros(shape=[1]), name='D_b3')
theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3, W1C, B1C, W2C, B2C, W3C, B3C]

#G(X)
Z = tf.placeholder(tf.float32, shape=[None, latent_space_size], name='G_Z')
G_W1 = tf.Variable(tf.truncated_normal([latent_space_size, 200]), name='G_W1')
G_b1 = tf.Variable(tf.zeros(shape=[200]), name='G_b1')
G_W2 = tf.Variable(tf.truncated_normal([200, 800]), name='G_W2')
G_b2 = tf.Variable(tf.zeros(shape=[800]), name='G_b2')
G_W3 = tf.Variable(tf.truncated_normal([800, inputsize]), name='G_W3')
G_b3 = tf.Variable(tf.zeros(shape=[inputsize]), name='G_b3')
theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

G_sample = Generator(Z)
D_real, D_logit_real = Discriminator(X)
D_fake, D_logit_fake = Discriminator(G_sample)

#Vannila GAN
#D_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(D_real,1e-9,1)) + tf.log(1. - tf.clip_by_value(D_fake,1e-9,0.99999)))
#G_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(D_fake,1e-9,1)))

#WGAN
D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
G_loss = tf.reduce_mean(D_fake)

#regularization_penalty = tf.contrib.layers.apply_regularization(regularize('L2'), theta_G)

train_stepD = tf.train.RMSPropOptimizer(0.0001).minimize(-D_loss,var_list=theta_D)
train_stepG = tf.train.RMSPropOptimizer(0.00005).minimize(G_loss,var_list=theta_G)


#################generate sample song
sample_song=tf.Variable(np.random.normal(size=[100,latent_space_size]),dtype=tf.float32)

init = tf.global_variables_initializer()

"-----------------------------------------------training the network-----------------------------------"

###########################create session and initialize all param
sess=tf.Session()
sess.run(init)

"""_________________tensorboard_________________"""
logdir='C:/ProgramData/Anaconda3/Lib/site-packages/tensorflow/temp'
writer = tf.summary.FileWriter(logdir, sess.graph)

trainingLoss={"D_loss":[],"G_loss":[],"time":[]}
for i in range(num_steps):
    # load batch of audio 

    
    indices = np.random.choice(final_data["train_X"].shape[0], batch_size)
    batch_X = final_data["train_X"][indices]

    # train
    if i<20:
        _,D_loss_curr = sess.run([train_stepD, D_loss],feed_dict={X:batch_X,Z:sample_Z(batch_size,latent_space_size)})
        _,G_loss_curr = sess.run([train_stepG, G_loss],feed_dict={Z:sample_Z(batch_size,latent_space_size)})
        
    elif i<4000:
        _,D_loss_curr = sess.run([train_stepD, D_loss],feed_dict={X:batch_X,Z:sample_Z(batch_size,latent_space_size)})
        #_,G_loss_curr = sess.run([train_stepG, G_loss],feed_dict={Z:sample_Z(batch_size,latent_space_size)})
    else:
        _,D_loss_curr = sess.run([train_stepD, D_loss],feed_dict={X:batch_X,Z:sample_Z(batch_size,latent_space_size)})
        _,G_loss_curr = sess.run([train_stepG, G_loss],feed_dict={Z:sample_Z(batch_size,latent_space_size)})
    
    trainingLoss["D_loss"].append(D_loss_curr)
    trainingLoss["G_loss"].append(G_loss_curr)
    trainingLoss["time"].append(i)
    if i % 50 == 0:
        print("Step", i," LR: ",learning_rate, " D_loss: ",D_loss_curr," G_loss: ",G_loss_curr)
    


with sess.as_default():
    pp=Generator(sample_song).eval()
    for k in range(len(pp)):
        writesong("sample%s.wav"%k,pp[k])
writer.close()

#seaborn.tsplot(data=trainingLoss["G_loss"],time=trainingLoss["time"],legend =True)
seaborn.tsplot(data=trainingLoss["D_loss"],time=trainingLoss["time"],legend =True)