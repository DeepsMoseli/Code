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
batch_size = 5
TRAINING_RATIO = 5
num_classes = 1
epochs = 200
Gz_epochs=200
hidden_units = 176
cell_size=256
learning_rate = 0.0005
clip_norm = 0.01

#######################if loaded from pickle###########################
pickler=picklehandler()
data = pickler.loadPickle(ProcData_location,"GAN4_19_10_2018")   
x= data['x']
y= data['y']
#######################################################################


#######################################################################
############################Helper Functions###########################
#######################################################################    

def d_loss(y_true, y_pred):
    return k.mean(y_true * y_pred)


def Generator(data,z):
    print('-----Generator Network-----')
    Ge_shape=(None,z)
    realshape=np.shape(data)
    
    generator_inputs = Input(shape=Ge_shape,name='Input_G')
    
    #Unidirectional stacked LSTM CELL
    generator_LSTM = LSTM(hidden_units,return_sequences=True,return_state=True)
    generator_LSTM2=LSTM(cell_size,return_state=True,return_sequences=True)
    generator_LSTM3 = LSTM(cell_size,return_sequences=True,return_state=True)
    
    gen_outputs, _, _ = generator_LSTM(generator_inputs)
    gen_outputs2, _, _ = generator_LSTM2(gen_outputs)
    #generator_outputs_final= Add()([gen_outputs,gen_outputs_r])
    
    generator_outputs_final2,_,_ = generator_LSTM3(gen_outputs2)
    
    #dense layers
    generator_dense = Dense(int(cell_size), activation = 'linear',kernel_initializer='he_normal')(generator_outputs_final2)
    generator_dense2 = Dense(realshape[-1], activation = 'relu',kernel_initializer='he_normal',name='Output_G')(generator_dense)   
    
    model= Model(inputs=generator_inputs, outputs=generator_dense2,name='G')
    print(model.summary())
    return model
    

def Discriminator(data,labels):
    print('------Discriminator Network-------')
    di_shape=np.shape(data[0])
    discriminator_inputs = Input(shape=di_shape,name='Input_D')
    
    #Bidirectional LSTM CELL
    discriminator_LSTM = LSTM(cell_size,return_sequences=True,return_state=True)
    discriminator_LSTM_rev=LSTM(cell_size,return_state=True,return_sequences=True,go_backwards=True)
    discriminator_LSTM2=LSTM(cell_size,return_state=True,return_sequences=True)
    
    discriminator_outputs,_,_= discriminator_LSTM(discriminator_inputs)
    discriminator_outputs_rev,_,_ = discriminator_LSTM_rev(discriminator_outputs)
    
    #discriminator_outputs_final = Add()([discriminator_outputs,discriminator_outputs_rev])
    
    discriminator_outputs_final,_,_=discriminator_LSTM2(discriminator_outputs_rev)
    
    #First dense layer
    discriminator_dense = Dense(int(hidden_units),activation='linear',kernel_initializer='he_normal')
    discriminator_outputs_final = discriminator_dense(discriminator_outputs_final)
    
    #last dense to output a probability
    discriminator_dense2 = Dense(1,activation='linear',kernel_initializer='he_normal',name="Output_D")
    discriminator_outputs_final = discriminator_dense2(discriminator_outputs_final)
    
    model= Model(inputs=discriminator_inputs, outputs=discriminator_outputs_final, name="D")
    print(model.summary())
    return model


def TrainGan(data,labels,z):
    D = Discriminator(data,labels)
    D.compile(optimizer=RMSprop(lr=0.0005,clipnorm=clip_norm),loss='binary_crossentropy')
    input_z = Input(shape=(None,z), name='input_z_')
    G = Generator(data,z)
    # create combined D(G) model
    output_is_fake= D(G(inputs=input_z))
    DG = Model(inputs=input_z, outputs=output_is_fake)
    DG.compile(optimizer=RMSprop(lr=0.0005,clipnorm=clip_norm),loss=d_loss)
    return D,G,DG



def generate_songs(generator_model, epoch):
    """Feeds random seeds into the generator and tiles and saves the output in a list."""
    test_song_stack = generator_model.predict(np.random.normal(size=(10, 300,hidden_units)))
    test_song_stack = np.squeeze(np.round(test_song_stack).astype(np.uint8))
    return test_song_stack
    


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

#gendata=x[:40]
generator = Generator(x,hidden_units)
discriminator = Discriminator(x,y)

#set discriminator not trainable during generator
for layer in discriminator.layers:
    layer.trainable = False

discriminator.trainable = False
generator_input = Input((None,hidden_units))
generator_layers = generator(generator_input)
discriminator_layers_for_generator = discriminator(generator_layers)
generator_model = Model(inputs=[generator_input], outputs=[discriminator_layers_for_generator])
# We use the Adam paramaters from Gulrajani et al.
generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), loss=d_loss)


# Now that the generator_model is compiled, we can make the discriminator layers trainable.
for layer in discriminator.layers:
    layer.trainable = True
for layer in generator.layers:
    layer.trainable = False
discriminator.trainable = True
generator.trainable = False

# The discriminator_model is more complex. It takes both real image samples and random noise seeds as input.
# The noise seed is run through the generator model to get generated images. Both real and generated images
# are then run through the discriminator. Although we could concatenate the real and generated images into a
# single tensor, we don't (see model compilation for why).
real_samples = Input(shape=x.shape[1:])
generator_input_for_discriminator = Input(shape=np.shape(x[0]))
generated_samples_for_discriminator = generator(generator_input_for_discriminator)
discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
discriminator_output_from_real_samples = discriminator(real_samples)

discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                            outputs=[discriminator_output_from_real_samples,
                                     discriminator_output_from_generator])

discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                            loss=[d_loss,d_loss])    

#################################################################################################
# We make three label vectors for training. positive_y is the label vector for real samples, with value 1.
# negative_y is the label vector for generated samples, with value -1. The dummy_y vector is passed to the
# gradient_penalty loss function and is not used.
positive_y = np.ones((batch_size, 1), dtype=np.float32)
negative_y = -positive_y

GeneratedSongsPerEpoch=[]

discriminator_overall_loss = []
generator_overall_loss =[]

for epoch in range(150):
    np.random.shuffle(x)
    print("Epoch: ", epoch)
    print("Number of batches: ", int(x.shape[0] // batch_size))
    discriminator_loss = []
    generator_loss = []
    minibatches_size = batch_size * TRAINING_RATIO
    for i in range(int(x.shape[0] // (batch_size * TRAINING_RATIO))):
        discriminator_minibatches = x[i * minibatches_size:(i + 1) * minibatches_size]
        for j in range(TRAINING_RATIO):
            image_batch = discriminator_minibatches[j * batch_size:(j + 1) * batch_size]
            noise = np.random.normal(size=(batch_size, 300,hidden_units)).astype(np.float32)
            discriminator_loss.append(discriminator_model.train_on_batch([image_batch, noise],
                                                                         [positive_y, negative_y]))
        generator_loss.append(generator_model.train_on_batch(np.random.normal(size=(batch_size, 300,hidden_units)), positive_y))
    # Still needs some code to display losses from the generator and discriminator, progress bars, etc.
    discriminator_overall_loss.append(discriminator_model.train_on_batch([image_batch, noise],
                                                                         [positive_y, negative_y]))
    generator_overall_loss.append(generator_model.train_on_batch(np.random.normal(size=(batch_size, 300,hidden_units)), positive_y))
    GeneratedSongsPerEpoch.append(generate_songs(generator, epoch))




f=GeneratedSongsPerEpoch[150]







#############################################################################################
pp=G_z.predict(np.random.normal(size=(10,100,hidden_units)))
pp2=cleanupsamples(pp)
pickler.SaveSongForGen(f)


x2=list(x)
y2=list(y)
pp3=list(pp2)
for p in pp3:
    x2.append(p)
    y2.append(0)

x2=np.array(x2)
y2=np.array(y2)

D_x,historyDx= Discriminator(x2,y2)
D_x.predict(np.reshape(pp2[7],(1,100,176)))





test = np.random.randint(0,128,size = (5,np.shape(dataset)[1],np.shape(dataset)[2]))

for k in range(len(test[1])):
    test[1][k][0]=k
    
y_pred = model.predict(pp,verbose=1)


print(y_pred)


