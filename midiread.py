# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 20:28:00 2018

@author: moseli
"""

import midi,numpy,numpy as np
import logging
import pickle as pick
import datetime
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)


#####################################File Locations##########################
file_location="C:\\Users\\Deeps\\Documents\\School\\MIT807\\Data\\"
ProcData_location ="C:\\Users\\Deeps\Documents\\School\\MIT807\\Code\\saved_data\\"
output_location="C:\\Users\\Deeps\\Documents\\School\\MIT807\\Code\\output\\"
############################################################################


####################################Constants################################
lowerBound = 21
upperBound = 109
threshold = 0.5
##############################################################################


"""---------------------------------------------------------------------------------------"""
class midiIO:
              
    #creates state matrix from a midi file
    def midiToNoteStateMatrix(self,midifile):
        pattern = midi.read_midifile(midifile)
        timeleft = [track[0].tick for track in pattern]
        posns = [0 for track in pattern]
        statematrix = []
        span = upperBound-lowerBound
        time = 0
    
        state = [[0,0] for x in range(span)]
        statematrix.append(state)
        while True:
            if time % (pattern.resolution / 4) == (pattern.resolution / 8):
                # Crossed a note boundary. Create a new state, defaulting to holding notes
                oldstate = state
                state = [[oldstate[x][0],0] for x in range(span)]
                statematrix.append(state)
    
            for i in range(len(timeleft)):
                while timeleft[i] == 0:
                    track = pattern[i]
                    pos = posns[i]
    
                    evt = track[pos]
                    if isinstance(evt, midi.NoteEvent):
                        if (evt.pitch < lowerBound) or (evt.pitch >= upperBound):
                            pass
                            # print "Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time)
                        else:
                            if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                                state[evt.pitch-lowerBound] = [0, 0]
                            else:
                                state[evt.pitch-lowerBound] = [1, 1]
                    elif isinstance(evt, midi.TimeSignatureEvent):
                        if evt.numerator not in (2, 4):
                            # We don't want to worry about non-4 time signatures. Bail early!
                            # print "Found time signature event {}. Bailing!".format(evt)
                            return statematrix
    
                    try:
                        timeleft[i] = track[pos + 1].tick
                        posns[i] += 1
                    except IndexError:
                        timeleft[i] = None
    
                if timeleft[i] is not None:
                    timeleft[i] -= 1
    
            if all(t is None for t in timeleft):
                break
    
            time += 1
    
        return statematrix
    
        #Looad all songs and convert to state matrices
    def loadallsongs(self,dire):
        allSongs=[]
        num=1
        count=1
        for dirs,subdr, files in os.walk(dire):
            for fil in files:
                try:
                    allSongs.append(self.midiToNoteStateMatrix("%s"%(dirs+'\\'+fil)))
                    count+=1
                except Exception as e:
                    print("file Number %s: %s"%(num,e))
                    pass
                num+=1
        print("Successfully loaded %s of %s Midi files"%(count,num))
        return allSongs
    
    #converts a statematrix into a midifile and exports
    def noteStateMatrixToMidi(self,statematrix, name="example"):
        statematrix = numpy.asarray(statematrix)
        pattern = midi.Pattern()
        track = midi.Track()
        pattern.append(track)
        
        span = upperBound-lowerBound
        tickscale = 65
        
        lastcmdtime = 0
        prevstate = [[0,0] for x in range(span)]
        for time, state in enumerate(statematrix + [prevstate[:]]):  
            offNotes = []
            onNotes = []
            for i in range(span):
                n = state[i]
                p = prevstate[i]
                if p[0] == 1:
                    if n[0] == 0:
                        offNotes.append(i)
                    elif n[1] == 1:
                        offNotes.append(i)
                        onNotes.append(i)
                elif n[0] == 1:
                    onNotes.append(i)
            for note in offNotes:
                track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale,channel=10,pitch=note+lowerBound))
                lastcmdtime = time
            for note in onNotes:
                track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale,channel=10,velocity=70, pitch=note+lowerBound))
                lastcmdtime = time
                
            prevstate = state
        
        eot = midi.EndOfTrackEvent(tick=1)
        track.append(eot)
    
        midi.write_midifile("%s%s.mid"%(output_location,name), pattern)
        
        
"""-------------------------------------------------------------------------------------"""

class statematrixmanipulation:
    def __init__(self,getKeepActivated):
        self.getKeepActivated=True
        
    # Returns a flattened state matrix (size 2* span)
    def flatStateMatrix(self,statematrix):
    	flatStateMatrix = []
    	for state in statematrix:
    		flatActivate = []
    		flatKeepActivated = []
    		for note in state:
    			flatActivate.append(max(note))
    			if self.getKeepActivated:
    				flatKeepActivated.append(note[1])
    		if self.getKeepActivated:
    			flatStateMatrix.append(flatActivate + flatKeepActivated)
    		else:
    			flatStateMatrix.append(flatActivate)
    	return np.asarray(flatStateMatrix)
    
    
    # Flat state matrix to two states 
    def unflattenStateMatrix(self,flatStateMatrix):
    	statematrix = []
    	if self.getKeepActivated:
    		offset = len(flatStateMatrix[0])/2
    	else:
    		offset = len(flatStateMatrix[0])
    	for state in flatStateMatrix:
    		newState = []
    		for i in range(offset):
    			if self.getKeepActivated:
    				newState.append([state[i], state[i+offset]])
    			else:
    				newState.append([state[i], 0])
    		statematrix.append(newState)
    	return statematrix
    
    def createTrainingData(self,database,x_len=100,y_len=300):
        x=[]
        y=[]
        for song in database:
            song2=statematrixmanipulation.flatStateMatrix(self,song)
            x.append(song2[:x_len])
            y.append(song2[x_len:x_len+y_len])
        return np.array(x),np.array(y)
    
    
class picklehandler:
        #save data in pickle file    
    def pickleFickle(self,data_x,data_y,fileName):
        data = {'x': data_x, 'y': data_y}
        now = datetime.datetime.now()
        date ='%s_%s_%s'%(now.day,now.month,now.year)
        try:
            with open('%s%s_%s.pickle'%(ProcData_location,fileName,date), 'wb') as f:
                pick.dump(data, f)
            status=True
        except Exception as e:
            raise
            status = False
        return "Save pickle status is: %s"%status

    #load data dictionary
    def loadPickle(self,location,fileName):
        with open('%s%s.pickle'%(location,fileName), 'rb') as f:
            new_data_variable = pick.load(f)
        return new_data_variable
    
    #save one song for python 2.7
    def SaveSongForGen(self,song):
        try:
            with open('%s%s.pickle'%(output_location,"tempsong"), 'wb') as f:
                pick.dump(song, f,protocol=2)
            status=True
        except Exception as e:
            raise
            status = False
        return status
    
        

###############################################################################
########################for outputing  data from p2 to p3#####################
parser=midiIO()
dataset=parser.loadallsongs(file_location)

for k in range(len(dataset)):
    if len(dataset[k])<400:
        del dataset[k]
    else:
        pass

for k in range(len(dataset)):
    print(len(dataset[k]))

manip = statematrixmanipulation(True)
training_data=manip.createTrainingData(dataset)

x=training_data[0]
y=training_data[1]

pickler=picklehandler()
pickler.pickleFickle(x,y,"pianoroll38")

#########################################################################################
#######################For output only, after generating in python3######################

pickler=picklehandler()
gensong=pickler.loadPickle(output_location,"tempsong")

#unflatten
statemanip=statematrixmanipulation(True)
gensongmatrix=statemanip.unflattenStateMatrix(gensong)
parser.noteStateMatrixToMidi(gensongmatrix,name="seventeenth")
#####################################################################