# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 10:44:22 2018

@author: moseli
"""
import numpy as np
import os


file_location="C:\\Users\\moseli\\Documents\\Masters of Information technology\\MIT809\\datasets\\midi\\Collins Phil\\"
output_location="C:\\Users\\moseli\\Documents\\Masters of Information technology\\MIT809\\code\\myImplementation\\output\\"


statemat=midiToNoteStateMatrix("%sAgainst All Odds.6.mid"%file_location)
noteStateMatrixToMidi(statemat)



######################################################################################

# Returns a flattened state matrix (size 2* span)
def flatStateMatrix(statematrix, getKeepActivated):
	flatStateMatrix = []
	for state in statematrix:
		flatActivate = []
		flatKeepActivated = []
		for note in state:
			flatActivate.append(max(note))
			if getKeepActivated:
				flatKeepActivated.append(note[1])
		if getKeepActivated:
			flatStateMatrix.append(flatActivate + flatKeepActivated)
		else:
			flatStateMatrix.append(flatActivate)
	return np.asarray(flatStateMatrix)

# Flat state matrix to two states 
def unflattenStateMatrix(flatStateMatrix, getKeepActivated):
	statematrix = []
	if getKeepActivated:
		offset = len(flatStateMatrix[0])/2
	else:
		offset = len(flatStateMatrix[0])
	for state in flatStateMatrix:
		newState = []
		for i in range(offset):
			if getKeepActivated:
				newState.append([state[i], state[i+offset]])
			else:
				newState.append([state[i], 0])
		statematrix.append(newState)
	return statematrix


# Retrieves state matrices from midi files
def getStateMatrices(getKeepActivated):

	stateMatrices = []
	for file in os.listdir(os.path.abspath(file_location)):
		matrix = midiToNoteStateMatrix(file_location+file)
		if matrix is not None:
			stateMatrices.append(flatStateMatrix(matrix, getKeepActivated))
		else:
			os.remove(file_location+file);
	print "Number of files processed: ", len(stateMatrices)
	print "Total number of states: ", sum(len(mat) for mat in stateMatrices)
	return np.asarray(stateMatrices)

# Gets a random batch
def getNextBatch(stateMatrices, batchSize, notesNb):
	batch_xs = []
	batch_ys = []

	if batchSize > 0:
		for i in np.random.randint(0, len(stateMatrices), batchSize):
			while(len(stateMatrices[i]) <= notesNb+1):
				print "Number of states too small for stateMatrices[", i, "]"
				i += 1
			sampleStartPoint = np.random.randint(1, len(stateMatrices[i])-notesNb, 1)[0]
			batch_xs.append(stateMatrices[i][sampleStartPoint-1:sampleStartPoint-1+notesNb])
			batch_ys.append(stateMatrices[i][sampleStartPoint-1+notesNb])

		print "Total training data of size", len(batch_xs), "generated"
		return np.asarray(batch_xs), np.asarray(batch_ys)

	else:
		for matrix in stateMatrices:
			if(len(matrix) <= notesNb+1 ):
				print "Number of states too small for stateMatrices[", i, "]"
				continue
			for i in range(0, len(matrix)-notesNb, 1):
				batch_xs.append(matrix[i:i+notesNb])
				batch_ys.append(matrix[i+notesNb])
		print "Total training data of size", len(batch_xs), "generated"
		return np.asarray(batch_xs), np.asarray(batch_ys)
    
    
    
    
    
############################################################
flatstate=flatStateMatrix(statemat,True)
statemat2= unflattenStateMatrix(flatstate,True)
