import pyhht
from pyhht.emd import EMD
from pyhht.visualization import plot_imfs
import numpy as np
import math
from pyhht.utils import extr
from pyhht.utils import get_envelops
import matplotlib.pyplot as plt
from pyhht.utils import inst_freq
import wfdb
import os
import sys
import glob

def EMD_data_preparation(csv_folder,samplenumber):
	###########    Trining and Test Data Spliting   ######################
	Ensembled_train = open(csv_folder+'Ensembled_train.csv', 'w')
	#Ensembled_test = open(csv_folder+'Ensembled_test.csv', 'w')
	Total_data = 0
	#Training data prepare
	F = open('../../Data/Saint-Petesberg/Original_train.csv','r')
	line = F.readline()
	while line:
		Original_signal = []
		splitted = line.split(',')
		
		for h in range(0,samplenumber):
			Original_signal.append(float(splitted[h])) 
		disease = splitted[-1][:-1]
		
		Original_signal = np.asarray(Original_signal)
		try:
			decomposer = EMD(Original_signal,n_imfs=3,maxiter=3000)
			imfs = decomposer.decompose()
	
			ensembled_data = []
			for h in range(0,samplenumber):
				ensembled_data.append(imfs[0][h]+imfs[1][h]+imfs[2][h])
	
			Total_data = Total_data+1

			string = str(float("{0:.8f}".format(ensembled_data[0])))

			for h in range(1,samplenumber):
				string = string +','+str(float("{0:.8f}".format(ensembled_data[h])))
			string = string+','+disease+'\n'
			Ensembled_train.write(string)

			print 'Train Data = ',Total_data,'---Disease = ',disease
			line = F.readline()
		
		except:
			print 'Could not Write'
			line = F.readline()
		
	

	Ensembled_train.close()
	#Ensembled_test.close()
	F.close()
	#G.close()
	
	
csv_path = '../../Data/Saint-Petesberg/'
samplenumber = 1000
EMD_data_preparation(csv_path,samplenumber)
