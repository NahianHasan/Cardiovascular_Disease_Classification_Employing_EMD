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

def EMD_data_preparation(filepath,patient_data,samplenumber,number_of_IMFs):

	miscle=['Stable angina','Palpitation', 'Unstable angina']
	cardiom=['Heart failure (NYHA 4)', 'Heart failure (NYHA 3)', 'Heart failure (NYHA 2)']
	ecg_lead = ['i','ii','iii','avr','avl','avf','v1','v2','v3','v4','v5','v6','vx','vy','vz']

	f = open(patient_data)
	line = f.readline()
	disease_array=[]

	while line:
		#splitted = line.split('/')
		#file_name = str(splitted[1][0:8])
		file_name = line[0:-1]
		#patient_folder = str(splitted[0])

		total_path = filepath+file_name

		print total_path

		#try:
		signal,ecgrecord = wfdb.rdsamp(total_path)
		record = wfdb.rdsamp(total_path)
		#print ecgrecord['comments'][4][22:]

		signal_length = len(signal)
		repetition = int(math.floor(signal_length/samplenumber))

		samplelength = 0
		undecomposed = 0

		stop = int(math.ceil(repetition*0.7))

		###########    Trining and Test Data Spliting   ######################
		#Training data prepare
		for j in range(0,stop):
			write_signal = []
			for sample in range(samplelength,samplelength+samplenumber):
				ecg_signal = 0
				for i1 in range(0,12):
					ecg_signal = ecg_signal+signal[sample][i1]
				write_signal.append(ecg_signal)

			EMD_signal = np.asarray(write_signal)

			#try:
			decomposer = EMD(EMD_signal,n_imfs=number_of_IMFs,maxiter=3000)
			imfs = decomposer.decompose()
			#print len(imfs)

			modified_EMD = []
			for h in range(0,samplenumber):
				modified_EMD.append(imfs[0][h]+imfs[1][h]+imfs[2][h])

			### Plot data
			fig = plt.figure(figsize=(25,15))
			plt.subplot(2,1,1)
			plt.plot(EMD_signal)
			plt.ylabel('Original Signal\n Amplitude', labelpad=15, fontsize=35)
			plt.xticks(fontsize=35)
			plt.yticks(fontsize=35)
			plt.subplot(2,1,2)
			plt.plot(modified_EMD)
			plt.ylabel('Modified Signal \n Amplitude',labelpad=15, fontsize=35)
			plt.xticks(fontsize=35)
			plt.yticks(fontsize=35)
			plt.xlabel('Sample Number',fontsize=35)
			fig.tight_layout()
			plt.savefig('modified_ECG_Petersburg.eps', format='eps', dpi=6000)
			plt.show()

			samplelength = samplelength+samplenumber

		line = f.readline()

	f.close()
	problem_data.close()
	print disease_array

filepath = '../../ECG_signal_noise_analysis/Database/ST_Petersberg/'
patient_data = filepath+'RECORDS.txt'
samplenumber = 1000
number_of_IMFs = 6
EMD_data_preparation(filepath,patient_data,samplenumber,number_of_IMFs)
