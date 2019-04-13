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

def EMD_data_preparation(filepath,patient_data,csv_folder,samplenumber,split_perc):

	miscle=['Stable angina','Palpitation', 'Unstable angina']
	cardiom=['Heart failure (NYHA 4)', 'Heart failure (NYHA 3)', 'Heart failure (NYHA 2)']
	ecg_lead = ['i','ii','iii','avr','avl','avf','v1','v2','v3','v4','v5','v6','vx','vy','vz']

	Original_train = open(csv_folder+'Original_train.csv', 'w')
	Original_test = open(csv_folder+'Original_test.csv', 'w')
	f = open(patient_data)
	line = f.readline()
	disease_array=[]
	Total_data = 0

	while line:

		splitted = line.split('/')
		file_name = str(splitted[1][0:8])
		patient_folder = str(splitted[0])

		total_path = filepath+patient_folder+'/'+file_name

		print patient_folder,'---',file_name,
		#print total_path

		try:
			signal,ecgrecord = wfdb.rdsamp(total_path)
			print ecgrecord['comments'][4][22:]

			signal_length = len(signal)

			if not ecgrecord['comments'][4][22:] == 'n/a':

				disease = ecgrecord['comments'][4][22:]

				if disease in miscle:
					disease = "Miscellaneous"
				elif disease in cardiom:
					disease = "Cardiomyopathy"

				if disease not in disease_array:
					disease_array.append(disease)

				samplelength = 0
				undecomposed = 0
				not_match = 0

				if disease == 'Myocardial infarction':
					overlap = 1000
					repetition = int(math.floor(signal_length/samplenumber))
				elif disease == 'Healthy control':
					overlap = 220
					repetition = int(math.floor(((signal_length-samplenumber)/overlap) + 1))
				elif disease == 'Cardiomyopathy':
					overlap = 45
					repetition = int(math.floor(((signal_length-samplenumber)/overlap) + 1))
				elif disease == 'Bundle branch block':
					overlap = 46
					repetition = int(math.floor(((signal_length-samplenumber)/overlap) + 1))
				elif disease == 'Dysrhythmia':
					overlap = 30
					repetition = int(math.floor(((signal_length-samplenumber)/overlap) + 1))
				elif disease == 'Hypertrophy':
					overlap = 20
					repetition = int(math.floor(((signal_length-samplenumber)/overlap) + 1))
				elif disease == 'Valvular heart disease':
					overlap = 9
					repetition = int(math.floor(((signal_length-samplenumber)/overlap) + 1))
				elif disease == 'Myocarditis':
					overlap = 11
					repetition = int(math.floor(((signal_length-samplenumber)/overlap) + 1))
				elif disease == 'Miscellaneous':
					overlap = 55
					repetition = int(math.floor(((signal_length-samplenumber)/overlap) + 1))

				stop = int(math.ceil(repetition*split_perc))

				###########    Trining and Test Data Spliting   ######################
				#Training data prepare
				for j in range(0,stop):
					write_signal = []
					for sample in range(samplelength,samplelength+samplenumber):
						ecg_signal = 0
						for i1 in range(0,15):
							ecg_signal = ecg_signal+signal[sample][i1]
						write_signal.append(ecg_signal)

					Original_signal = np.asarray(write_signal)
					
					try:
						
						Total_data = Total_data+1
			
						string = str(Original_signal[0])
					
						for h in range(1,samplenumber):
							string = string +','+str(float("{0:.3f}".format(Original_signal[h])))
						string = string+','+disease+'\n'
						Original_train.write(string)
			
						samplelength = samplelength+overlap
					
					except:
						print 'Could not Write'
						samplelength = samplelength+overlap
					
				#Testing data preparation
				for j in range(stop,repetition):
					write_signal = []
					for sample in range(samplelength,samplelength+samplenumber):
						ecg_signal = 0
						for i1 in range(0,15):
							ecg_signal = ecg_signal+signal[sample][i1]
						write_signal.append(ecg_signal)

					Original_signal = np.asarray(write_signal)

					try:
						
						Total_data = Total_data+1
				
						string = str(Original_signal[0])
						for h in range(1,samplenumber):
							string = string +','+str(float("{0:.6f}".format(Original_signal[h])))
						string = string+','+disease+'\n'
						Original_test.write(string)
		
						samplelength = samplelength+overlap

					except:
						print 'Could not write'
						samplelength = samplelength+overlap

			line = f.readline()

		except:
			line = f.readline()
			print sys.exc_info(),'\n'

	f.close()
	problem_data.close()
	print disease_array
	print Total_data

file_path='/media/nahian/New Volume/My_Research-Running/Machine Learning/ECG/Databases/PTB database/'
csv_path = '../Data/PTB/'
patient_data=file_path+'RECORDS.txt'
samplenumber = 1000
split_perc = 0.7
EMD_data_preparation(file_path,patient_data,csv_path,samplenumber,split_perc)
