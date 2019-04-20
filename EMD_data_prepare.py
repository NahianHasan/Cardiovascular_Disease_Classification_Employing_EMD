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
import GPUtil
import argparse
import tensorflow as tf
import time

def EMD_data_preparation(filepath,patient_data,csv_folder,problem_data_file,samplenumber,number_of_IMFs,split_perc):

	files = glob.glob('./csv_folder/*')
	for f in files:
		os.remove(f)
	problem_data=open(problem_data_file,'w')

	#PTB Diagnostic ECG database Disease labels 
	miscle=['Stable angina','Palpitation', 'Unstable angina']
	cardiom=['Heart failure (NYHA 4)', 'Heart failure (NYHA 3)', 'Heart failure (NYHA 2)']
	ecg_lead = ['i','ii','iii','avr','avl','avf','v1','v2','v3','v4','v5','v6','vx','vy','vz']
	Sig_Records = {'Bundle branch block': 38092, 'Valvular heart disease': 37647, 'Myocarditis': 39672, 'Healthy control': 37500, 'Dysrhythmia': 39557, 'Myocardial infarction': 38951, 'Cardiomyopathy': 37659}

	unIMFs = open('./Problem_Data/unIMFs.csv','a')
	IMF1_train = open(csv_folder+'IMF1_train.csv', 'a')
	IMF2_train = open(csv_folder+'IMF2_train.csv', 'a')
	IMF1_test = open(csv_folder+'IMF1_test.csv', 'a')
	IMF2_test = open(csv_folder+'IMF2_test.csv', 'a')
	Train_time = open('Train_time.csv','a')
	Test_time = open('Test_time.csv','a')

	if number_of_IMFs >= 3:
		IMF3_train = open(csv_folder+'IMF3_train.csv', 'a')
		IMF3_test = open(csv_folder+'IMF3_test.csv', 'a')
	if number_of_IMFs >= 4:
		IMF4_train = open(csv_folder+'IMF4_train.csv', 'a')
		IMF4_test = open(csv_folder+'IMF4_test.csv', 'a')
	if number_of_IMFs >= 5:
		IMF5_train = open(csv_folder+'IMF5_train.csv', 'a')
		IMF5_test = open(csv_folder+'IMF5_test.csv', 'a')
	if number_of_IMFs == 6:
		IMF6_train = open(csv_folder+'IMF6_train.csv', 'a')
		IMF6_test = open(csv_folder+'IMF6_test.csv', 'a')


	f = open(patient_data)
	line = f.readline()
	disease_array=[]
	file_count = 0
	while line:
		file_count += 1
		if file_count < 1000:
			line = f.readline()
			file_count += 1
			print line, file_count
		else:
			file_count += 1
			splitted = line.split('/')
			file_name = str(splitted[1][0:8])
			patient_folder = str(splitted[0])

			total_path = filepath+patient_folder+'/'+file_name

			print patient_folder,'---',file_name,
			#print total_path

			try:
				signal,ecgrecord = wfdb.srdsamp(total_path)
				record = wfdb.rdsamp(total_path)
				print ecgrecord['comments'][4][22:],

				signal_length = len(signal)
				#repetition = int(math.floor(signal_length/samplenumber))

				if not ecgrecord['comments'][4][22:] == 'n/a':

					disease = ecgrecord['comments'][4][22:]
					if disease in miscle:
						disease = "Miscellaneous"
					elif disease in cardiom:
						disease = "Cardiomyopathy"

					if disease == 'Myocardial infarction':
						overlap = 1000
					elif disease == "Bundle branch block":
						overlap = 55
					elif disease == "Cardiomyopathy":
						overlap = 55
					elif disease == "Dysrhythmia":
						overlap = 35
					elif disease == "Healthy control":
						overlap = 255
					elif disease == "Myocarditis":
						overlap = 15
					elif disease == "Valvular heart disease":
						overlap = 15

					if disease not in disease_array:
						disease_array.append(disease)

					samplelength = 0
					undecomposed = 0
					sig_start_ov = 0
					repetition = 0
					while(signal_length-sig_start_ov >= samplenumber):
						repetition += 1
						sig_start_ov += overlap
					stop = int(math.ceil(repetition*split_perc))
					print 'repetition = ',repetition
					###########    Trining and Test Data Spliting   ######################
					#Training data prepare
					for j in range(0,stop):
						write_signal = []
						for sample in range(samplelength,samplelength+samplenumber):
							ecg_signal = 0
							for i1 in range(0,15):
								ecg_signal = ecg_signal+signal[sample][i1]
							write_signal.append(ecg_signal)

						EMD_signal = np.asarray(write_signal)

						try:
							start_time_train = time.time()
							decomposer = EMD(EMD_signal,n_imfs=number_of_IMFs,maxiter=3000)
							imfs = decomposer.decompose()
							#Construct Modified EMD
							modified_EMD_train = []
							for q in range(0,samplenumber):
								modified_EMD_train.append(imfs[0][q]+imfs[1][q]+imfs[2][q])
							elapsed_time_train = time.time() - start_time_train
							Train_time.write(total_path+','+disease+','+str(elapsed_time_train)+'\n')
							#print len(imfs)

							str1 = str(imfs[0][0])
							str2 = str(imfs[1][0])

							if (len(imfs) == number_of_IMFs+1):
								for h in range(1,samplenumber):
									str1 = str1+','+str(imfs[0][h])
									str2 = str2+','+str(imfs[1][h])

								str1 = str1+','+disease+'\n'
								str2 = str2+','+disease+'\n'

								IMF1_train.write(str1)
								IMF2_train.write(str2)

								if number_of_IMFs >= 3:
									str3 = str(imfs[2][0])
									for h in range(1,samplenumber):
										str3 = str3+','+str(imfs[2][h])
									str3 = str3+','+disease+'\n'
									IMF3_train.write(str3)
								if number_of_IMFs >= 4:
									str4 = str(imfs[3][0])
									for h in range(1,samplenumber):
										str4 = str4+','+str(imfs[3][h])
									str4 = str4+','+disease+'\n'
									IMF4_train.write(str4)
								if number_of_IMFs >= 5:
									str5 = str(imfs[4][0])
									for h in range(1,samplenumber):
										str5 = str5+','+str(imfs[4][h])
									str5 = str5+','+disease+'\n'
									IMF5_train.write(str5)
								if number_of_IMFs==6:
									str6 = str(imfs[5][0])
									for h in range(1,samplenumber):
										str6 = str6+','+str(imfs[5][h])
									str6 = str6+','+disease+'\n'
									IMF6_train.write(str6)
							else:
								print ('IMF Number do not match')
								undecomposed = undecomposed + 1

							samplelength = samplelength+overlap

						except:
							print 'Could not be decomposed'
							samplelength = samplelength+overlap

					#Testing data preparation
					for j in range(stop,repetition):
						write_signal = []
						for sample in range(samplelength,samplelength+samplenumber):
							ecg_signal = 0
							for i1 in range(0,15):
								ecg_signal = ecg_signal+signal[sample][i1]
							write_signal.append(ecg_signal)

						EMD_signal = np.asarray(write_signal)

						try:
							start_time_test = time.time()
							decomposer = EMD(EMD_signal,n_imfs=number_of_IMFs,maxiter=3000)
							imfs = decomposer.decompose()
							#Construct Modified EMD
							modified_EMD_test = []
							for q in range(0,samplenumber):
								modified_EMD_test.append(imfs[0][q]+imfs[1][q]+imfs[2][q])
							elapsed_time_test = time.time() - start_time_test
							Test_time.write(total_path+','+disease+','+str(elapsed_time_test)+'\n')
							#print len(imfs)

							str1 = str(imfs[0][0])
							str2 = str(imfs[1][0])

							if (len(imfs) == number_of_IMFs+1):
								for h in range(1,samplenumber):
									str1 = str1+','+str(imfs[0][h])
									str2 = str2+','+str(imfs[1][h])

								str1 = str1+','+disease+'\n'
								str2 = str2+','+disease+'\n'

								IMF1_test.write(str1)
								IMF2_test.write(str2)

								if number_of_IMFs >= 3:
									str3 = str(imfs[2][0])
									for h in range(1,samplenumber):
										str3 = str3+','+str(imfs[2][h])
									str3 = str3+','+disease+'\n'
									IMF3_test.write(str3)
								if number_of_IMFs >= 4:
									str4 = str(imfs[3][0])
									for h in range(1,samplenumber):
										str4 = str4+','+str(imfs[3][h])
									str4 = str4+','+disease+'\n'
									IMF4_test.write(str4)
								if number_of_IMFs >= 5:
									str5 = str(imfs[4][0])
									for h in range(1,samplenumber):
										str5 = str5+','+str(imfs[4][h])
									str5 = str5+','+disease+'\n'
									IMF5_test.write(str5)
								if number_of_IMFs==6:
									str6 = str(imfs[5][0])
									for h in range(1,samplenumber):
										str6 = str6+','+str(imfs[5][h])
									str6 = str6+','+disease+'\n'
									IMF6_test.write(str6)
							else:
								print ('IMF Number do not match')
								undecomposed = undecomposed + 1

							samplelength = samplelength+overlap

						except:
							print 'Could not be decomposed'
							samplelength = samplelength+overlap

				string = patient_folder+'---'+file_name+'UNIMFed Records = '+str(undecomposed)+'\n'
				unIMFs.write(string)
				line = f.readline()

			except:
				problem=patient_folder+'/'+file_name+'\n'
				problem_data.write(problem)
				line = f.readline()
				print sys.exc_info(),'\n'

	f.close()
	problem_data.close()
	print disease_array

	IMF1_train.close()
	IMF2_train.close()
	IMF1_test.close()
	IMF2_test.close()

	if number_of_IMFs>=3:
		IMF3_train.close()
		IMF3_test.close()
	if number_of_IMFs>=4:
		IMF4_train.close()
		IMF4_test.close()
	if number_of_IMFs>=5:
		IMF5_train.close()
		IMF5_test.close()
	if number_of_IMFs==6:
		IMF6_train.close()
		IMF6_test.close()
	unIMFs.close()

def Main():

	parser = argparse.ArgumentParser(description='ECG data training using EMD Data with separate threading',
									usage='Classifying EMD Data',
									epilog='Give proper arguments')
	parser.add_argument('-p',"--data_path",metavar='', help="Path to the main database",default=C.data_path)
	parser.add_argument('-c',"--csv_path",metavar='',help="Path to the CSV Folder of EMD Data",default=C.IMF_csv_path)
 Training",default=C.initial_epoch)
	parser.add_argument('-rc',"--patient_data_path",metavar='',help="Path to the Patient file RECORD.txt",default=C.patient_data_path)
	parser.add_argument('-pd',"--problem_data_path",metavar='',help="Path to the text file where problematic data to be stored",default=C.preoblem_data_path)
	parser.add_argument('-s',"--sample_number",metavar='',help="Number of samples to be taken by each record",type=int,default=C.samplenumber)
	parser.add_argument('-imf',"--number_of_IMFs",metavar='',help="Number of IMFs to be extracted",default=C.number_of_IMFs,type=int,choices=[2,3,4,5,6])
	parser.add_argument('-spl',"--split_perc",metavar='',help="Splitting percentage of train and test(upper limit)",type=float,default=C.split_perc)

	args = parser.parse_args()

	file_path=args.data_path
	csv_folder=args.csv_path
	patient_data=args.patient_data_path
	problem_data_file=args.problem_data_path
	samplenumber=int(args.sample_number)
	number_of_IMFs=int(args.number_of_IMFs)
	spl_perc = float(args.split_perc)

	EMD_data_preparation(filepath,patient_data,csv_folder,problem_data_file,samplenumber,number_of_IMFs,spl_perc)

Main()
