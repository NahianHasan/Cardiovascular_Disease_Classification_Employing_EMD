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

def EMD_data_preparation(csv_folder,samplenumber,test_list):
	###########    Trining and Test Data Spliting   ######################
	Ensembled_test = open(csv_folder+'Ensembled_test.csv', 'w')
	Total_data = 0
		
	#Testing data preparation
	G = open(test_list,'r')
	line = G.readline()
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
			Ensembled_test.write(string)

			print 'Test Data = ',Total_data,'---Disease = ',disease
			line = G.readline()
		except:
			print 'Could not write'
			line = G.readline()


	#Ensembled_train.close()
	Ensembled_test.close()
	#F.close()
	G.close()
	
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
	parser.add_argument('-tel',"--te_list",metavar='',help="A csv file containing the list of test files")

	args = parser.parse_args()

	file_path=args.data_path
	csv_path=args.csv_path
	test_list = args.te_list
	patient_data=args.patient_data_path
	problem_data_file=args.problem_data_path
	samplenumber=int(args.sample_number)
	number_of_IMFs=int(args.number_of_IMFs)


	EMD_data_preparation(csv_path,samplenumber,test_list)
Main()
