import pyhht
from pyhht.emd import EMD
from pyhht.visualization import plot_imfs
import numpy as np
import math
import pandas
from pyhht.utils import extr
from pyhht.utils import get_envelops
import matplotlib.pyplot as plt
from pyhht.utils import inst_freq
import wfdb
import os
import sys
import glob
import config
import GPUtil
import EMD_Models as M
from Queue import Queue
import threading
import time
import itertools
from keras.models import model_from_json
import pickle
import Confusion_Matrix as CM
import tensorflow as tf

global C,diseases,Ensembled_Q,number_test_image,finished,q,true_classes,X_prim,Y_prim
C = config.Config()

Ensembled_Q = Queue(maxsize = 500)

q=[]

'''
diseases={'Myocardial infarction':5,
			'Healthy control':3,
			'Cardiomyopathy':1,
			'Bundle branch block':0,
			'Dysrhythmia':2,
			'Hypertrophy':4,
			'Valvular heart disease':7,
			'Myocarditis':6}
'''

diseases={'Myocardial infarction':4,
                        'Healthy control':3,
                        'Cardiomyopathy':1,
                        'Bundle branch block':0,
                        'Dysrhythmia':2,
                        'Valvular heart disease':6,
                        'Myocarditis':5}

#diseases = {'AF':0,'Noisy':1,'Normal':2,'Other':3}

'''
diseases={'AV Nodal Block':0,'Acute MI':1, 'Atrial Fibrilation':2,'Coronary artery disease':3,'Earlier MI':4, 'Healthy':5,'Sinus node dysfunction':6,'Transient ischemic attack':7,'WPW':8}
'''

#diseases = {'AFIB':0,'N':1,'P':2,'SBR':3}

global mapped_IMF_pred
global fo

#################################################################################################################################

def result_analysis(number_test_image):
	global fo
	result=[]
	detailed_output = open(fo+'Test_Result/detailed_prob/detailed_probability.txt','w')
	max_output = open(fo+'Test_Result/Maximum_Prob/maximum_probability.txt','w')
	max_output.write('#IMFnumber, max_probability, predicted class, Ensembled class\n')

	for i in range(0,number_test_image):
		v=q[i][0].index(max(q[i][0]))

		tr=diseases[str(true_classes[i])]

		string = 'Ensembled'

		for z in range(0,len(diseases)):
			string = string + ','+str(q[i][0][z])
		string = string +','+str(round(max(q[i][0]),5))+','+str(tr)+'\n'


		detailed_output.write(string)
		string = 'Ensembled,'+str(round(max(q[i][0]),5))+','+str(v)+','+str(tr)+'\n'
		max_output.write(string)

		if(v == tr):
			result.append(1)
		else:
			result.append(0)

	#save it
	with open(fo+"Test_Result/result_all_combinations_test.pickle","wb") as f:
		pickle.dump(result,f)
	#result list has the number 1 indices for which match occur and 0 indices for which match do not occur
	#Calculate the mean, accuracy, std etc
	result_1 = result.count(1)
	result_length = len(result)
	accuracy = (result_1/float(result_length))*100
	with open(fo+"Test_Result/result_all_combinations_test.txt",'w') as f:
		string = 'Accuracy = '+str(accuracy)
		f.write(string)

	print 'Accuracy = ',str(accuracy),'\n\n'
	print('\n\n\nFinished Testing--------Thank You for Using My Code\nNAHIAN IBN HASAN\nDepartment of EEE')
	print('Bangladesh University of Engineering and Technology\nBangladesh\ngmail: nahianhasan1994@gmail.com')
	print('Contact: ')

	detailed_output.close()
	max_output.close()

	#CM.Confusion_Matrix_Plot(C.number_of_IMFs)

def Evaluate_Test_IMFs():

	global q,Ensembled_Q,fo
	
	deviceIDs=[]
	'''
	while not deviceIDs:
		deviceIDs = GPUtil.getAvailable(order='first',limit=1,maxMemory=0.65,maxLoad=0.99)
		print 'searching for GPU to be available. Please wait.....'
	'''
	print 'GPU Found...Starting Training\n'
	# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.47)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	
	json_file = open(fo+'Final_Weights/model_all_combinations.json' ,'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	# load weights into new model
	model.load_weights(fo+"Training_Records/weights_best_of_.hdf5")
	print("Loaded model from disk")

	model.compile(loss = 'categorical_crossentropy', optimizer = C.optimizer, metrics = ['accuracy'])

	read_data=0
	track = 0#a variable to indicte the end of test set
	while(1):
		if not Ensembled_Q.empty():
			read_data=read_data+1
			print('Test data runnning = {}'.format(read_data)),'\r',
			q.append(model.predict(np.expand_dims((np.array([Ensembled_Q.get(),])),axis=2)).tolist())
			track=0
		else:
			track=track+1
			if track==2000:
				break
			continue

	print ('Finished Testing')
	#Now q1,q2,q3...q6 have the predicted results containing the probabilities of each class for each IMF
	#For example q1[0] has 9(number of classes) probabilities
	#get the index of highest probability and match the index with that of disease indexes above after label encoder

	result_analysis(number_test_image)
def Total_Data_Load(csv_folder,IMF_number,samplenumber):
	global X_prim,Y_prim
	csv_path = {}
	for i in IMF_number:
		csv_path[str(i)] = csv_folder+'IMF'+str(i)+'_test'+'.csv'
	#for original data training
	#csv_path = csv_folder+'Original_train.csv'
	classes = C.disease_names
	X_prim = {}
	Y_prim = {}
	for i in IMF_number:
		X_prim[str(i)] = []
	for i in IMF_number:
		dataframe = pandas.read_csv(csv_path[str(i)], header=None)
		dataset = dataframe.values
		if i==6:
			X_prim[str(i)] = dataset[:-1,0:samplenumber].astype(float)
		else:
			X_prim[str(i)] = dataset[:-2,0:samplenumber].astype(float)
			Y_prim = dataset[:-2,samplenumber]
		print 'IMF ',i,' is loaded'
		print len(X_prim[str(i)])
	print 'All Data Loaded\n\n\n'
	print len(Y_prim)

def Test_Data_Load(IMF_number,classes,total_training_data_length,samplenumber):
	global X_prim,Y_prim
	print 'Test Data is being generated'

	X_modified = []
	Y_modified = []
	line_count = 0
	data_number = 0
	for i in range(0,len(X_prim[str(IMF_number[0])])):
		my_sum = np.zeros(samplenumber)
		if Y_prim[i] not in ['Miscellaneous','Hypertrophy','n/a']:
			for j in IMF_number:
				my_sum = [a + b for a, b in zip(my_sum, X_prim[str(j)][i,:])]
			X_modified.append(my_sum)
			Y_modified.append(Y_prim[i])
			data_number +=1
			print 'Total Test Data Loaded = ',data_number,'\r',


	X_test = np.asarray(X_modified)
	Y_test = Y_modified
	return X_test,Y_test

def Main():
	global number_test_image,Ensembled_Q,true_classes,X_prim,Y_prim,fo
	print('Reading IMF csv data files for testing')
	
	IMF_number = [1,2,3,4,5,6]
	print IMF_number
	fo = "./Outputs-all-combinations-"
	for t in range(0,len(IMF_number)):
		fo = fo+str(IMF_number[t])
	fo=fo+'/'
	print fo
	samplenumber = C.samplenumber
	classes = C.disease_names
	Total_Data_Load(C.IMF_csv_path,IMF_number,samplenumber)
	X_test,Y_test = Test_Data_Load(IMF_number,classes,len(X_prim[str(IMF_number[0])]),samplenumber)
	print("Finished Loading Testing Data")
	print('Reading Models')

	t = threading.Thread(target=Evaluate_Test_IMFs, name='thread1')
	t.start()

	true_classes = []
	number_test_image = 0
	for i in range(0,len(Y_test)):
		try:
			if Y_test[i] not in ['Miscellaneous','Hypertrophy','n/a']:
				class_name = Y_test[i]
				true_classes.append(class_name)
				Ensembled_Q.put(X_test[i])
				number_test_image = number_test_image + 1
		except:
			print 'Problem loading data to queue\n'
				

#if __name__ == '__Main__':
Main()
