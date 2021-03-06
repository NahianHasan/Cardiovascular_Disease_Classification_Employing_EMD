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
import config
import EMD_Models as M
from Queue import Queue
import threading
import tensorflow as tf
import time
from keras.models import model_from_json
import pickle
import Confusion_Matrix_Parallel as CM

global C,diseases,IMF_1Q,IMF_2Q,IMF_4Q,number_test_image,finished,q,true_classes
C = config.Config()
IMF_1Q = Queue(maxsize = 5000)
IMF_2Q = Queue(maxsize = 5000)
IMF_4Q = Queue(maxsize = 5000)


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
'''

#For MIT-BIH
diseases={'SBR':3,
           'N':1,
           'P':2,
           'AFIB':0}

diseases = {'AV Nodal Block':0,'Acute MI':1, 'Atrial Fibrilation':2,'Coronary artery disease':3,'Earlier MI':4, 'Healthy':5,'Sinus node dysfunction':6,'Transient ischemic attack':7,'WPW':8}
'''
fo = './Outputs-parallel-PTB-7-class-IMF123'
def result_analysis(number_test_image):
	result=[]
	detailed_output = open(fo+'/Test_Result/detailed_prob/detailed_parallel_probability_AlexNet_test.txt','w')
	max_output = open(fo+'/Test_Result/Maximum_Prob/maximum_parallel_probability_AlexNet_test.txt','w')
	max_output.write('#IMFnumber, max_probability, predicted class, original class\n')

	for i in range(0,number_test_image):
		v=q[i][0].index(max(q[i][0]))
		tr=diseases[str(true_classes[i])]

		string = 'IMF_'
		for z in range(0,C.classes):
			string = string + ','+str(q[i][0][z])
		string = string +','+str(round(max(q[i][0]),5))+','+str(tr)+'\n'
		detailed_output.write(string)

		string = 'IMF_,'+str(round(max(q[i][0]),5))+','+str(v)+','+str(tr)+'\n'
		max_output.write(string)

		if(v == tr):
			result.append(1)
		else:
			result.append(0)

	#save it
	with open(fo+"/Test_Result/result_parallel_AlexNet_test.pickle","wb") as f:
		pickle.dump(result,f)
	#result list has the number 1 indices for which match occur and 0 indices for which match do not occur
	#Calculate the mean, accuracy, std etc
	result_1 = result.count(1)
	result_length = len(result)
	accuracy = (result_1/float(result_length))*100
	with open(fo+"/Test_Result/result_parallel_AlexNet_test.txt",'w') as f:
		string = 'Accuracy = '+str(accuracy)
		f.write(string)

	print('\nTotal Percentage of match = {}'.format(accuracy))
	print('\n\n\nFinished Testing--------Thank You for Using My Code\nNAHIAN IBN HASAN\nDepartment of EEE')
	print('Bangladesh University of Engineering and Technology\nBangladesh\ngmail: nahianhasan1994@gmail.com')
	print('LinkedIn Profile: ')

	detailed_output.close()
	max_output.close()
	
	#CM.Confusion_Matrix_Plot()

def Evaluate_Test_IMFs():

	global q,IMF_1Q,IMF_2Q,IMF_4Q
	'''	
	deviceIDs=[]
	while not deviceIDs:
		deviceIDs = GPUtil.getAvailable(order='first',limit=1,maxMemory=0.80,maxLoad=0.99)
		print 'searching for GPU to be available. Please wait.....'
	print 'GPU Found...Starting Training\n'
	# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.20)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	'''
	json_file = open(fo +'/Final_Weights/model_parallel_3_IMF.json' ,'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	# load weights into new model
	weight_file = fo+"/Training_Records/weights_parallel_best_.hdf5"
	model.load_weights(weight_file)
	print "\n\nLoaded ",weight_file," from disk\n\n"
	model.compile(loss = 'categorical_crossentropy', optimizer = C.optimizer, metrics = ['accuracy'])

	read_data=0
	track = 0#a variable to indicte the end of test set
	while(1):
		if not IMF_1Q.empty():
			try:
				read_data=read_data+1
				print('Test data runnning = {}'.format(read_data))
				arr_1 = np.expand_dims((np.array([IMF_1Q.get(),])),axis=2)
				arr_2 = np.expand_dims((np.array([IMF_2Q.get(),])),axis=2)
				arr_3 = np.expand_dims((np.array([IMF_4Q.get(),])),axis=2)
			
				q.append(model.predict([arr_1,arr_2,arr_3]).tolist())
				track=0
			except:
				print 'Could not get image from the queue'
 				continue
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


def Main():
	global number_test_image,IMF_1Q,IMF_2Q,IMF_4Q,true_classes
	print('Reading IMF csv data files for testing')
	IMF1_test = open(C.IMF_csv_path+'IMF1_test.csv','r')
	IMF2_test = open(C.IMF_csv_path+'IMF2_test.csv','r')
	line1 = IMF1_test.readline()
	line2 = IMF2_test.readline()

	#if C.number_of_IMFs >= 3:
	IMF4_test = open(C.IMF_csv_path+'IMF3_test.csv','r')
	line3 = IMF4_test.readline()

	print("Finished Loading Testing Data")
	print('Reading Models')

	t = threading.Thread(target=Evaluate_Test_IMFs, name='thread1')
	t.start()

	true_classes = []
	number_test_image = 0
	while (line1 and line2 and line3):
		splitted1 = line1.split(',')
		splitted_1 = splitted1[0:C.samplenumber]
		splitted2 = line2.split(',')
		splitted_2 = splitted2[0:C.samplenumber]

		if C.number_of_IMFs >= 3:
			splitted3 = line3.split(',')
			splitted_3 = splitted3[0:C.samplenumber]

		class_name = str(splitted1[C.samplenumber][:-1])
		
		if class_name in ['Hypertrophy','Miscellaneous']:
			line1 = IMF1_test.readline()
			line2 = IMF2_test.readline()
			line3 = IMF4_test.readline()
			continue
		'''
		
		if class_name in ['AFL','B','T','VFL','NOD','VT','IVR']:
		   line1 = IMF1_test.readline()
		   line2 = IMF2_test.readline()
		   line3 = IMF3_test.readline()
		   continue
		'''
		
		true_classes.append(class_name)

		splitted1 = np.asarray(splitted_1)
		splitted2 = np.asarray(splitted_2)
		splitted3 = np.asarray(splitted_3)

		try:
			IMF_1Q.put(splitted1)
			IMF_2Q.put(splitted2)
			#if C.number_of_IMFs >= 3:
			IMF_4Q.put(splitted3)

			number_test_image = number_test_image+1	
			
			#print('In the queue so far = {}  ,  '.format(number_test_image),)
			line1 = IMF1_test.readline()
			line2 = IMF2_test.readline()
			line3 = IMF4_test.readline()

		except:
			#print sys.exc_info(),'\n'
			line1 = IMF1_test.readline()
			line2 = IMF2_test.readline()
			line3 = IMF4_test.readline()

	#Check whether the testing  has been completed
	
#if __name__ == '__Main__':
Main()
