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

global C,diseases,IMF_1Q,IMF_2Q,IMF_3Q,IMF_4Q,IMF_5Q,IMF_6Q,number_test_image,finished,q1,q2,q3,q4,q5,q6,true_classes
C = config.Config()
IMF_1Q = Queue(maxsize = 500)
IMF_2Q = Queue(maxsize = 500)
IMF_3Q = Queue(maxsize = 500)
IMF_4Q = Queue(maxsize = 500)
IMF_5Q = Queue(maxsize = 500)
IMF_6Q = Queue(maxsize = 500)

q1=[]
q2=[]
q3=[]
q4=[]
q5=[]
q6=[]

'''
diseases={'Myocardial infarction':3,
			'Healthy control':2,
			'Bundle branch block':0,
			'Dysrhythmia':1,
			'Valvular heart disease':5,
			'Myocarditis':4}
'''
'''
#For MIT-BIH
diseases={'SBR':3,
			'N':1,
			'P':2,
			'AFIB':0}
'''

diseases = {'AV Nodal Block':0,'Acute MI':1, 'Atrial Fibrilation':2,'Coronary artery disease':3,'Earlier MI':4, 'Healthy':5,'Sinus node dysfunction':6,'Transient ischemic attack':7,'WPW':8}


global IMF_combination
global IMF_comb_list
global mapped_IMF_pred 
global fo
fo = "./Outputs-Alexnet-Individual-Saint-Petesberg/"
IMF_combination = {}
IMF_comb_list = []


def IMF_Combinations(labeled_outputs,tr):
	#Try different Combinations
	global IMF_combination,IMF_comb_list,mapped_IMF_pred,fo
	
	v1 = labeled_outputs[0]
	v2 = labeled_outputs[1]
	v3 = labeled_outputs[2]
	v4 = labeled_outputs[3]
	v5 = labeled_outputs[4]
	v6 = labeled_outputs[5]
	
	mapped_IMF_pred = {'1':v1,'2':v2,'3':v3,'4':v4,'5':v5,'6':v6}

	IMF_combination[str(IMF_comb_list[0])].append(1) if (v1 == tr) else IMF_combination[str(IMF_comb_list[0])].append(0)
	IMF_combination[str(IMF_comb_list[1])].append(1) if (v2 == tr) else IMF_combination[str(IMF_comb_list[1])].append(0)
	IMF_combination[str(IMF_comb_list[2])].append(1) if (v3 == tr) else IMF_combination[str(IMF_comb_list[2])].append(0)
	IMF_combination[str(IMF_comb_list[3])].append(1) if (v4 == tr) else IMF_combination[str(IMF_comb_list[3])].append(0)
	IMF_combination[str(IMF_comb_list[4])].append(1) if (v5 == tr) else IMF_combination[str(IMF_comb_list[4])].append(0)
	IMF_combination[str(IMF_comb_list[5])].append(1) if (v6 == tr) else IMF_combination[str(IMF_comb_list[5])].append(0)
	
	for k in range(0,len(IMF_comb_list)):
		if IMF_comb_list[k] in [1,2,3,4,5,6]:
			IMF_combination[str(IMF_comb_list[k])].append(1) if mapped_IMF_pred[str(IMF_comb_list[k])] == tr else IMF_combination[str(IMF_comb_list[k])].append(0)
		elif len(IMF_comb_list[k]) == 2:
			IMF_combination[str(IMF_comb_list[k])].append(1) if ([mapped_IMF_pred[str(IMF_comb_list[k][0])],mapped_IMF_pred[str(IMF_comb_list[k][1])]].count(tr)== 2) else IMF_combination[str(IMF_comb_list[k])].append(0)
		elif len(IMF_comb_list[k]) == 3:
			IMF_combination[str(IMF_comb_list[k])].append(1) if ([mapped_IMF_pred[str(IMF_comb_list[k][0])],mapped_IMF_pred[str(IMF_comb_list[k][1])],mapped_IMF_pred[str(IMF_comb_list[k][2])]].count(tr)>= 2) else IMF_combination[str(IMF_comb_list[k])].append(0)
		elif len(IMF_comb_list[k]) == 4:
			IMF_combination[str(IMF_comb_list[k])].append(1) if ([mapped_IMF_pred[str(IMF_comb_list[k][0])],mapped_IMF_pred[str(IMF_comb_list[k][1])],mapped_IMF_pred[str(IMF_comb_list[k][2])],mapped_IMF_pred[str(IMF_comb_list[k][3])]].count(tr)>= 3) else IMF_combination[str(IMF_comb_list[k])].append(0)
		elif len(IMF_comb_list[k]) == 5:
			IMF_combination[str(IMF_comb_list[k])].append(1) if ([mapped_IMF_pred[str(IMF_comb_list[k][0])],mapped_IMF_pred[str(IMF_comb_list[k][1])],mapped_IMF_pred[str(IMF_comb_list[k][2])],mapped_IMF_pred[str(IMF_comb_list[k][3])],mapped_IMF_pred[str(IMF_comb_list[k][4])]].count(tr)>= 3) else IMF_combination[str(IMF_comb_list[k])].append(0)
		elif len(IMF_comb_list[k]) == 6:
			IMF_combination[str(IMF_comb_list[k])].append(1) if ([mapped_IMF_pred[str(IMF_comb_list[k][0])],mapped_IMF_pred[str(IMF_comb_list[k][1])],mapped_IMF_pred[str(IMF_comb_list[k][2])],mapped_IMF_pred[str(IMF_comb_list[k][3])],mapped_IMF_pred[str(IMF_comb_list[k][4])],mapped_IMF_pred[str(IMF_comb_list[k][5])]].count(tr)>= 4) else IMF_combination[str(IMF_comb_list[k])].append(0)

#################################################################################################################################

def result_analysis(number_test_image):
	global IMF_combination,IMF_comb_list,fo
	
	detailed_output1 = open(fo+'Test_Result/detailed_prob/detailed_probability1.txt','w')
	detailed_output2 = open(fo+'Test_Result/detailed_prob/detailed_probability2.txt','w')
	detailed_output3 = open(fo+'Test_Result/detailed_prob/detailed_probability3.txt','w')
	detailed_output4 = open(fo+'Test_Result/detailed_prob/detailed_probability4.txt','w')
	detailed_output5 = open(fo+'Test_Result/detailed_prob/detailed_probability5.txt','w')
	detailed_output6 = open(fo+'Test_Result/detailed_prob/detailed_probability6.txt','w')

	max_output1 = open(fo+'Test_Result/Maximum_Prob/maximum_probability1.txt','w')
	max_output2 = open(fo+'Test_Result/Maximum_Prob/maximum_probability2.txt','w')
	max_output3 = open(fo+'Test_Result/Maximum_Prob/maximum_probability3.txt','w')
	max_output4 = open(fo+'Test_Result/Maximum_Prob/maximum_probability4.txt','w')
	max_output5 = open(fo+'Test_Result/Maximum_Prob/maximum_probability5.txt','w')
	max_output6 = open(fo+'Test_Result/Maximum_Prob/maximum_probability6.txt','w')

	max_output1.write('#IMFnumber, max_probability, predicted class, original class\n')
	max_output2.write('#IMFnumber, max_probability, predicted class, original class\n')
	max_output3.write('#IMFnumber, max_probability, predicted class, original class\n')
	max_output4.write('#IMFnumber, max_probability, predicted class, original class\n')
	max_output5.write('#IMFnumber, max_probability, predicted class, original class\n')
	max_output6.write('#IMFnumber, max_probability, predicted class, original class\n')

	a = [1,2,3,4,5,6]
	for i in range(2,len(a)+1):
		combinations = list(itertools.combinations(a, i))
		IMF_comb_list = IMF_comb_list + combinations
	IMF_comb_list = a + IMF_comb_list
	print '\n\n', IMF_comb_list, '\n\n'
	for j in range(0,len(IMF_comb_list)):
		IMF_combination[str(IMF_comb_list[j])] = []

	for i in range(0,number_test_image):
		v1=q1[i][0].index(max(q1[i][0]))
		v2=q2[i][0].index(max(q2[i][0]))
		if C.number_of_IMFs >= 3:
			v3=q3[i][0].index(max(q3[i][0]))
		if C.number_of_IMFs >= 4:
			v4=q4[i][0].index(max(q4[i][0]))
		if C.number_of_IMFs >= 5:
			v5=q5[i][0].index(max(q5[i][0]))
		if C.number_of_IMFs >= 6:
			v6=q6[i][0].index(max(q6[i][0]))

		tr=diseases[str(true_classes[i])]

		string1 = 'IMF_1'
		string2 = 'IMF_2'
		string3 = 'IMF_3'
		string4 = 'IMF_4'
		string5 = 'IMF_5'
		string6 = 'IMF_6'
		for z in range(0,C.classes):
			string1 = string1 + ','+str(q1[i][0][z])
		string1 = string1 +','+str(round(max(q1[i][0]),5))+','+str(tr)+'\n'
		for z in range(0,C.classes):
			string2 = string2 + ','+str(round(q2[i][0][z],5))
		string2 = string2 +','+str(round(max(q2[i][0]),5))+','+str(tr)+'\n'
		for z in range(0,C.classes):
			string3 = string3 + ','+str(round(q3[i][0][z],5))
		string3 = string3 +','+str(round(max(q3[i][0]),5))+','+str(tr)+'\n'
		for z in range(0,C.classes):
			string4 = string4 + ','+str(round(q4[i][0][z],5))
		string4 = string4 +','+str(round(max(q4[i][0]),5))+','+str(tr)+'\n'
		
		for z in range(0,C.classes):
			string5 = string5 + ','+str(round(q5[i][0][z],5))
		string5 = string5 +','+str(round(max(q5[i][0]),5))+','+str(tr)+'\n'
		for z in range(0,C.classes):
			string6 = string6 + ','+str(round(q6[i][0][z],5))
		string6 = string6 +','+str(round(max(q6[i][0]),5))+','+str(tr)+'\n'
		
		detailed_output1.write(string1)
		detailed_output2.write(string2)
		detailed_output3.write(string3)
		detailed_output4.write(string4)
		detailed_output5.write(string5)
		detailed_output6.write(string6)

		string1 = 'IMF_1,'+str(round(max(q1[i][0]),5))+','+str(v1)+','+str(tr)+'\n'
		string2 = 'IMF_2,'+str(round(max(q2[i][0]),5))+','+str(v2)+','+str(tr)+'\n'
		string3 = 'IMF_3,'+str(round(max(q3[i][0]),5))+','+str(v3)+','+str(tr)+'\n'
		string4 = 'IMF_4,'+str(round(max(q4[i][0]),5))+','+str(v4)+','+str(tr)+'\n'
		string5 = 'IMF_5,'+str(round(max(q5[i][0]),5))+','+str(v5)+','+str(tr)+'\n'
		string6 = 'IMF_6,'+str(round(max(q6[i][0]),5))+','+str(v6)+','+str(tr)+'\n'
		max_output1.write(string1)
		max_output2.write(string2)
		max_output3.write(string3)
		max_output4.write(string4)
		max_output5.write(string5)
		max_output6.write(string6)

		labeled_outputs=[v1,v2,v3,v4,v5,v6]
		IMF_Combinations(labeled_outputs,tr)
		
	with open(fo+"Test_Result/Individual_IMF_Test_Result.txt",'w') as f:
		for i in range(0,len(IMF_comb_list)):
			f.write('Accuracy_'+str(IMF_comb_list[i])+'= '+
						str(IMF_combination[str(IMF_comb_list[i])].count(1)/float(len(IMF_combination[str(IMF_comb_list[i])])) * 100 )+'\n')
	
	print('\n\n\nFinished Testing--------Thank You for Using My Code\nNAHIAN IBN HASAN\nDepartment of EEE')
	print('Bangladesh University of Engineering and Technology\nBangladesh\ngmail: nahianhasan1994@gmail.com')
	print('Contact: ')

	detailed_output1.close()
	detailed_output2.close()
	detailed_output3.close()
	detailed_output4.close()
	detailed_output5.close()
	detailed_output6.close()
	max_output1.close()
	max_output2.close()
	max_output3.close()
	max_output4.close()
	max_output5.close()
	max_output6.close()
	
	#CM.Confusion_Matrix_Plot(C.number_of_IMFs)

def Evaluate_Test_IMFs():

	global q1,q2,q3,q4,q5,q6,IMF_1Q,IMF_2Q,IMF_3Q,IMF_4Q,IMF_5Q,IMF_6Q,fo
	
	json_file = open(fo+'Final_Weights/model_IMF_1.json' ,'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model1 = model_from_json(loaded_model_json)
	# load weights into new model
	model1.load_weights(fo+"Training_Records/IMF_1/weights_best_of_IMF_1.hdf5")
	print("Loaded model1 from disk")

	json_file = open(fo+'Final_Weights/model_IMF_2.json' ,'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model2 = model_from_json(loaded_model_json)
	model2.load_weights(fo+"Training_Records/IMF_2/weights_best_of_IMF_2.hdf5")
	print("Loaded model2 from disk")

	model1.compile(loss = 'categorical_crossentropy', optimizer = C.optimizer, metrics = ['accuracy'])
	model2.compile(loss = 'categorical_crossentropy', optimizer = C.optimizer, metrics = ['accuracy'])

	if C.number_of_IMFs >= 3:
		json_file = open(fo+'Final_Weights/model_IMF_3.json' ,'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model3 = model_from_json(loaded_model_json)
		model3.load_weights(fo+"Training_Records/IMF_3/weights_best_of_IMF_3.hdf5")
		print("Loaded model3 from disk")
		model3.compile(loss = 'categorical_crossentropy', optimizer = C.optimizer, metrics = ['accuracy'])
	if C.number_of_IMFs >= 4:
		json_file = open(fo+'Final_Weights/model_IMF_4.json' ,'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model4 = model_from_json(loaded_model_json)
		model4.load_weights(fo+"Training_Records/IMF_4/weights_best_of_IMF_4.hdf5")
		print("Loaded model4 from disk")
		model4.compile(loss = 'categorical_crossentropy', optimizer = C.optimizer, metrics = ['accuracy'])
	if C.number_of_IMFs >= 5:
		json_file = open(fo+'Final_Weights/model_IMF_5.json' ,'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model5 = model_from_json(loaded_model_json)
		model5.load_weights(fo+"Training_Records/IMF_5/weights_best_of_IMF_5.hdf5")
		print("Loaded model5 from disk")
		model5.compile(loss = 'categorical_crossentropy', optimizer = C.optimizer, metrics = ['accuracy'])
	if C.number_of_IMFs >= 6:
		json_file = open(fo+'Final_Weights/model_IMF_6.json' ,'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model6 = model_from_json(loaded_model_json)
		model6.load_weights(fo+"Training_Records/IMF_6/weights_best_of_IMF_6.hdf5")
		print("Loaded model6 from disk")
		model6.compile(loss = 'categorical_crossentropy', optimizer = C.optimizer, metrics = ['accuracy'])

	read_data=0
	track = 0#a variable to indicte the end of test set
	while(1):
		if not IMF_1Q.empty():
			read_data=read_data+1
			print('Test data runnning = {}'.format(read_data))
			q1.append(model1.predict(np.expand_dims((np.array([IMF_1Q.get(),])),axis=2)).tolist())
			q2.append(model2.predict(np.expand_dims((np.array([IMF_2Q.get(),])),axis=2)).tolist())
			if C.number_of_IMFs >= 3:
				q3.append(model3.predict(np.expand_dims((np.array([IMF_3Q.get(),])),axis=2)).tolist())
			if C.number_of_IMFs >= 4:
				q4.append(model4.predict(np.expand_dims((np.array([IMF_4Q.get(),])),axis=2)).tolist())
			if C.number_of_IMFs >= 5:
				q5.append(model5.predict(np.expand_dims((np.array([IMF_5Q.get(),])),axis=2)).tolist())
			if C.number_of_IMFs >= 6:
				q6.append(model6.predict(np.expand_dims((np.array([IMF_6Q.get(),])),axis=2)).tolist())
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


def Main():
	
	deviceIDs=[]
	while not deviceIDs:
		deviceIDs = GPUtil.getAvailable(order='first',limit=1,maxMemory=0.05,maxLoad=0.99)
		print 'searching for GPU to be available. Please wait.....'
	print 'GPU Found...Starting Training\n'
	# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	
	global number_test_image,IMF_1Q,IMF_2Q,IMF_3Q,IMF_4Q,IMF_5Q,IMF_6Q,true_classes
	print('Reading IMF csv data files for testing')
	IMF1_test = open(C.IMF_csv_path+'IMF1_test.csv','r')
	IMF2_test = open(C.IMF_csv_path+'IMF2_test.csv','r')
	line1 = IMF1_test.readline()
	line2 = IMF2_test.readline()

	if C.number_of_IMFs >= 3:
		IMF3_test = open(C.IMF_csv_path+'IMF3_test.csv','r')
		line3 = IMF3_test.readline()
	if C.number_of_IMFs >= 4:
		IMF4_test = open(C.IMF_csv_path+'IMF4_test.csv','r')
		line4 = IMF4_test.readline()
	if C.number_of_IMFs >= 5:
		IMF5_test = open(C.IMF_csv_path+'IMF5_test.csv','r')

		line5 = IMF5_test.readline()
	if C.number_of_IMFs >= 6:
		IMF6_test = open(C.IMF_csv_path+'IMF6_test.csv','r')
		line6 = IMF6_test.readline()

	print("Finished Loading Testing Data")
	print('Reading Models')

	t = threading.Thread(target=Evaluate_Test_IMFs, name='thread1')
	t.start()

	true_classes = []
	number_test_image = 0
	while (line1 and line2 and line3 and line4 and line5 and line6):
		splitted1 = line1.split(',')
		splitted_1 = splitted1[0:C.samplenumber]
		splitted2 = line2.split(',')
		splitted_2 = splitted2[0:C.samplenumber]

		if C.number_of_IMFs >= 3:
			splitted3 = line3.split(',')
			splitted_3 = splitted3[0:C.samplenumber]
		if C.number_of_IMFs >= 4:
			splitted4 = line4.split(',')
			splitted_4 = splitted4[0:C.samplenumber]
		if C.number_of_IMFs >= 5:
			splitted5 = line5.split(',')
			splitted_5 = splitted5[0:C.samplenumber]
		if C.number_of_IMFs >= 6:
			splitted6 = line6.split(',')
			splitted_6 = splitted6[0:C.samplenumber]

		class_name = str(splitted1[C.samplenumber][:-1])
		
		
		if class_name in ['Hypertrophy','Cardiomyopathy']:
			line1 = IMF1_test.readline()
			line2 = IMF2_test.readline()
			line3 = IMF3_test.readline()
			line4 = IMF4_test.readline()
			line5 = IMF5_test.readline()
			line6 = IMF6_test.readline()
			continue
		
		'''
		#for MIT-BIH
		if class_name in ['AFL','B','T','VFL','NOD','VT','IVR']:
			line1 = IMF1_test.readline()
			line2 = IMF2_test.readline()
			line3 = IMF3_test.readline()
			line4 = IMF4_test.readline()
			line5 = IMF5_test.readline()
			line6 = IMF6_test.readline()
			continue
		'''
		true_classes.append(class_name)
		

		try:
			IMF_1Q.put(splitted_1)
			IMF_2Q.put(splitted_2)
			if C.number_of_IMFs >= 3:
				IMF_3Q.put(splitted_3)
			if C.number_of_IMFs >= 4:
				IMF_4Q.put(splitted_4)
			if C.number_of_IMFs >= 5:
				IMF_5Q.put(splitted_5)
			if C.number_of_IMFs >= 6:
				IMF_6Q.put(splitted_6)

			number_test_image = number_test_image+1
			
			line1 = IMF1_test.readline()
			line2 = IMF2_test.readline()
			line3 = IMF3_test.readline()
			line4 = IMF4_test.readline()
			line5 = IMF5_test.readline()
			line6 = IMF6_test.readline()


		except:
			print sys.exc_info(),'\n'
			line1 = IMF1_test.readline()
			line2 = IMF2_test.readline()
			line3 = IMF3_test.readline()
			line4 = IMF4_test.readline()
			line5 = IMF5_test.readline()
			line6 = IMF6_test.readline()

#if __name__ == '__Main__':
Main()
