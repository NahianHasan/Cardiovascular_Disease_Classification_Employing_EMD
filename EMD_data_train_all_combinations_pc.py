#import other files
from keras.layers.core import K
#import EMD_data_prepare as E
import EMD_Models
import config
import Folder_creation as FC
import Training_Analysis as TRA
import Confusion_Matrix as CM
#import other libraries
import wfdb
import os
import sys
import threading
import time
import glob
import argparse
import numpy as np
import pandas
import pickle
import math
import random

#Random seed value set
from numpy.random import seed
seed(2)
from tensorflow import set_random_seed
set_random_seed(2)
random.seed(2)
import os
os.environ['PYTHONHASHSEED']=str(2)

from collections import Counter
import matplotlib.pyplot as plt
#import keras libraries
from keras.utils import np_utils
from keras.callbacks import CSVLogger
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm
from keras.models import model_from_json
from keras.callbacks import LearningRateScheduler,EarlyStopping
from keras.utils import plot_model
from keras.optimizers import SGD
#import from scikit learn libraries
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight
#####################################################################################################################
global C,Y_val,X_prim,Y_prim
global M
C = config.Config()
M = EMD_Models.MODELS()

def Total_Data_Load(csv_folder,IMF_number,samplenumber):
	global X_prim,Y_prim
	csv_path = {}
	for i in IMF_number:
		csv_path[str(i)] = csv_folder+'IMF'+str(i)+'_train'+'.csv'
	#for original data training
	#csv_path = csv_folder+'Original_train.csv'
	classes = C.disease_names
	X_prim = {}
	for i in IMF_number:
		X_prim[str(i)] = []
	for i in IMF_number:
		dataframe = pandas.read_csv(csv_path[str(i)], header=None)
		dataset = dataframe.values
		X_prim[str(i)] = dataset[:-1,0:samplenumber].astype(float)
		Y_prim = dataset[:-1,samplenumber]
		print 'IMF ',i,' is loaded'
		print len(X_prim[str(i)])
	print 'All Data Loaded\n\n\n'
	print Y_prim
def Valuation_data_load(IMF_number,classes,valuation_samples,samplenumber,batch=128):
	global Y_val,X_prim,Y_prim
	X_modified = []
	Y_modified = []
	print 'Validation Data is being prepared'
	data_number = 0
	for i in range(0,len(X_prim['1'])):
		my_sum = np.zeros(samplenumber)
		if i in Y_val:
			if Y_prim[i] not in ['Miscellaneous','Hypertrophy','n/a']:
				for j in IMF_number:
					my_sum = [a + b for a, b in zip(my_sum, X_prim[str(j)][i,:])]
				X_modified.append(my_sum)
				Y_modified.append(Y_prim[i])
				data_number += 1
				print 'Total Validation Data Loaded = ',data_number,'\r',

	X_valuation = np.asarray(X_modified)
	Y_valuation = Y_modified
	encoder = LabelBinarizer()
	encoder.fit(classes)
	Y_valuation = encoder.transform(Y_valuation)
	if C.CNN_model_use:
		X_valuation = np.expand_dims(X_valuation, axis=2)
	return X_valuation,Y_valuation

def Train_Data_Load(IMF_number,classes,total_training_data_length,samplenumber):
	global Y_val,X_prim,Y_prim
	print 'Train Data is being generated'

	while(1):
			X_modified = []
			Y_modified = []
			line_count = 0
			data_number = 0
			for i in range(0,len(X_prim['1'])):
				my_sum = np.zeros(samplenumber)
				if i not in Y_val:
					if Y_prim[i] not in ['Miscellaneous','Hypertrophy','n/a']:
						for j in IMF_number:
							my_sum = [a + b for a, b in zip(my_sum, X_prim[str(j)][i,:])]
						X_modified.append(my_sum)
						Y_modified.append(Y_prim[i])
						data_number +=1
					line_count += 1
					if line_count==C.batch_size or data_number == total_training_data_length:
					#print 'Total Train Data Loaded = ',data_number,'\r',
						X_train = np.asarray(X_modified)
						Y_train = Y_modified
						encoder = LabelBinarizer()
						encoder.fit(classes)
						Y_train = encoder.transform(Y_train)
						if C.CNN_model_use:
							X_train = np.expand_dims(X_train, axis=2)
						yield X_train,Y_train
						X_modified = []
						Y_modified = []
						line_count = 0


def separate_threads(folder,IMF_number,filepath,patient_data,problem_data,csv_folder,samplenumber,resume,initial_epoch):
	global Y_val,X_prim,Y_prim
	Total_Data_Load(csv_folder,IMF_number,samplenumber)
	samplenumber=samplenumber
	#Valuation Data generation
	Y_val = random.sample(range(0, C.Total_Train_Data), int(math.ceil(0.2*float(C.Total_Train_Data))))

	# fix random seed for reproducibility
	seed = 7
	np.random.seed(seed)

	classes = C.disease_names

	if resume=='False':
		#get the model and print the summary
		model = M.IMF_models[str(1)]()
		if C.optimizer=='sgd':
			sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
			#Compile Model
			model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
		else:
			model.compile(loss = 'categorical_crossentropy', optimizer = C.optimizer, metrics = ['accuracy'])
	elif resume=='True':
		#load_architecture
		json_file = open(folder+'/Final_Weights/model_all_combinations.json','r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		#Load weights
		model_weight_file = folder+'/Training_Records/weights_best_of_.hdf5'
		#filenames =  glob.glob(weight_folder)
		#filenames.sort(reverse = True)
		print '\n\n\n', model_weight_file, '\n\n\n'
		model.load_weights(model_weight_file)
		if C.optimizer=='sgd':
			sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
			#Compile Model
			model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
		else:
			model.compile(loss = 'categorical_crossentropy', optimizer = C.optimizer, metrics = ['accuracy'])

	print model.summary()

	if resume=='False':
	#SAVE THE MODEL Architecture
		model_json = model.to_json()
		mdl_save_path = './'+folder+'/Final_Weights/model_all_combinations.json'
		with open(mdl_save_path, "w") as json_file:
			json_file.write(model_json)

	# learning rate schedule
	def step_decay(epoch):
		initial_lrate = C.initial_lrate
		drop = C.lrate_drop
		epochs_drop = C.lrate_epochs_drop
		lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
		return lrate
	####################    callback list     #######################
	#checkpoint path
	chk_path = './'+folder+'/Training_Records/weights_best_of_.hdf5'
	checkpoint_best = ModelCheckpoint(chk_path,monitor='acc',verbose=1,save_best_only=True,mode=C.chkpointpath_saving_mode)
	#Save Every epoch
	#chk_path = './'+folder+'/Saved_All_Weights/all_comb'+"_-{epoch:02d}-{acc:.2f}.hdf5"
	#each_epoch = ModelCheckpoint(chk_path,monitor='acc',verbose=1,save_best_only=False,mode='auto', period=1)
	# learning schedule callback
	lrate = LearningRateScheduler(step_decay)
	#Early Stopping
	Early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=100, verbose=0, mode='max')
	#Callback that streams epoch results to a csv file.
	csv_logger = CSVLogger('./'+folder+'/Training_CSV_log/training_all_comb.log')
	#Tensorboard visualization
	tensor_board = TensorBoard(log_dir='./'+folder+'/Tensorboard_Visualization/',histogram_freq=0,write_graph=False,write_images=False)
	#open a terminal and write 'tensorboard --logdir=logdir/' and go to the browser
	#################################################################

	callback_list=[checkpoint_best,lrate,csv_logger,Early_stop,tensor_board]
	#Data_Generator
	total_training_data_length = C.Total_Train_Data - len(Y_val)
	valuation_samples = len(Y_val)

	Train_Data_generator = Train_Data_Load(IMF_number,classes,total_training_data_length,samplenumber)
	X_validation,Y_validation = Valuation_data_load(IMF_number,classes,valuation_samples,samplenumber,batch=C.batch_size)
	del X_prim
	del Y_prim
	'''
	#dataset Standardization
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_valuation = scaler.transform(X_valuation)
	'''
	#Fit the model
	if C.grid_search:
		# grid search epochs, batch size and optimizer
		optimizers = ['rmsprop' , 'adam']
		init = ['glorot_uniform' , 'normal' , 'uniform']
		epochs = np.array([50, 100, 150])
		batches = np.array([5, 10, 20])
		param_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches, init=init)
		grid = GridSearchCV(estimator=model, param_grid=param_grid)
		history = grid.fit(Train_Data_generator, nb_epoch=C.nb_epoch,steps_per_epoch = int(math.ceil(total_training_data_length/	C.batch_size)),verbose=2,shuffle=C.shuffle,validation_data=Validation_Data_generator,callbacks = callback_list)
	elif not C.grid_search:
		history = model.fit_generator(Train_Data_generator,validation_data=(X_validation,Y_validation),
						epochs=C.nb_epoch,steps_per_epoch=int(math.ceil(total_training_data_length/	C.batch_size)),verbose=1,shuffle=C.shuffle,callbacks =callback_list, initial_epoch=initial_epoch,nb_val_samples=valuation_samples)

	#save the history of whole training
	filehandler = open('./'+folder+"/Training_History/IMF_all_comb.obj","wb")
	pickle.dump(history.history,filehandler)
	filehandler.close()


#########################################################################################################################

def Main():
	'''
	deviceIDs=[]
	while not deviceIDs:
		deviceIDs = GPUtil.getAvailable(order='first',limit=1,maxMemory=0.80,maxLoad=0.99)
		print 'searching for GPU to be available. Please wait.....'
	print 'GPU Found...Starting Training\n'
	# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	'''
	parser = argparse.ArgumentParser(description='ECG data training using EMD Data with separate threading',
									usage='Classifying EMD Data',
									epilog='Give proper arguments')
	parser.add_argument('-p',"--data_path",metavar='', help="Path to temain database",default=C.data_path)
	parser.add_argument('-c',"--csv_path",metavar='',help="Path to the CSV Folder of EMD Data",default=C.IMF_csv_path)
	parser.add_argument('-res',"--resume_train",metavar='',help="Resume Training",default='False')
	parser.add_argument('-inep',"--ini_epoch",metavar='',help="Initial Epoch after Resuming Training",default=C.initial_epoch)
	parser.add_argument('-reim',"--res_imf",metavar='',help="Resumed IMF number after resuming",default=1)
	parser.add_argument('-rc',"--patient_data_path",metavar='',help="Path to the Patient file RECORD.txt",default=C.patient_data_path)
	parser.add_argument('-pd',"--problem_data_path",metavar='',help="Path to the text file where problematic data to be stored",default=C.preoblem_data_path)
	parser.add_argument('-s',"--sample_number",metavar='',help="Number of samples to be taken by each record",type=int,default=C.samplenumber)
	parser.add_argument('-imf',"--number_of_IMFs",metavar='',help="Number of IMFs to be extracted",default=C.number_of_IMFs,type=int,choices=[2,3,4,5,6])
	parser.add_argument('-spl',"--split_perc",metavar='',help="Splitting percentage of train and test(upper limit)",type=float,default=C.split_perc)
	parser.add_argument('-fold',"--res_fold",metavar='',help="Save training and testing results in folder")

	args = parser.parse_args()

	file_path=args.data_path
	csv_folder=args.csv_path
	patient_data=args.patient_data_path
	problem_data=args.problem_data_path
	samplenumber=int(args.sample_number)
	number_of_IMFs=int(args.number_of_IMFs)
	spl_perc = float(args.split_perc)
	resume = args.resume_train
	resumed_IMF_number = int(args.res_imf)
	initial_epoch = int(args.ini_epoch)
	folder = args.res_fold
	#Check whether specific folders are present or not....if not create them
	FC.Folder_creation(number_of_IMFs,folder)

	#generate EMD separate IMF csv files in the csv path
	if C.EMD_data_prepare is True:
		response = raw_input("Are you sure that you want to prepare the EMD Data Files again(Y/N): ")
		if response == 'Y':
			print('EMD data preparing\n')
			E.EMD_data_preparation(file_path,patient_data,csv_folder,problem_data,samplenumber,number_of_IMFs,spl_perc)
			print('EMD data preparation finished\n')
		elif response == 'N':
			print('Skippng EMD Data Preparation Step')
	elif C.EMD_data_prepare is False:
		response = raw_input("Are you sure that you do not want to prepare the EMD Data Files(Y/N): ")
		if response == 'N':
			print('EMD data preparing\n')
			E.EMD_data_preparation(file_path,patient_data,csv_folder,problem_data,samplenumber,number_of_IMFs,spl_perc)
			print('EMD data preparation finished\n')
		elif response == 'Y':
			print('EMD Data already prepared.So going to training phase of each IMF')

	'''
	for i in range(1,number_of_IMFs+1):
		print('IMF_Thread {} has started'.format(i))
		separate_threads(i,file_path,patient_data,problem_data,csv_folder,samplenumber,resume,initial_epoch)
	'''
	separate_threads(folder,C.IMF_array,file_path,patient_data,problem_data,csv_folder,samplenumber,resume,initial_epoch)

	#Plotting the history of training
	print 'Finished Training all the IMF segments'
	print "Let's see how the training was"

	#Plotting various data
	#CM.Confusion_Matrix_Plot(C.number_of_IMFs)
	#TRA.Plot_Acc_Loss_Function(number_of_IMFs)
	#TRA.IMF_Based_Evaluation_Acc_bar_plot(number_of_IMFs)


#if(__name__ == '__Main__'):
Main()
#Delete all .pyc files
direc = os.getcwd()
test=os.listdir(direc)
for item in test:
	if item.endswith(".pyc"):
		os.remove(item)
