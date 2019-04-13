#import other files
import EMD_data_prepare as E
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
import GPUtil
import pandas
from time import time
import pickle
import math
import random
from collections import Counter
import matplotlib.pyplot as plt
#import keras libraries
import keras.layers.core as K
from keras.utils import np_utils
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler,EarlyStopping,TensorBoard
from keras.utils import plot_model
from keras.models import model_from_json
from Keras_FB import main as fb
from keras.models import load_model
#import from scikit learn libraries
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder,LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.utils import class_weight
import tensorflow as tf
#####################################################################################################################
global C,Y_val
global M
C = config.Config()
M = EMD_Models.MODELS()

def separate_threads(folder,IMF_number,filepath,patient_data,problem_data,csv_folder,samplenumber,resume,initial_epoch):
	print ('IMF {} is training'.format(IMF_number))
	samplenumber=samplenumber
	# fix random seed for reproducibility
	seed = 7
	np.random.seed(seed)
	#itterate through csv file
	#for emd based training
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
		X_prim[str(i)] = dataset[:,0:samplenumber].astype(float)
		Y_prim = dataset[:,samplenumber]
		print 'IMF ',i,' is loaded'
		print len(X_prim[str(i)])

	X_modified = []
	Y_modified = []
	for i in range(0,C.Total_Train_Data):
		sum = np.zeros(samplenumber)
		for j in IMF_number:
			sum = [a + b for a, b in zip(sum, X_prim[str(j)])]
		X_modified.append(sum)
		print i
	Y_modified = Y_prim

	X = []
	Y = []


	##Remove Hypertrophy Class
	indices = [s for s, x in enumerate(Y_prim) if x not in ['Hypertrophy','Miscellaneous','n/a']]
	for f in indices:
			X.append(X_modified[f])
			Y.append(Y_modified[f])


	'''
	# for training Original data
	indices = [s for s, x in enumerate(Y_prim) if x not in [ 'Miscellaneous', 'Hypertrophy']]
        for f in indices:
                        X.append(X_prim[f])
                        Y.append(Y_prim[f])
	'''

	print Counter(Y)


	#encode class_values as integers
	encoder = LabelEncoder()
	encoder.fit(Y)
	encoder_Y = encoder.transform(Y)
	#convert integers to dummy variables(i.e: one hot encoding)
	dummy_Y = np_utils.to_categorical(encoder_Y)
	#Split the dataset to train and test data
	X_train,X_test,Y_train,Y_test = train_test_split(X,dummy_Y,test_size = C.valuation_split, random_state = seed)

	print '\n\nData Loaded\n\n'


	if C.CNN_model_use:
		X_train = np.expand_dims(X_train, axis=2)
		X_test = np.expand_dims(X_test, axis=2)

	if resume=='False':
		#get the model and print the summary
		model = M.IMF_models[str(IMF_number)]()
		if C.optimizer=='sgd':
			sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
			#Compile Model
			model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
		else:
			model.compile(loss = 'categorical_crossentropy', optimizer = C.optimizer, metrics = ['accuracy'])
		plot_model(model, to_file=folder+'/Model_Figures/EMD_model.png')
	elif resume=='True':
		#load_architecture
		json_file = open(folder+'/Final_Weights/model_IMF_'+str(IMF_number)+'.json','r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		#Load weights
		weight_folder = folder+'/Training_Records/IMF_'+str(IMF_number)+'/weights_best_of_'+'*'
		filenames =  glob.glob(weight_folder)
		filenames.sort(reverse = True)
		model_weight_file = filenames[0]
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
		mdl_save_path = folder+'/Final_Weights/model_IMF_'+str(IMF_number)+'.json'
		with open(mdl_save_path, "w") as json_file:
			json_file.write(model_json)

	####################    callback list     #######################
	def step_decay(epoch):

		#Drop based Learning rate
		initial_lrate = C.initial_lrate
		drop = C.lrate_drop
		epochs_drop = C.lrate_epochs_drop
		lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
		return lrate
		'''
		#Cyclical learning rate(triangular)
		"""Given the inputs, calculates the lr that should be applicable for this iteration"""
		base_lr = 0.0001
		max_lr = 0.001
		cycle = np.floor(1 + epoch/(2  * C.lrate_epochs_drop))
		x = np.abs(epoch/C.lrate_epochs_drop - 2 * cycle + 1)
		lrate = base_lr + (max_lr - base_lr) * np.maximum(0, (1-x))
		return lrate
		'''
	#checkpoint path
	chk_path = folder+'/Training_Records/'+'IMF_'+str(IMF_number)+'/weights_best_of_'+'IMF_'+str(IMF_number)+'.hdf5'
	checkpoint_best = ModelCheckpoint(chk_path,monitor='val_acc',verbose=1,save_best_only=True,mode=C.chkpointpath_saving_mode)
	#Save Every epoch
	chk_path = folder+'/Saved_All_Weights/'+'IMF_'+str(IMF_number)+'/IMF_'+str(IMF_number)+"_Each_Epoch.hdf5"
	each_epoch = ModelCheckpoint(chk_path,monitor='val_acc',verbose=1,save_best_only=False,mode='auto', period=1)
	# learning schedule callback
	lrate = LearningRateScheduler(step_decay)
	#Early Stopping
	Early_stop = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=20, verbose=0, mode='max')
	#Callback that streams epoch results to a csv file.
	csv_logger = CSVLogger(folder+'/Training_CSV_log/training_IMF_'+str(IMF_number)+'.log')
	#keras FB ip
	#FB = fb.sendmessage(savelog=True,fexten='TEST',username='nahianhasanbuet@gmail.com',password='Ratul1994')
	#Tensorboard visualization
	TENS_FILE = folder+'/Tensorboard_Visualization/IMF_'+str(IMF_number)+'/{}'
	tensor_board = TensorBoard(log_dir = TENS_FILE.format(time()),histogram_freq=0,write_graph=True,write_images=False)
	#open a terminal and write 'tensorboard --logdir=logdir/' and go to the browser
	#################################################################

	callback_list=[checkpoint_best,each_epoch,lrate,csv_logger,tensor_board,Early_stop]
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
		history = grid.fit(X_train, Y_train, validation_data=(X_test,Y_test), nb_epoch=C.nb_epoch,
		 					batch_size=C.batch_size, verbose=1,callbacks=callback_list,
							shuffle=C.shuffle,initial_epoch=initial_epoch)
	elif not C.grid_search:
		#Fit the model
		history = model.fit(X_train, Y_train, validation_data=(X_test,Y_test), nb_epoch=C.nb_epoch,
		 					batch_size=C.batch_size, verbose=1,callbacks=callback_list,
							shuffle=C.shuffle,initial_epoch=initial_epoch)

	#save the history of whole training
	filehandler = open(folder+"/Training_History/IMF_"+str(IMF_number)+".obj","wb")
	pickle.dump(history.history,filehandler)
	filehandler.close()
	'''
	#evaluate the model on whole training dataset
	scores = model.evaluate(X,dummy_Y, verbose=0)
	print("IMF_%s---%s: %.2f%%" % (IMF_number,model.metrics_names[1], scores[1]*100))

	#Save the final scores to text file
	with open(folder+"/Training_Results/IMF_Training_Result.txt", "a") as myfile:
		string = 'IMF_'+str(IMF_number)+'----'+model.metrics_names[1]+' = '+str(scores[1]*100)+'-------'+'\n'
		myfile.write(string)
	'''

#########################################################################################################################

def Main():

	deviceIDs=[]
	while not deviceIDs:
		deviceIDs = GPUtil.getAvailable(order='first',limit=1,maxMemory=0.80,maxLoad=0.99)
		print 'searching for GPU to be available. Please wait.....'
	print 'GPU Found...Starting Training\n'
	# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

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
	
	#Generate EMD separate IMF csv files in the csv path
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
	

	print '\n\nOriginal Data training started\n\n'
	separate_threads(folder,[1,2,3],file_path,patient_data,problem_data,csv_folder,samplenumber,resume,initial_epoch)

	#Plotting the history of training
	print 'Finished Training all the IMF segments'
	print "Let's see how the training was"

#if(__name__ == '__Main__'):
Main()
#Delete all .pyc files
direc = os.getcwd()
test=os.listdir(direc)
for item in test:
	if item.endswith(".pyc"):
		os.remove(item)
