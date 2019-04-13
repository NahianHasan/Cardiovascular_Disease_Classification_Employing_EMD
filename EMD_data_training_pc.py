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
from collections import Counter
import matplotlib.pyplot as plt
#import keras libraries
from keras.utils import np_utils
from keras.callbacks import CSVLogger
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
global C,Y_val
global M
C = config.Config()
M = EMD_Models.MODELS()

def Valuation_data_load(csv_path,valuation_samples,batch=128):
	global Y_val
	Y_valuation = []
	X_valuation = []
	print 'Validation Data is being prepared'
	batch_count = 0
	classes=C.disease_names
	with open(csv_path,'r') as G:
		count = 0
		line = G.readline()
		while(line):
			if count in Y_val:
				split = line.split(',')
				X_valuation.append(split[:-1])
				Y_valuation.append(split[-1][:-1])
				batch_count = batch_count + 1
			count = count + 1
			if batch_count == batch or count == valuation_samples:
				X_valuation = np.asarray(X_valuation)
				encoder = LabelBinarizer()
				encoder.fit(classes)
				Y_valuation = encoder.transform(Y_valuation)
				if C.CNN_model_use:
					X_valuation = np.expand_dims(X_valuation, axis=2)
				yield X_valuation,Y_valuation
				batch_count = 0
				Y_valuation = []
				X_valuation = []
			line =G.readline()

def Train_Data_Load(csv_path,classes,total_training_data_length):
	global Y_val
	#Data Load in Chunks
	num_iters = int(math.floor((C.Total_Train_Data*(1.0-C.valuation_split))/C.batch_size))
	with open(csv_path,'r') as f:
		data_number = 0
		Data = []
		Label = []
		line = f.readline()
		line_count = 0
		while(line):

			split = line.split(',')
			Data.append(split[:-1])
			Label.append(split[-1][:-1])
			line_count = line_count+1
			data_number = data_number + 1
			if line_count==C.batch_size or data_number == total_training_data_length:
				X_train = np.asarray(Data)
				Y_train = Label
				#encode class_values as integers
				encoder = LabelBinarizer()
				encoder.fit(classes)
				Y_train = encoder.transform(Y_train)
				if C.CNN_model_use:
					X_train = np.expand_dims(X_train, axis=2)
				yield X_train,Y_train
				Data = []
				Label = []
				line_count = 0
			line = f.readline()


def separate_threads(IMF_number,filepath,patient_data,problem_data,csv_folder,samplenumber,resume,initial_epoch):
	global Y_val
	samplenumber=samplenumber
	#Valuation Data generation
	Y_val = random.sample(range(0, C.Total_Train_Data), int(math.ceil(0.2*float(C.Total_Train_Data))))

	# fix random seed for reproducibility
	seed = 7
	np.random.seed(seed)
	#itterate through csv file
	csv_path = csv_folder+'IMF'+str(IMF_number)+'_train'+'.csv'
	classes = C.disease_names

	if resume=='False':
		#get the model and print the summary
		model = M.IMF_models[str(IMF_number)]()
		if C.optimizer=='sgd':
			sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
			#Compile Model
			model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
		else:
			model.compile(loss = 'categorical_crossentropy', optimizer = C.optimizer, metrics = ['accuracy'])
	elif resume=='True':
		#load_architecture
		json_file = open('./Outputs/Final_Weights/model_IMF_'+str(IMF_number)+'.json','r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		#Load weights
		weight_folder = './Outputs/Saved_All_Weights/IMF_'+str(IMF_number)+'/IMF_'+str(IMF_number)+'*'
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
		mdl_save_path = './Outputs/Final_Weights/model_IMF_'+str(IMF_number)+'.json'
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
	chk_path = './Outputs/Training_Records/'+'IMF_'+str(IMF_number)+'/weights_best_of_'+'IMF_'+str(IMF_number)+'_{epoch:02d}-{acc:.2f}.hdf5'
	checkpoint_best = ModelCheckpoint(chk_path,monitor='acc',verbose=1,save_best_only=True,mode=C.chkpointpath_saving_mode)
	#Save Every epoch
	chk_path = './Outputs/Saved_All_Weights/'+'IMF_'+str(IMF_number)+'/IMF_'+str(IMF_number)+"_-{epoch:02d}-{acc:.2f}.hdf5"
	each_epoch = ModelCheckpoint(chk_path,monitor='acc',verbose=1,save_best_only=False,mode='auto', period=1)
	# learning schedule callback
	lrate = LearningRateScheduler(step_decay)
	#Early Stopping
	Early_stop = EarlyStopping(monitor='loss', min_delta=0, patience=100, verbose=0, mode='max')
	#Callback that streams epoch results to a csv file.
	csv_logger = CSVLogger('./Outputs/Training_CSV_log/training_IMF_'+str(IMF_number)+'.log')
	#Tensorboard visualization
	tensor_board = TensorBoard(log_dir='./Outputs/Tensorboard_Visualization/',histogram_freq=0,write_graph=False,write_images=False)
	#open a terminal and write 'tensorboard --logdir=logdir/' and go to the browser
	#################################################################

	callback_list=[checkpoint_best,each_epoch,lrate,csv_logger,Early_stop,tensor_board]
	#Data_Generator
	total_training_data_length = C.Total_Train_Data - len(Y_val)
	valuation_samples = len(Y_val)
	Train_Data_generator = Train_Data_Load(csv_path,classes,total_training_data_length)
	Validation_Data_generator = Valuation_data_load(csv_path,valuation_samples,batch=C.batch_size)

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
		history = grid.fit(Train_Data_generator, nb_epoch=C.nb_epoch,samples_per_epoch = total_training_data_length,
							verbose=2,shuffle=C.shuffle,validation_data=Validation_Data_generator,
							nb_val_samples=valuation_samples,callbacks = callback_list)
	elif not C.grid_search:
		history = model.fit_generator(Train_Data_generator,validation_data=Validation_Data_generator,
						nb_epoch=C.nb_epoch,samples_per_epoch =total_training_data_length,
						verbose=1,shuffle=C.shuffle,callbacks = callback_list,nb_val_samples=valuation_samples)

	#save the history of whole training
	filehandler = open("./Outputs/Training_History/IMF_"+str(IMF_number)+".obj","wb")
	pickle.dump(history.history,filehandler)
	filehandler.close()


#########################################################################################################################

def Main():
	parser = argparse.ArgumentParser(description='ECG data training using EMD Data with separate threading',
									usage='Classifying EMD Data',
									epilog='Give proper arguments')
	parser.add_argument('-p',"--data_path",metavar='', help="Path to temain database",default=C.data_path)
	parser.add_argument('-c',"--csv_path",metavar='',help="Path to the CSV Folder of EMD Data",default=C.IMF_csv_path)
	parser.add_argument('-rc',"--patient_data_path",metavar='',help="Path to the Patient file RECORD.txt",default=C.patient_data_path)
	parser.add_argument('-res',"--resume_train",metavar='',help="Resume Training",default='False')
	parser.add_argument('-inep',"--ini_epoch",metavar='',help="Initial Epoch after Resuming Training",default=C.initial_epoch)
	parser.add_argument('-reim',"--res_imf",metavar='',help="Resumed IMF number after resuming",default=1)
	parser.add_argument('-pd',"--problem_data_path",metavar='',help="Path to the text file where problematic data to be stored",default=C.preoblem_data_path)
	parser.add_argument('-s',"--sample_number",metavar='',help="Number of samples to be taken by each record",type=int,default=C.samplenumber)
	parser.add_argument('-imf',"--number_of_IMFs",metavar='',help="Number of IMFs to be extracted",default=C.number_of_IMFs,type=int,choices=[2,3,4,5,6])
	parser.add_argument('-spl',"--split_perc",metavar='',help="Splitting percentage of train and test(upper limit)",type=float,default=C.split_perc)

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
	#Check whether specific folders are present or not....if not create them
	FC.Folder_creation(number_of_IMFs,'./Outputs')
	'''
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
