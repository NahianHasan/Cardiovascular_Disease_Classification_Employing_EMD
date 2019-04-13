import math

class Config:

	def __init__(self):
		self.samplenumber=1000#sample number of each record
		self.optimizer='sgd'#optimization algorithm used
		self.classes=7#number of classes in your database
		self.loss_type='categorical_crossentropy'#Loss function used for calculating loss in the DNN architecture
		self.number_of_IMFs =6#No of IMFs to be computed during EMD
		self.Total_Train_Data = 131174#Total number of training data
		self.data_path='/media/nahianhasan/New Volume1/PTB_Database/PTB/'#data path to your database
		#self.Resultant_Folder = './Outputs_Ensembled_3_IMF_7_class_PTB'
		self.IMF_csv_path = '/PTB_csv_folder/'#path to the CSV data files of your database
		self.patient_data_path=self.data_path+'RECORDS.txt'#Used for Patient data record list..if you have any
		self.preoblem_data_path='./Outputs/Problem_Data/problem.txt'#A file that will contain the list of records that had problems in preparing IMFs during EMD
		self.nb_epoch=50#Number of Epochs
		self.batch_size=32#Batch size during training
		self.shuffle = True#Data shuffle during training
		self.initial_epoch = 0#Use it for model Training Resuming;Initially it is zero, but if you resume training, set it to requred epoch from where you want to start the training
		self.valuation_split = 0.20#Validation Data split in training data
		self.chkpointpath_saving_mode='max'#the mode of calculating the metric for checkpointing the best model
		self.EMD_data_prepare = False#Whether you want to prepare EMD data or not.If not, you must specify the location of IMF csv paths.
		self.split_perc= 0.7# This means 70% of the data are training data and rest 30% are testing data
		self.initial_lrate=0.01#Inintial learning rate for scheduling
		self.lrate_drop = 0.5#Learning rate drop by this amount
		self.lrate_epochs_drop=5#After this amount of epoch the learning rate drops by self.lrate_drop
		self.grid_search = False#Whether\
		self.IMF_array = [1,2,3]#Array specifying the combination of IMFs that you want to check while running EMD_data_train_all_combinations.py

		#Disease names of your database
		self.disease_names = ['Bundle branch block','Cardiomyopathy','Dysrhythmia','Healthy control',
								'Myocardial infarction','Myocarditis','Valvular heart disease']
		#DNN architecture to be used in the training procedure. #You can also choose--------------
		#'VGGNet16','VGGNet19','RESNet50','AlexNet','ZFNet','Custom'
		self.architect = 'AlexNet'

		if not self.architect == 'Custom':
			self.CNN_model_use = True
		else:
			self.CNN_model_use = False

		#Alphabetically ordered disease labels for testing purpose.
		self.disease_labels={'Myocardial infarction':4,
						'Healthy control':3,
						'Cardiomyopathy':1,
						'Bundle branch block':0,
						'Dysrhythmia':2,
						'Valvular heart disease':6,
						'Myocarditis':5}
