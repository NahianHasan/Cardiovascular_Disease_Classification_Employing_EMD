import os
import sys
import config
global C
C = config.Config()

def Folder_creation(number_of_IMFs,Resultant_Folder):
	#Check whether specific folders are present or not....if not create them
	if not os.path.exists(Resultant_Folder):
		os.makedirs(Resultant_Folder)
	if not os.path.exists('IMF_csv'):
		os.makedirs('IMF_csv')
	if not os.path.exists(Resultant_Folder+'/Training_Results'):
		os.makedirs(Resultant_Folder+'/Training_Results')
		for i in range(1,number_of_IMFs+1):
			if not os.path.exists(Resultant_Folder+'/Training_Results/IMF_'+str(i) + '/'):
				os.makedirs(Resultant_Folder+'/Training_Results/IMF_'+str(i) + '/')
	if not os.path.exists(Resultant_Folder+'/Training_History'):
		os.makedirs(Resultant_Folder+'/Training_History')
	if not os.path.exists(Resultant_Folder+'/Saved_All_Weights'):
		os.makedirs(Resultant_Folder+'/Saved_All_Weights')
		for i in range(1,number_of_IMFs+1):
			os.makedirs(Resultant_Folder+'/Saved_All_Weights/IMF_'+str(i)+'/')
	if not os.path.exists(Resultant_Folder+'/Training_Records'):
		os.makedirs(Resultant_Folder+'/Training_Records')
		for i in range(1,number_of_IMFs+1):
			os.makedirs(Resultant_Folder+'/Training_Records/IMF_'+str(i)+'/')
	if not os.path.exists(Resultant_Folder+'/Final_Weights'):
		os.makedirs(Resultant_Folder+'/Final_Weights')
	if not os.path.exists(Resultant_Folder+'/Model_Figures'):
		os.makedirs(Resultant_Folder+'/Model_Figures')
	if not os.path.exists(Resultant_Folder+'/Training_CSV_log'):
		os.makedirs(Resultant_Folder+'/Training_CSV_log')
	if not os.path.exists(Resultant_Folder+'/Tensorboard_Visualization'):
		os.makedirs(Resultant_Folder+'/Tensorboard_Visualization')
	if not os.path.exists(Resultant_Folder+'/Problem_Data'):
		os.makedirs(Resultant_Folder+'/Problem_Data')
	if not os.path.exists(Resultant_Folder+'/Test_Result'):
		os.makedirs(Resultant_Folder+'/Test_Result')
		os.makedirs(Resultant_Folder+'/Test_Result/Pickle_Files/')
		os.makedirs(Resultant_Folder+'/Test_Result/detailed_prob')
		os.makedirs(Resultant_Folder+'/Test_Result/Maximum_Prob')
	#Analysis folders
	if not os.path.exists(Resultant_Folder+'/Analysis'):
		os.makedirs(Resultant_Folder+'/Analysis')
	if not os.path.exists(Resultant_Folder+'/Analysis/Confusion_matrices'):
		os.makedirs(Resultant_Folder+'/Analysis/Confusion_matrices')

#Folder_creation(6)
