from sklearn.metrics import confusion_matrix
import confusion_mat_plot as cmpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import config
global C
C = config.Config()
global fo
fo = './Outputs'

def Confusion_Matrix_Plot():
	
	true_class = []
	pred_class = []
	with open(fo+'/Test_Result/Maximum_Prob/maximum_parallel_probability_3rd_test.txt','r') as f:
		line = f.readline()
		line = f.readline()
		while(line):
			splitted = line.split(',')
			true_class.append(str(splitted[-1][:-1]))
			pred_class.append(str(splitted[-2]))
			line = f.readline()
		print len(true_class),'---',len(pred_class)
		cm = confusion_matrix(true_class, pred_class)
		cm_plot_labels = C.disease_names
		## convert  array into a dataframe
		df = pd.DataFrame (cm)
		## save to xlsx file
		filepath = fo+'/Analysis/Confusion_matrices/confusion_matrix_excel_file_prallel_3rd_test'+'.xlsx'
		df.to_excel(filepath, index=False)
		plt.figure(figsize=(25,15))
		cmpl.plot_confusion_matrix(cm,cm_plot_labels)
		plt.title('Parallel_Model_Testing_Result')
	plt.savefig(fo+'/Analysis/Confusion_matrices/Confusion_matrix_Parallel_3rd_test.png')

#Confusion_Matrix_Plot()
