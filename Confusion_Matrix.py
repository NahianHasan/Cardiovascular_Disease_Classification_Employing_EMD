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

def Confusion_Matrix_Plot(number_of_IMFs):
	#fig,ax= plt.subplots(nrows=2, ncols=3,figsize=(25,15))
	for i in range(1,number_of_IMFs+1):
		print 'IMF_',str(i)
		true_class = []
		pred_class = []
		with open(fo+'/Test_Result/Maximum_Prob/maximum_probability'+str(i)+'.txt','r') as f:
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
			filepath = fo+'/Analysis/Confusion_matrices/confusion_matrix_excel_file_IMF_'+str(i)+'.xlsx'
			df.to_excel(filepath, index=False)
			#plt.subplot(2,3,i)
			#plt.ylabel('True Classes')
			#plt.xlabel('Predicted Classes')
			#cmpl.plot_confusion_matrix(cm,cm_plot_labels)
			#plt.title('IMF_%s_Confusion Matrix'%(str(i)))
		#plt.savefig(fo+'/Analysis/Confusion_matrices/Plot_of_confusion_matrix.png')

#Confusion_Matrix_Plot(C.number_of_IMFs)
