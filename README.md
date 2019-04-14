# Cardiovascular_Disease_Classification_Employing_EMD
Cardiovascular Disease Classification Employing Empirical Mode Decomposition (EMD) of Modified ECG

Folder Files:\n
config.py--------------------------------- contains parameters
Folder_creation.py------------------------ creates separate folders to store the training outputs
EMD_Models.py----------------------------- contains different network architectures to train the model
EMD_Models_Parallel.py-------------------- contains parallel architectures for training. [for details refer to the original elsevier article.]
EMD_data_prepare.py------ code for generating IMF signals from EMD analysis of signals.
>>EMD_data_train_all_combinations_pc.py----- train on specified IMF combination in a CPU
>>EMD_data_train_any_combinations.py-------- train on specified IMF combination in a GPU
>>EMD_data_training_pc.py------------------- train on individual IMF in CPU
>>EMD_data_training_server.py--------------- train on individual IMF in GPU
>>EMD_data_training_server_ensembled.py----- train on IMF[1,2,3] combination only in GPU
>>EMD_ensembled_data_prepare_Test_server.py- Modified ECG signal formation from IMF[1,2,3] for test purpose.
>>EMD_ensembled_data_prepare_Train_server.py-Modified ECG signal formation from IMF[1,2,3] for train purpose.
>>EMD_Test_all_combinations.py-------------- Test code on specified IMF combination
>>EMD_Test.py------------------------------- Test code on individual IMF.
>>EMD_Parallel_Test.py---------------------- Test code on parallel architecture
>>EMD_Parallel_Test_3_IMF.py---------------- Test code on parallel architecture of 3 specified IMFs only.
>>EMD_Test_original.py---------------------- Test code on original raw signal
>>Original_signal_Data_prepare.py----------- Prepare original raw data for training
>>confusion_mat_plot.py--------------------- Confusion matrix plot[used by Confusion_Matrix.py]
>>Confusion_Matrix.py----------------------- Confusion matrix plot
>>Confusion_Matrix_Parallel.py-------------- Confusion matrix for parallel architecture
>>all_combinations_plot.py------------------ Plot the accuracy of all combinations used for modified ECG

#FILES INSIDE 'Noise Analysis' FOLDER
>>modified_ECG_analysis_MIT_BIH.py---------- Pearson correlation coefficient calculation for each of the classes considered from MIT-BIH Arrhythmia database
>>modified_ECG_analysis_PTB_Petes.py-------- Pearson correlation coefficient calculation for each of the classes considered from PTB and ST.-Petersberg database

#FILES INSIDE 'Visualization' FOLDER
>>EMD_visualization_petersburg.py----------- IMFs visualization of Petersburg Database
>>EMD_visualization_PTB.py------------------ IMFs visualization of PTB databse
>>learning_rate_scheduler.py---------------- learning rate scheduler visualize used in training
>>modified_EMD_visualization_Petersburg.py-- Modified EMD comparison with raw signal of Petersburg database
>>modified_EMD_visualization_PTB.py--------- Modified EMD comparison with raw signal of PTB database



