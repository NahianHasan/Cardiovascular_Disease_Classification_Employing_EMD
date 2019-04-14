# Cardiovascular_Disease_Classification_Employing_EMD
Cardiovascular Disease Classification Employing Empirical Mode Decomposition (EMD) of Modified ECG

## Getting Started
This readme file will instruct you to run the codes in this repository and customize it to your own database.

###Folder Files
>config.py--------------------------------- contains parameters.<br />
>Folder_creation.py------------------------ creates separate folders to store the training outputs.<br />
>EMD_Models.py----------------------------- contains different network architectures to train the model.<br />
>EMD_Models_Parallel.py-------------------- contains parallel architectures for training. [for details refer to the original elsevier article.]<br />
>EMD_data_prepare.py------ code for generating IMF signals from EMD analysis of signals.<br />
>EMD_data_train_all_combinations_pc.py----- train on specified IMF combination in a CPU<br />
>EMD_data_train_any_combinations.py-------- train on specified IMF combination in a GPU<br />
>EMD_data_training_pc.py------------------- train on individual IMF in CPU<br />
>EMD_data_training_server.py--------------- train on individual IMF in GPU<br />
>EMD_data_training_server_ensembled.py----- train on IMF[1,2,3] combination only in GPU<br />
>EMD_ensembled_data_prepare_Test_server.py- Modified ECG signal formation from IMF[1,2,3] for test purpose.<br />
>EMD_ensembled_data_prepare_Train_server.py-Modified ECG signal formation from IMF[1,2,3] for train purpose.<br />
>EMD_Test_all_combinations.py-------------- Test code on specified IMF combination<br />
>EMD_Test.py------------------------------- Test code on individual IMF.<br />
>EMD_Parallel_Test.py---------------------- Test code on parallel architecture<br />
>EMD_Parallel_Test_3_IMF.py---------------- Test code on parallel architecture of 3 specified IMFs only.<br />
>EMD_Test_original.py---------------------- Test code on original raw signal<br />
>Original_signal_Data_prepare.py----------- Prepare original raw data for training<br />
>confusion_mat_plot.py--------------------- Confusion matrix plot[used by Confusion_Matrix.py]<br />
>Confusion_Matrix.py----------------------- Confusion matrix plot<br />
>Confusion_Matrix_Parallel.py-------------- Confusion matrix for parallel architecture<br />
>all_combinations_plot.py------------------ Plot the accuracy of all combinations used for modified ECG<br />

###FILES INSIDE 'Noise Analysis' FOLDER
>modified_ECG_analysis_MIT_BIH.py---------- Pearson correlation coefficient calculation for each of the classes considered from MIT-BIH Arrhythmia database<br />
>modified_ECG_analysis_PTB_Petes.py-------- Pearson correlation coefficient calculation for each of the classes considered from PTB and ST.-Petersberg database<br />

###FILES INSIDE 'Visualization' FOLDER
>EMD_visualization_petersburg.py----------- IMFs visualization of Petersburg Database<br />
>EMD_visualization_PTB.py------------------ IMFs visualization of PTB databse<br />
>learning_rate_scheduler.py---------------- learning rate scheduler visualize used in training<br />
>modified_EMD_visualization_Petersburg.py-- Modified EMD comparison with raw signal of Petersburg database<br />
>modified_EMD_visualization_PTB.py--------- Modified EMD comparison with raw signal of PTB database<br />



