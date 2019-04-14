# Cardiovascular_Disease_Classification_Employing_EMD
Cardiovascular Disease Classification Employing Empirical Mode Decomposition (EMD) of Modified ECG

## Getting Started
This readme file will instruct you to run the codes in this repository and customize it to your own database.

## Folder Files
1.config.py--------------------------------- contains parameters.<br />
2.Folder_creation.py------------------------ creates separate folders to store the training outputs.<br />
3.EMD_Models.py----------------------------- contains different network architectures to train the model.<br />
4.EMD_Models_Parallel.py-------------------- contains parallel architectures for training. [for details refer to the original elsevier article.]<br />
5.EMD_data_prepare.py------ code for generating IMF signals from EMD analysis of signals.<br />
6.EMD_data_train_all_combinations_pc.py----- train on specified IMF combination in a CPU<br />
7.EMD_data_train_any_combinations.py-------- train on specified IMF combination in a GPU<br />
8.EMD_data_training_pc.py------------------- train on individual IMF in CPU<br />
9.EMD_data_training_server.py--------------- train on individual IMF in GPU<br />
10.EMD_data_training_server_ensembled.py----- train on IMF[1,2,3] combination only in GPU<br />
11.EMD_ensembled_data_prepare_Test_server.py- Modified ECG signal formation from IMF[1,2,3] for test purpose.<br />
12.EMD_ensembled_data_prepare_Train_server.py-Modified ECG signal formation from IMF[1,2,3] for train purpose.<br />
13.EMD_Test_all_combinations.py-------------- Test code on specified IMF combination<br />
14.EMD_Test.py------------------------------- Test code on individual IMF.<br />
15.EMD_Parallel_Test.py---------------------- Test code on parallel architecture<br />
16.EMD_Parallel_Test_3_IMF.py---------------- Test code on parallel architecture of 3 specified IMFs only.<br />
17.EMD_Test_original.py---------------------- Test code on original raw signal<br />
18.Original_signal_Data_prepare.py----------- Prepare original raw data for training<br />
19.confusion_mat_plot.py--------------------- Confusion matrix plot[used by Confusion_Matrix.py]<br />
20.Confusion_Matrix.py----------------------- Confusion matrix plot<br />
21.Confusion_Matrix_Parallel.py-------------- Confusion matrix for parallel architecture<br />
22.all_combinations_plot.py------------------ Plot the accuracy of all combinations used for modified ECG

## FILES INSIDE 'Noise Analysis' FOLDER
1.modified_ECG_analysis_MIT_BIH.py---------- Pearson correlation coefficient calculation for each of the classes considered from MIT-BIH Arrhythmia database<br />
2.modified_ECG_analysis_PTB_Petes.py-------- Pearson correlation coefficient calculation for each of the classes considered from PTB and ST.-Petersberg database<br />

## FILES INSIDE 'Visualization' FOLDER
1.EMD_visualization_petersburg.py----------- IMFs visualization of Petersburg Database<br />
2.EMD_visualization_PTB.py------------------ IMFs visualization of PTB databse<br />
3.learning_rate_scheduler.py---------------- learning rate scheduler visualize used in training<br />
4.modified_EMD_visualization_Petersburg.py-- Modified EMD comparison with raw signal of Petersburg database<br />
5.modified_EMD_visualization_PTB.py--------- Modified EMD comparison with raw signal of PTB database<br />

## License
This repository is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


## Code Run
### EMD_data_train_any_combinations.py
```
python EMD_data_train_any_combinations.py -options
```

options<br />
- -p = Path to the main database   \|  deafault = config.data_path
- -c = Path to the CSV Folder of EMD Data  \|  default = config.IMF_csv_path
- -res = Resume Training \| default='False'
- -inep = Initial Epoch after Resuming Training \| default=config.initial_epoch
- -reim = Resumed IMF number after resuming \| default=1
- -rc = Path to the Patient file RECORD.txt \| default=config.patient_data_path
- -pd = Path to the text file where problematic data to be stored \| default=config.preoblem_data_path
- -s = Number of samples to be taken by each record",type=int \| default=config.samplenumber
- -imf = Number of IMFs to be extracted \| default=config.number_of_IMFs \| choices=[2,3,4,5,6]
- -spl = Splitting percentage of train and test(upper limit) \| type=float \| default=config.split_perc 
- -fold = Save training and testing results in folder


