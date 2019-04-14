from biosppy import storage
from biosppy.signals import ecg
from numpy import median,mean
import numpy as np
from scipy.stats import pearsonr
import pyhht
from pyhht.emd import EMD
import wfdb
import math
import os

def ECG_analysis(signal,show=False,sampling_rate=1000):
    # process it and plot
    out = ecg.ecg(signal=signal, sampling_rate=sampling_rate, show=show)
    HR = out["heart_rate"]
    RPeaks = out["rpeaks"]

    RR = []
    RR_interval_in_time = []
    for i in range(0,len(RPeaks)-1):
        s1 = RPeaks[i]
        s2 = RPeaks[i+1]
        RR.append(abs(s2-s1))
        RR_interval_in_time.append(abs(s2-s1)/float(sampling_rate))
    max_RR = max(RR)
    min_RR = min(RR)
    Ratio = float(max(RR))/float(min(RR))

    ### Adaptive template matching  ##########
    Average_RR = mean(RR)
    Median_RR = median(RR)

    center_of_median_RR = int(round(Median_RR/2.0))
    ## remove the first R Peak
    RPeaks = RPeaks[1:]
    Individual_Beats = []
    for i in range(0,len(RPeaks)):
        sample_start = RPeaks[i]-center_of_median_RR
        sample_end = RPeaks[i]+center_of_median_RR
        cut_template = signal[sample_start:sample_end]
        Individual_Beats.append(cut_template)

    Individual_Beats = np.asarray(Individual_Beats)
    # finding the average template
    r,c = Individual_Beats.shape
    Average_Template = []
    for i in range(0,c):
        avg = sum(Individual_Beats[:,i])/float(r)
        Average_Template.append(avg)

    #Individual Correlation Coefficient
    Individual_Corr_coeff=[]
    for i in range(0,r):
        pearson_r,pearson_p = pearsonr(Individual_Beats[i,:],Average_Template)
        Individual_Corr_coeff.append(pearson_r)
    #Average Correlation Coefficient
    Avg_Corr_coeff = mean(Individual_Corr_coeff)

    return RR,RR_interval_in_time,min_RR,max_RR,Average_RR,Ratio,Individual_Corr_coeff,Avg_Corr_coeff, Average_Template, Individual_Beats

def Main():
    # load raw ECG signal
    #signal, mdata = storage.load_txt('./examples/ecg.txt')

    samplenumber = 5000
    File_Path = './Database/PTB/'
    samp_rating = 1000

    dir_files1=[]
    for (dirpath, dirnames, filenames) in os.walk(File_Path):
        dir_files1 += [os.path.join(dirpath, file[0:-4]) for file in filenames]

    dir_files = list(set(dir_files1))
    print dir_files
    Read_Files = []

    avg_min_RR_emd = []
    avg_max_RR_emd = []
    avg_avg_RR_emd = []
    avg_ratio_emd = []
    avg_coeff_emd = []

    avg_min_RR_orig = []
    avg_max_RR_orig = []
    avg_avg_RR_orig = []
    avg_ratio_orig = []
    avg_coeff_orig = []
    Diseases = []

    ##### Save the Data
    A = open('./Analysis/PTB/Analysis_avg_avg_RR.csv','w')
    B = open('./Analysis/PTB/Analysis_avg_ratio.csv','w')
    C = open('./Analysis/PTB/Analysis_avg_coeff.csv','w')
    A.write('Patient_ID'+','+'EMD'+','+'Original'+'\n')
    B.write('Patient_ID'+','+'EMD'+','+'Original'+'\n')
    C.write('Patient_ID'+','+'EMD'+','+'Original'+'\n')

    for j in range(0,len(dir_files)):
        try:
            print dir_files[j],
            ECG_signal,ecgrecord = wfdb.srdsamp(dir_files[j])
            record = wfdb.rdsamp(dir_files[j])
            sig_length = len(ECG_signal)
            disease = ecgrecord['comments'][4][22:]
            print disease,
            #print record.__dict__

            repetition = int(math.floor(sig_length/samplenumber))
            sig_start = 0
            count = 0
            for h in range(0,repetition):
                signal = []
                for i in range(sig_start,sig_start+samplenumber):
                    sum = 0
                    for channel in range(0,15):
                        sum += ECG_signal[i][channel]
                    signal.append(sum)
                try:
                    RR_orig,RR_time_orig,min_RR_orig,max_RR_orig,Average_RR_orig,Ratio_orig,Individual_coeff_orig,Avg_coeff_orig, Avg_template_orig, Individual_Beats_orig = ECG_analysis(signal[0:samplenumber],show=False,sampling_rate=samp_rating)
                    #Read_Files.append(dir_files[j])
                    #EMD Analysis
                    signal_for_EMD = np.asarray(signal[0:samplenumber])

                    decomposer = EMD(signal_for_EMD,n_imfs=3,maxiter=2000)
                    imfs = decomposer.decompose()

                    EMD_data = []
                    for i in range(0,samplenumber):
                        EMD_data.append(imfs[0][i]+imfs[1][i]+imfs[2][i])
                    RR_emd,RR_time_emd,min_RR_emd,max_RR_emd,Average_RR_emd,Ratio_emd,Individual_coeff_emd,Avg_coeff_emd,Avg_template_emd, Individual_Beats_emd = ECG_analysis(EMD_data[0:samplenumber],show=False,sampling_rate=samp_rating)

                    # Print
                    #print min_RR_emd, ',', min_RR_orig
                    #print max_RR_emd,',',max_RR_orig
                    #print 'AVG_RR_emd=',Average_RR_emd,' Avg_RR_orig=' ,Average_RR_orig,
                    #print Ratio_emd,',',Ratio_orig
                    print 'Emd_coeff=',Avg_coeff_emd,' Orig_coeff=',Avg_coeff_orig,
                    print 'start=',sig_start,' count=',count

                    '''
                    avg_min_RR_emd.append(min_RR_emd)
                    avg_max_RR_emd.append(max_RR_emd)
                    avg_avg_RR_emd.append(Average_RR_emd)
                    avg_ratio_emd.append(Ratio_emd)
                    avg_coeff_emd.append(Avg_coeff_emd)

                    avg_min_RR_orig.append(min_RR_orig)
                    avg_max_RR_orig.append(max_RR_orig)
                    avg_avg_RR_orig.append(Average_RR_orig)
                    avg_ratio_orig.append(Ratio_orig)
                    avg_coeff_orig.append(Avg_coeff_orig)
                    '''

                    #Diseases.append(disease)
                    sig_start = sig_start + samplenumber

                    A.write(dir_files[j]+','+str(Average_RR_emd)+','+str(Average_RR_orig)+','+disease+'\n')
                    B.write(dir_files[j]+','+str(Ratio_emd)+','+str(Ratio_orig)+','+disease+'\n')
                    C.write(dir_files[j]+','+str(Avg_coeff_emd)+','+str(Avg_coeff_orig)+','+disease+'\n')
                    count += 1
                except:
                    sig_start = sig_start + samplenumber
                    print 'Problem in the cut sequencee'

        except:
            print 'Problem: ',dir_files[j][-7:]


    '''
    for i in range(0,len(avg_avg_RR_emd)):
        A.write(dir_files[i]+','+str(avg_avg_RR_emd[i])+','+str(avg_avg_RR_orig[i])+','+Diseases[i]+'\n')
        B.write(dir_files[i]+','+str(avg_ratio_emd[i])+','+str(avg_ratio_orig[i])+','+Diseases[i]+'\n')
        C.write(dir_files[i]+','+str(avg_coeff_emd[i])+','+str(avg_coeff_orig[i])+','+Diseases[i]+'\n')
    '''


Main()
