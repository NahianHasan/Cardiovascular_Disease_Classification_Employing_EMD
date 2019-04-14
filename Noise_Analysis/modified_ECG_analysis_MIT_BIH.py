disease_100 = ['N']
beats_disease_100 = {'N':'30:6'}

disease_101 = ['N']
beats_disease_101 = {'N':'30:6'}

disease_102 = ['N','P']
beats_disease_102 = {'N':'1:22','P':'28:44'}

disease_103 = ['N']
beats_disease_103 = {'N':'30:06'}

disease_104 = ['N','P']
beats_disease_104 = {'N':'3:52','P':'26:13'}

disease_105 = ['N']
beats_disease_105 = {'N':'30:06'}

disease_106 = ['N','B','T','VT']
beats_disease_106 = {'N':'22:36','B':'7:15','T':'0:13','VT':'0:2'}

disease_107 = ['P']
beats_disease_107 = {'P':'30:06'}

disease_108 = ['N']
beats_disease_108 = {'N':'30:06'}

disease_109 = ['N']
beats_disease_109 = {'N':'30:06'}

disease_111 = ['N']
beats_disease_111 = {'N':'30:06'}

disease_112 = ['N']
beats_disease_112 = {'N':'30:06'}

disease_113 = ['N']
beats_disease_113 = {'N':'30:06'}

disease_114 = ['N','SVTA']
beats_disease_114 = {'N':'30:06','SVTA':'0:5'}

disease_115 = ['N']
beats_disease_115 = {'N':'30:06'}

disease_116 = ['N']
beats_disease_116 = {'N':'30:06'}

disease_117 = ['N']
beats_disease_117 = {'N':'30:06'}

disease_118 = ['N']
beats_disease_118 = {'N':'30:06'}

disease_119 = ['N','B','T']
beats_disease_119 = {'N':'22:36','B':'3:55','T':'3:34'}

disease_121 = ['N']
beats_disease_121 = {'N':'30:06'}

disease_122 = ['N']
beats_disease_122 = {'N':'30:06'}

disease_123 = ['N']
beats_disease_123 = {'N':'30:06'}

disease_124 = ['N','NOD','T','IVR']
beats_disease_124 = {'N':'28:36','NOD':'0:30','T':'0:22', 'IVR':'0:37'}

disease_200 = ['N','B','VT']
beats_disease_200 = {'N':'15:58','B':'13:52','VT':'0:15'}

disease_201 = ['N','SVTA','AFIB','NOD','T']
beats_disease_201 = {'N':'12:57','SVTA':'0:2','AFIB':'10:6','NOD':'0:24','T':'0:37'}

disease_202 = ['N','AFL','AFIB']
beats_disease_202 = {'N':'19:31','AFL':'0:48','AFIB':'9:46'}

disease_203 = ['N','AFL','AFIB','T','VT']
beats_disease_203 = {'N':'2:43','AFL':'5:14','AFIB':'21:32','T':'0:4','VT':'0:33'}

disease_205 = ['N','VT']
beats_disease_205 = {'N':'29:43','VT':'0:23'}

disease_207 = ['N','SVTA','B','IVR','VT','VFL']
beats_disease_207 = {'N':'22:20','SVTA':'0:52','B':'2:38', 'IVR':'1:49','VT':'0:3','VFL':'2:24'}

disease_208 = ['N','T']
beats_disease_208 = {'N':'24:43','T':'5:22'}

disease_209 = ['N','SVTA']
beats_disease_209 = {'N':'28:23','SVTA':'1:42'}

disease_210 = ['AFIB','B','T','VT']
beats_disease_210 = {'AFIB':'29:30','B':'0:23','T':'0:7','VT':'0:6'}

disease_212 = ['N']
beats_disease_212 = {'N':'30:06'}

disease_213 = ['N','B','VT']
beats_disease_213 = {'N':'29:01','B':'1:0','VT':'0:4'}

disease_214 = ['N','T','VT']
beats_disease_214 = {'N':'28:53','T':'1:8','VT':'0:5'}

disease_215 = ['N','VT']
beats_disease_215 = {'N':'30:03','VT':'0:2'}

disease_217 = ['AFIB','P','B','VT']
beats_disease_217 = {'AFIB':'4:12','P':'25:10','B':'0:42','VT':'0:2'}

disease_219 = ['N','AFIB','B','T']
beats_disease_219 = {'N':'6:1','AFIB':'23:47','B':'0:8', 'T':'0:10'}

disease_220 = ['N','SVTA']
beats_disease_220 = {'N':'29:50','SVTA':'0:16'}

disease_221 = ['AFIB','B','T','VT']
beats_disease_221 = {'AFIB':'29:17','B':'0:3', 'T':'0:42','VT':'0:4'}

disease_222 = ['N','AB','SVTA','AFL','AFIB','NOD']
beats_disease_222 = {'N':'15:57','AB':'1:28','SVTA':'0:8','AFL':'7:3','AFIB':'1:44','NOD':'3:45'}

disease_223 = ['N','B','T','VT']
beats_disease_223 = {'N':'23:23','B':'4:19', 'T':'0:38', 'VT':'1:46'}

disease_228 = ['N','B']
beats_disease_228 = {'N':'24:17','B':'5:48'}

disease_230 = ['N','PREX']
beats_disease_230 = {'N':'17:45','PREX':'12:21'}

disease_231 = ['N','BII']
beats_disease_231 = {'N':'18:26','BII':'11:40'}

disease_232 = ['SBR']
beats_disease_232 = {'SBR':'30:06'}

disease_233 = ['N','B','T','VT']
beats_disease_233 = {'N':'28:3','B':'1:48','T':'0:4','VT':'0:11'}

disease_234 = ['N','SVTA']
beats_disease_234 = {'N':'29:40','SVTA':'0:26'}

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
    samplenumber = 5000
    File_Path = './Database/MIT-BIH'
    samp_rating = 360

    dir_files1=[]
    for (dirpath, dirnames, filenames) in os.walk(File_Path):
        dir_files1 += [os.path.join(File_Path, file[0:-4]) for file in filenames]

    dir_files = list(set(dir_files1))
    dir_files.sort()
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
    A = open('./Analysis/MIT-BIH/Analysis_avg_avg_RR.csv','w')
    B = open('./Analysis/MIT-BIH/Analysis_avg_ratio.csv','w')
    C = open('./Analysis/MIT-BIH/Analysis_avg_coeff.csv','w')
    A.write('Patient_ID'+','+'EMD'+','+'Original'+','+'disease'+'\n')
    B.write('Patient_ID'+','+'EMD'+','+'Original'+','+'disease'+'\n')
    C.write('Patient_ID'+','+'EMD'+','+'Original'+','+'disease'+'\n')

    for j in range(0,len(dir_files)):
        try:
            print dir_files[j],
            original_signal,ecgrecord = wfdb.srdsamp(dir_files[j])
            record = wfdb.rdsamp(dir_files[j])

            data_file = dir_files[j][-3:]
            sig_diseases = globals()['disease_'+str(data_file)]
            for gf in sig_diseases:
                time = globals()['beats_disease_'+str(data_file)][gf]
                time_split = time.split(':')
                minutes = time_split[0]
                seconds = time_split[1]
                total_seconds = int(minutes)*60 + int(seconds)
                total_samples = total_seconds * samp_rating
                disease = gf
                print gf,

                initial_start = 0 # per record starting index of each disease of that record
                ECG_signal = original_signal[initial_start:total_samples]
                sig_length = len(ECG_signal)
                print 'original sig length ', len(original_signal),
                print 'cut_signal_length ',sig_length,

                repetition = int(math.floor(sig_length/samplenumber))
                print 'repeat ', repetition,
                sig_start = 0
                count = 0
                for h in range(0,repetition):
                    signal = []
                    for i in range(sig_start,sig_start+samplenumber):
                        signal.append(ECG_signal[i][0]+ECG_signal[i][1])
                    try:
                        RR_orig,RR_time_orig,min_RR_orig,max_RR_orig,Average_RR_orig,Ratio_orig,Individual_coeff_orig,Avg_coeff_orig, Avg_template_orig, Individual_Beats_orig = ECG_analysis(signal[0:samplenumber],show=False,sampling_rate=samp_rating)
                        #Read_Files.append(dir_files[j])
                        #EMD Analysis
                        signal_for_EMD = np.asarray(signal[0:samplenumber])

                        decomposer = EMD(signal_for_EMD,n_imfs=3,maxiter=3000)
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
                initial_start = total_samples
        except:
            print 'Problem: ',dir_files[j][-7:]


    '''
    for i in range(0,len(avg_avg_RR_emd)):
        A.write(dir_files[i]+','+str(avg_avg_RR_emd[i])+','+str(avg_avg_RR_orig[i])+','+Diseases[i]+'\n')
        B.write(dir_files[i]+','+str(avg_ratio_emd[i])+','+str(avg_ratio_orig[i])+','+Diseases[i]+'\n')
        C.write(dir_files[i]+','+str(avg_coeff_emd[i])+','+str(avg_coeff_orig[i])+','+Diseases[i]+'\n')
    '''


Main()
