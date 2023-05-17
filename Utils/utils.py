import os
import numpy as np
import librosa
from tqdm import tqdm
import pandas as pd
import pickle
import lz4.frame
import scipy
import random
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix

output_lines = []

def printAndAdd(lines, line_list=output_lines):
    if(isinstance(lines, list)):    
        for line in lines:
            line = str(line)
            print(line)
            line_list.append(line)
    else:
        lines = str(lines)
        print(lines)
        line_list.append(lines)

def printToFile(filename, line_list=output_lines):
    os.chdir('/home/m13518003/Tugas Akhir')
    with open(filename, 'w+') as f:
        f.write('\n'.join(line_list))
        f.close()

def pickleLoadAudio(fileList):
    iter = 0
    for files in fileList:
        with lz4.frame.open(files, 'rb') as file:
            sig_batch, y_batch = pickle.load(file)
            file.close()
        if(iter == 0):
            sig_batches = sig_batch
            y_batches = y_batch
        else:
            sig_batches = np.concatenate((sig_batches, sig_batch), axis=0)
            y_batches = np.concatenate((y_batches, y_batch), axis=0)
        iter += 1

    return sig_batches, y_batches

def shuffle_list(list1, list2):
    merged_list = tuple(zip(list1, list2))
    shuffled_list = random.sample(merged_list, k=len(merged_list))
    
    res_list1 = []
    res_list2 = []
    
    for el in shuffled_list:
        res_list1.append(el[0])
        res_list2.append(el[1])
    
    res_list1 = np.array(res_list1)
    res_list2 = np.array(res_list2)
    
    return res_list1, res_list2

def displayTime(startFrame, endFrame):    
    # print(' start time: ' + str(startFrame/sr) + ', end time: ' + str(endFrame/sr))
    printAndAdd(' start time: ' + str(startFrame/sr) + ', end time: ' + str(endFrame/sr), output_lines)
    
def split_audio(sig, sr):
    sig_list = []
    idx = 0
    
    nonMuteSections = librosa.effects.split(sig)  # split audio with any audio signal lesser than 20db as mute
    
    while idx < (len(nonMuteSections)):
        split_start = nonMuteSections[idx][0]
        split_end = nonMuteSections[idx][1]
        split_start_time = split_start/sr
        split_end_time = split_end/sr

        if(idx < len(nonMuteSections)-1):
            next_split_start_time = nonMuteSections[idx+1][0]/sr
            if((next_split_start_time - split_end_time) <= 0.5):
                split_end = nonMuteSections[idx+1][1]
                idx += 1

        section = sig[split_start:split_end]
        if((sum(np.absolute(section))/len(section)) > 0.01): #amplitude check
            sig_list.append(section)
        idx += 1
        
    return sig_list

def get_dur(df):
    duration = []

    for data in tqdm(df.iterrows(),  desc='Progress'):
        filename = df.uuid[data[0]]+str('.wav')
        sig , sr = librosa.load(os.path.join(filename), sr=16000)
            
        sig_list = split_audio(sig, sr)
        
        for sections in sig_list:
            dur = librosa.get_duration(y=sections, sr=sr)
            duration.append(dur)
    
    return duration

def get_outlier(dur_list):
    dur_list = sorted(dur_list)
    Q1 = np.median(dur_list[:(int(len(dur_list)/2))])
    Q3 = np.median(dur_list[(int(len(dur_list)/2)):])
    IQR = Q3 - Q1
    high_outlier = Q3 + 1.5 * (Q3 - Q1)
    # print(Q1)
    printAndAdd(str(Q1), output_lines)
    # print(Q3)
    printAndAdd(str(Q3), output_lines)
    # print(IQR)
    printAndAdd(str(IQR), output_lines)
    # print("higher outlier", high_outlier)
    printAndAdd(str(high_outlier), output_lines)
    return high_outlier

def padtrim_audio(df, high_outlier):
    sig_arr = []
    y = []

    for data in tqdm(df.iterrows(),  desc='Progress'):
        filename = df.uuid[data[0]]+str('.wav')
        sig , sr = librosa.load(os.path.join(filename), sr=16000)
            
        sig_list = split_audio(sig, sr)
            
        for sections in sig_list:
            if(len(sections) < high_outlier*sr):
                sections = librosa.util.pad_center(sections, size=int(high_outlier*sr), mode='constant')
            else:
                sections = librosa.util.fix_length(sections, int(high_outlier*sr))
            
            sig_arr.append(sections)
            y.append(df.status[data[0]])
    
    return sig_arr, y

def extract_melspec(X_train_res, X_train_val, X_test):
    X_train = []
    temp_test = []
    temp_val = []
    sr = 16000

    for sig in tqdm(X_train_res,  desc='Progress'):    
        melspec = librosa.feature.melspectrogram(y=sig, sr=sr, n_fft=512, win_length=512, window=scipy.signal.hann(512), n_mels=128)
        X_train.append(melspec)
        
    for sig in tqdm(X_test,  desc='Progress'):    
        melspec = librosa.feature.melspectrogram(y=sig, sr=sr, n_fft=512, win_length=512, window=scipy.signal.hann(512), n_mels=128)
        temp_test.append(melspec)
        
    for sig in tqdm(X_train_val,  desc='Progress'):    
        melspec = librosa.feature.melspectrogram(y=sig, sr=sr, n_fft=512, win_length=512, window=scipy.signal.hann(512), n_mels=128)
        temp_val.append(melspec)

    X_test = temp_test
    X_train_val = temp_val

    return X_train, X_train_val, X_test

def extract_mfcc(X_train_res, X_train_val, X_test):
    X_train = []
    temp_test = []
    temp_val = []
    sr = 16000

    for sig in tqdm(X_train_res,  desc='Progress'):    
        mfcc_ = librosa.feature.mfcc(y=sig, sr=sr, n_fft=512, n_mfcc=40)
        X_train.append(mfcc_)
        
    for sig in tqdm(X_test,  desc='Progress'):    
        mfcc_ = librosa.feature.mfcc(y=sig, sr=sr, n_fft=512, n_mfcc=40)
        temp_test.append(mfcc_)
        
    for sig in tqdm(X_train_val,  desc='Progress'):    
        mfcc_ = librosa.feature.mfcc(y=sig, sr=sr, n_fft=512, n_mfcc=40)
        temp_val.append(mfcc_)

    X_test = temp_test
    X_train_val = temp_val

    return X_train, X_train_val, X_test

def extract_melspec_single(X):
    X_res = []
    sr = 16000

    for sig in tqdm(X,  desc='Progress'):    
        melspec = librosa.feature.melspectrogram(y=sig, sr=sr, n_fft=512, win_length=512, window=scipy.signal.hann(512), n_mels=128)
        X_res.append(melspec)

    return X_res

def extract_mfcc_single(X):
    X_res = []
    sr = 16000

    for sig in tqdm(X,  desc='Progress'):    
        mfcc_ = librosa.feature.mfcc(y=sig, sr=sr, n_fft=512, n_mfcc=40)
        X_res.append(mfcc_)

    return X_res

def revertCategorical(arr):
    arr_res = []
    for el in arr:
        if(el[0] == 1):
            arr_res.append(0)
        if(el[1] == 1):
            arr_res.append(1)
    
    np_arr = np.array(arr_res)

    return np_arr

def adjustPreds(arr):
    preds = []
    for el in arr:
        if(np.argmax(el) == 0):
            preds.append(0)
        else:
            preds.append(1)
    
    np_preds = np.array(preds)
    return np_preds

def getMetric(y_true, y_pred, mode):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if(mode == 'sensitivity'):
        res = tp/(tp+fn)
    elif(mode == 'specificity'):
        res = tn/(tn+fp)

    return res

def matchList(arr1, arr2):
    match0 = 0
    match1 = 0
    matches = 0
    count0 = 0
    count1 = 0
    pred0count = 0
    pred1count = 0
    arr1_revert = revertCategorical(arr1)

    for i in range(len(arr1_revert)):
        if(arr1_revert[i] == 0):
            count0 += 1
            if(arr1_revert[i] == arr2[i]):
                matches += 1
                match0 += 1
        if(arr1_revert[i] == 1):
            count1 += 1
            if(arr1_revert[i] == arr2[i]):
                matches += 1
                match1 += 1

    for i in range(len(arr2)):
        if(arr2[i] == 0):
            pred0count += 1
        if(arr2[i] == 1):
            pred1count += 1
    
    printAndAdd('Predictions Matched: {} / {} ({})'.format(matches, len(arr1_revert), (matches/len(arr1_revert))))
    printAndAdd('0 Label Correctly Predicted: {} / {}'.format(match0, count0))
    printAndAdd('1 Label Correctly Predicted: {} / {}'.format(match1, count1))
    printAndAdd('Predict 0 Count: {}'.format(pred0count))
    printAndAdd('Predict 1 Count: {}'.format(pred1count))
    printAndAdd('Sensitivity: {}'.format(getMetric(arr1_revert, arr2, 'sensitivity')))
    printAndAdd('Specificity: {}'.format(getMetric(arr1_revert, arr2, 'specificity')))
    printAndAdd('\nREPORT:')
    printAndAdd(classification_report(arr1_revert, arr2))