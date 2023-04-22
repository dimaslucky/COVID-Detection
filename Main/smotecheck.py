import pandas as pd
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pickle
import joblib
from scipy import stats 
from librosa.util import fix_length
from collections import Counter
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
import tensorflow as tf
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.python.keras.applications.resnet import ResNet50
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import keras.backend as K
from imblearn.combine import SMOTEENN
import scipy

##############################################################################

##### Functions #####
def displayTime(startFrame, endFrame):    
    print(' start time: ' + str(startFrame/sr) + ', end time: ' + str(endFrame/sr))
    
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
    print(Q1)
    print(Q3)
    print(IQR)
    print("higher outlier", high_outlier)
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
        melspec = librosa.feature.melspectrogram(y=sig, sr=sr, n_fft=512, win_length=512, window=scipy.signal.hann(512))
        X_train.append(melspec)
        
    for sig in tqdm(X_test,  desc='Progress'):    
        melspec = librosa.feature.melspectrogram(y=sig, sr=sr, n_fft=512, win_length=512, window=scipy.signal.hann(512))
        temp_test.append(melspec)
        
    for sig in tqdm(X_train_val,  desc='Progress'):    
        melspec = librosa.feature.melspectrogram(y=sig, sr=sr, n_fft=512, win_length=512, window=scipy.signal.hann(512))
        temp_val.append(melspec)

    X_test = temp_test
    X_train_val = temp_val

    return X_train, X_train_val, X_test

def prep_cnn_input(X_train, X_test, X_train_val, y_train_res, y_test, y_train_val):
    X_train = np.array(X_train) 
    X_test = np.array(X_test)
    X_train_val = np.array(X_train_val)

    # y_train_nonCat = y_train_res
    # y_test_nonCat = y_test
    # y_train_val_nonCat = y_train_val

    y_train = tf.keras.utils.to_categorical(y_train_res , num_classes=2)
    y_test = tf.keras.utils.to_categorical(y_test , num_classes=2)
    y_train_val = tf.keras.utils.to_categorical(y_train_val , num_classes=2)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    X_train_val = X_train_val.reshape(X_train_val.shape[0], X_train_val.shape[1], X_train_val.shape[2], 1)

    return X_train, X_test, X_train_val, y_train, y_test, y_train_val

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def build_cnn():
    model =  models.Sequential([
    
        layers.Conv2D(32 , (3,3),activation = 'relu',padding='valid', input_shape = INPUTSHAPE),  
        layers.MaxPooling2D(2, padding='same'),
        layers.Conv2D(128, (3,3), activation='relu',padding='valid'),
        layers.MaxPooling2D(2, padding='same'),
        layers.Dropout(0.5),
        layers.Conv2D(128, (3,3), activation='relu',padding='valid'),
        layers.MaxPooling2D(2, padding='same'),
        layers.Dropout(0.5),
        layers.GlobalAveragePooling2D(),
        layers.Dense(512 , kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation = 'relu'),
        layers.Dropout(0.5),
        layers.Dense(2 , activation = 'sigmoid')
    ])

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', 
                metrics = [tf.keras.metrics.Precision(), tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), get_f1])
    model.summary()
    return model

def build_resnet():
    model_resnet = models.Sequential()
    model_resnet.add(ResNet50(include_top=False, pooling='avg', weights=None, input_shape=INPUTSHAPE))
    model_resnet.add(Flatten())
    model_resnet.add(BatchNormalization())
    model_resnet.add(Dense(512, activation='sigmoid'))
    model_resnet.add(Dropout(0.3))
    model_resnet.add(BatchNormalization())
    model_resnet.add(Dense(32, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='sigmoid'))
    model_resnet.add(Dropout(0.3))
    model_resnet.add(BatchNormalization())
    model_resnet.add(Dense(2, activation='sigmoid'))
    model_resnet.layers[0].trainable = False



    model_resnet.compile(optimizer='adam', loss='categorical_crossentropy', 
                        metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), get_f1])

    return model_resnet

def make_train_plot(model_type, history):
    os.chdir('/home/m13518003/Tugas Akhir/Plots/' + model_type)
    if(model_type == 'CNN'):
        precision_str = 'precision'
        auc_str = 'auc'
        recall_str = 'recall'
    elif(model_type == 'Resnet'):
        precision_str = 'precision_1'
        auc_str = 'auc_1'
        recall_str = 'recall_1'
    
    plt.plot(history.history[precision_str])
    plt.plot(history.history['val_' + precision_str])
    plt.title('model precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(model_type + ' Precision')

    plt.plot(history.history[recall_str])
    plt.plot(history.history['val_' + recall_str])
    plt.title('model recall')
    plt.ylabel('recall')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(model_type + ' Recall')

    plt.plot(history.history['get_f1'])
    plt.plot(history.history['val_get_f1'])
    plt.title('model f1-score')
    plt.ylabel('f1-score')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(model_type + ' F1 Score')


    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(model_type + ' Loss')

    plt.plot(history.history[auc_str])
    plt.plot(history.history['val_' + auc_str])
    plt.title('model auc')
    plt.ylabel('auc')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(model_type + ' AUC')


##### Load Data #####
print('############### LOAD DATA ###################')
os.chdir('/home/m13518003/Tugas Akhir/Dataset/COUGHVID/public_dataset')
path = os.getcwd()
df = pd.read_csv('metadata_compiled_edit.csv')
df = df[df['status'].notna()]
df['status'].replace(['healthy', 'COVID-19', 'symptomatic'],
                        [0, 1, 2], inplace=True)
df = df[df.status != 2]
df = df[df.cough_detected >= 0.9]
# df = df.loc[4000:5000]
print(Counter(df['status']))

##### Match Data with Imported Data ########
listdf = []
for data in tqdm(df.iterrows(),  desc='Progress'):
    filename = df.uuid[data[0]]+str('.wav')
    listdf.append(filename)
print('TOTAL IN DF:')
print(len(listdf))

dir_list = os.listdir(path)
print("TOTAL IN DIRECTORY:") 
print(len(dir_list))

diff_list = list(set(listdf) - set(dir_list))
print('LIST DIFFERENCE: ')
print(len(diff_list))

diff_list_stripped = []
for diff in diff_list:
    temp = diff.replace('.wav', '')
    diff_list_stripped.append(temp)
print(len(diff_list_stripped))

df = df
for val in diff_list_stripped:
    df = df[df.uuid != val]
print(Counter(df['status']))

######## Trimming Process ############
print('\n############### TRIMMING PROCESS ###################')
duration = get_dur(df)
high_outlier = get_outlier(duration)
sig_arr, y = padtrim_audio(df, high_outlier)

# for i in range(0,9000,1000):
#     print("{} - {} = {}".format(i, i+1000, Counter(y[i:i+1000])))

########## DATA SPLITTING #############
print('\n############### DATA SPLITTING ###################')

sig_arr_np = np.array(sig_arr)
y_arr_np = np.array(y)
print(sig_arr_np.shape)
print(y_arr_np.shape)
print(Counter(y_arr_np))

X_train , X_test , y_train , y_test = train_test_split(sig_arr_np , y_arr_np ,test_size=0.2, random_state=42, stratify=y_arr_np)
X_train , X_train_val , y_train , y_train_val = train_test_split(X_train , y_train ,test_size=0.2, random_state=42, stratify=y_train)

########## SMOTE-ENN PROCESS #############
print('\n############### SMOTE-ENN PROCESS ###################')

print("Before OverSampling, counts of y label: {}".format(Counter(y_train)))

def applysmote(X_train, y_train, samplingstrat):
    sm = SMOTEENN(random_state=2, sampling_strategy=samplingstrat)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())
    print('After OverSampling, the shape of X_train: {}'.format(X_train_res.shape))
    print('After OverSampling, the shape of y: {} \n'.format(y_train_res.shape))
    print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
    print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
    return sum(y_train_res==0), sum(y_train_res==1)

txt = "Before OverSampling, counts of y label: {}\n".format(Counter(y_train))
idx = 0.3
for i in range(4, 11):
    idx += 0.1
    print(i)
    print(idx)
    val0, val1 = applysmote(X_train, y_train, idx)
    txt = txt + str(idx) + ': (' + str(val0) + ':' + str(val1) +')\n'

os.chdir('/home/m13518003/Tugas Akhir')

f = open("smoteresult.txt", "w")
f.write(txt)
f.close()