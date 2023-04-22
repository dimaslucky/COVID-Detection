import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.append('/home/m13518003/Tugas Akhir/Utils')

import Utils
from utils import *
from dataAugment import *
from models import *

import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN

os.environ["CUDA_VISIBLE_DEVICES"]="6"

########## LOAD PICKLE #############
printAndAdd('\n############### LOAD PICKLE ###################', output_lines)
os.chdir('/home/m13518003/Tugas Akhir/Pickles/Raw Audio')

audioFileList = ['Raw_Audio_0.pkl', 'Raw_Audio_1.pkl', 'Raw_Audio_2.pkl', 'Raw_Audio_3.pkl', 'Raw_Audio_4.pkl']
sig_arr, y = pickleLoadAudio(audioFileList)
print("sig: {}".format(sig_arr.shape))
print("y: {}".format(y.shape))

########## DATA SPLITTING #############
printAndAdd('\n############### DATA SPLITTING ###################', output_lines)

sig_arr_np = np.array(sig_arr)
y_arr_np = np.array(y)
printAndAdd(sig_arr_np.shape, output_lines)
printAndAdd(y_arr_np.shape, output_lines)
printAndAdd(Counter(y_arr_np), output_lines)

X_train , X_test , y_train , y_test = train_test_split(sig_arr_np , y_arr_np ,test_size=0.2, random_state=42, stratify=y_arr_np)
X_train , X_train_val , y_train , y_train_val = train_test_split(X_train , y_train ,test_size=0.2, random_state=42, stratify=y_train)

printAndAdd(Counter(y_train), output_lines)
printAndAdd(Counter(y_train_val), output_lines)

########## DATA AUGMENTATION PROCESS #############
printAndAdd('\n############### DATA AUGMENTATION PROCESS ###################', output_lines)

printAndAdd('Train data Before:', output_lines)
printAndAdd(str(X_train.shape) + str(y_train.shape), output_lines)
printAndAdd('Validation data Before:', output_lines)
printAndAdd(str(X_train_val.shape) + str(y_train_val.shape), output_lines)

X_train, y_train = audioManipulate(X_train, y_train)
X_train_val, y_train_val = audioManipulate(X_train_val, y_train_val)

printAndAdd('Train data after augmentation:', output_lines)
printAndAdd(str(X_train.shape) + str(y_train.shape), output_lines)
printAndAdd('Validation data after augmentation:', output_lines)
printAndAdd(str(X_train_val.shape) + str(y_train_val.shape), output_lines)

X_train, y_train = shuffle_list(X_train, y_train)
X_train_val, y_train_val = shuffle_list(X_train_val, y_train_val)
X_test, y_test = shuffle_list(X_test, y_test)

########## FEATURE EXTRACTION #############
printAndAdd('\n############### FEATURE EXTRACTION ###################', output_lines)

X_train, X_train_val, X_test = extract_melspec(X_train, X_train_val, X_test)

X_train, X_test, X_train_val, y_train, y_test, y_train_val = prep_cnn_input(X_train, X_test, X_train_val, y_train, y_test, y_train_val)

printAndAdd("X Train Shape is: " + str(X_train.shape), output_lines)
printAndAdd("y Train Shape is: " + str(y_train.shape), output_lines)
printAndAdd("X Test Shape is: " + str(X_test.shape), output_lines)
printAndAdd("y Test Shape is: " + str(y_test.shape), output_lines)
printAndAdd("X Val Shape is: " + str(X_train_val.shape), output_lines)
printAndAdd("y Val Shape is: " + str(y_train_val.shape), output_lines)

INPUTSHAPE = (X_train.shape[1], X_train.shape[2], 1)

# ########## CNN MODEL #############
print('\n############### CNN MODEL ###################')

model = build_cnn(INPUTSHAPE)
batch_size = 64
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_get_f1', min_delta=0.01, patience=10, verbose=0, mode='max',
    baseline=None, restore_best_weights=False)

history = model.fit(X_train,y_train ,
            validation_data=(X_train_val,y_train_val),
            epochs=100,
            # callbacks = [callback],
            batch_size=batch_size)

make_train_plot('Melspec', 'CNN', history, 0)


# ########## RESNET MODEL #############
print('\n############### RESNET MODEL ###################')

model_resnet = build_resnet(INPUTSHAPE)

batch_size = 64
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_get_f1', min_delta=0.01, patience=10, verbose=0, mode='max',
    baseline=None, restore_best_weights=False)

history_resnet = model_resnet.fit(X_train,y_train ,
            validation_data=(X_train_val,y_train_val),
            epochs=100,
            # callbacks = [callback],
            batch_size=batch_size)

make_train_plot('Melspec', 'Resnet', history_resnet, 1)

# ########## EFFICIENTNET MODEL #############
print('\n############### EFFICIENTNET MODEL ###################')

model_efficientnet = build_efficientnet(INPUTSHAPE)

batch_size = 64
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_get_f1', min_delta=0.01, patience=10, verbose=0, mode='max',
    baseline=None, restore_best_weights=False)

history_efficientnet = model_efficientnet.fit(X_train,y_train ,
            validation_data=(X_train_val,y_train_val),
            epochs=100,
            # callbacks = [callback],
            batch_size=batch_size)

make_train_plot('Melspec', 'EfficientNet', history_efficientnet, 2)

# ########## VGGish MODEL #############
print('\n############### VGGish MODEL ###################')

model_vggish = build_vggish(INPUTSHAPE)

batch_size = 64
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_get_f1', min_delta=0.01, patience=10, verbose=0, mode='max',
    baseline=None, restore_best_weights=False)

history_vggish = model_vggish.fit(X_train,y_train ,
            validation_data=(X_train_val,y_train_val),
            epochs=100,
            # callbacks = [callback],
            batch_size=batch_size)

make_train_plot('Melspec', 'VGGish', history_vggish, 3)

# ########## EVALUATION #############
printAndAdd('\n############### EVALUATION ###################', output_lines)
printAndAdd('Metrics : Loss, Precision, AUC, Recall, F1 Score', output_lines)

printAndAdd('CNN MODEL:', output_lines)
printAndAdd(str(model.evaluate(X_test, y_test)), output_lines)

printAndAdd('\nRESNET MODEL:', output_lines)
printAndAdd(str(model_resnet.evaluate(X_test, y_test)), output_lines)

printAndAdd('\nEFFICIENTNET MODEL:', output_lines)
printAndAdd(str(model_efficientnet.evaluate(X_test, y_test)), output_lines)

printAndAdd('\nVGGISH MODEL:', output_lines)
printAndAdd(str(model_vggish.evaluate(X_test, y_test)), output_lines)

printToFile(output_lines, 'AudioManipulation_Melspec.txt')