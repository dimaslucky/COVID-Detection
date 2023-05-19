import sys
sys.path.append('/home/m13518003/Tugas Akhir/Utils/utils')
from utils import *

import numpy as np
import librosa
import pyroomacoustics as pra
from pyroomacoustics.directivities import DirectivityPattern, DirectionVector, CardioidFamily


def white_noise(y):
    wn = np.random.randn(len(y))
    # y_wn = y + np.random.uniform(0, 0.005)*wn
    y_wn = y + 0.002*wn
    return y_wn

def time_shift(y):
    y = np.roll(y, np.random.randint(-10000,10000))
    return y

def Gain(y):
    y = y + np.random.uniform(-0.2,0.2)*y
    return y       
                            
def stretch(y, rate=np.random.uniform(0.8,1.2)):
    y = librosa.effects.time_stretch(y, rate)
    return y

def reverb(y):
    rt60 = 0.5  # seconds
    room_dim = [5, 5, 5]  # meters
    
    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

    # Create the room
    room = pra.ShoeBox(
        room_dim, fs=16000, materials=pra.Material(e_absorption), max_order=max_order,
        use_rand_ism = True, max_rand_disp = 0.05
    )
    
    room.add_source([2.5, 3.73, 1.76], signal=y, delay=0)
    
    dir_obj = CardioidFamily(
        orientation=DirectionVector(azimuth=90, colatitude=15, degrees=True),
        pattern_enum=DirectivityPattern.HYPERCARDIOID,
    )
    room.add_microphone(loc=[2.5, 5, 1.76], directivity=dir_obj)
    room.compute_rir()
    room.simulate()
    return room.mic_array.signals[0]

def audioManipulate(sig_list, y_list):
    sr = 16000
    high_outlier = 4.768000000000001
    pos_idx = []
    augmented_sig = []
    for i in range(len(y_list)):
        if str(y_list[i]) == '1':
            pos_idx.append(i)

    augmented_temp = []
    y_temp = []

    for i in pos_idx:
        sig = sig_list[i]
        
        #### AUGMENTATION METHOD 1 ####
        # chance = random.randint(0,100)
        # if(chance <= 20):
        #     augmented_temp.append(stretch(sig))
        # if(chance <= 40):
        #     augmented_temp.append(time_shift(sig))
        # if(chance <= 60):
        #     augmented_temp.append(reverb(white_noise(sig)))
        # if(chance <= 80):
        #     augmented_temp.append(reverb(sig))
        # if(chance <= 100):
        #     augmented_temp.append(white_noise(sig))

        #### AUGMENTATION METHOD 2 ####
        chance = random.randint(0,100)
        if(chance <= 20):
            augmented_temp.append(stretch(sig))
        elif(chance <= 40):
            augmented_temp.append(time_shift(sig))
        elif(chance <= 60):
            augmented_temp.append(reverb(white_noise(sig)))
        elif(chance <= 80):
            augmented_temp.append(reverb(sig))
        elif(chance <= 100):
            augmented_temp.append(white_noise(sig))

        ### AUGMENTATION METHOD 3 ####
        # chance = random.randint(0,100)
        # if(chance <= 20):
        #     augmented_temp.append(white_noise(Gain(sig)))
        # if(chance <= 40):
        #     augmented_temp.append(reverb(Gain(sig)))
        # if(chance <= 60):
        #     augmented_temp.append(Gain(sig))
        # if(chance <= 80):
        #     augmented_temp.append(stretch(sig))
        # if(chance <= 100):
        #     augmented_temp.append(time_shift(sig))
        # augmented_temp.append(reverb(sig))
        # augmented_temp.append(white_noise(sig))

        #### AUGMENTATION METHOD 4 ####
        # chance = random.randint(0,100)
        # if(chance <= 25):
        #     augmented_temp.append(stretch(sig))
        # if((chance > 25) and (chance <= 50)):
        #     augmented_temp.append(time_shift(sig))
        # if((chance > 50) and (chance <= 75)):
        #     augmented_temp.append(white_noise(sig))
        # if(chance > 75):
        #     augmented_temp.append(reverb(sig))

        #### AUGMENTATION METHOD 5 ####
        # augmented_temp.append(stretch(sig))
        # augmented_temp.append(time_shift(sig))
        # augmented_temp.append(Gain(sig))
        # augmented_temp.append(white_noise(sig))
        # augmented_temp.append(reverb(sig))
        # augmented_temp.append(reverb(white_noise(sig)))
        # augmented_temp.append(reverb(Gain(sig)))
        # augmented_temp.append(white_noise(Gain(sig)))

        #### AUGMENTATION METHOD 6 ####
        # augmented_temp.append(stretch(sig))
        # augmented_temp.append(time_shift(sig))
        # augmented_temp.append(white_noise(sig))
        # augmented_temp.append(reverb(sig))
        # augmented_temp.append(reverb(white_noise(sig)))

    printAndAdd('Total of generated data:', output_lines)
    printAndAdd(len(augmented_temp), output_lines)

    for sig in augmented_temp:
        if(len(sig) < high_outlier*sr):
            sig = librosa.util.pad_center(sig, size=int(high_outlier*sr), mode='constant')
        else:
            sig = librosa.util.fix_length(sig, int(high_outlier*sr))
        augmented_sig.append(sig)
        y_temp.append(1)
    
    augmented_sig = np.array(augmented_sig)
    y_temp = np.array(y_temp)
    sig_list = np.concatenate((sig_list, augmented_sig), axis=0)
    y_list = np.concatenate((y_list, y_temp), axis=0)

    return sig_list, y_list