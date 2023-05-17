import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import keras.backend as K
from tensorflow.keras import models, layers
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization, Dropout
# from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.keras.applications import ResNet50, EfficientNetB0, Xception, MobileNetV2, DenseNet121, InceptionResNetV2
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import SGD
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

# opt = SGD(learning_rate = 0.001, nesterov = False, momentum = 0.0)
opt = 'adam'

def prep_cnn_input(X_train, X_test, X_train_val, y_train_res, y_test, y_train_val):
    X_train = np.array(X_train) 
    X_test = np.array(X_test)
    X_train_val = np.array(X_train_val)

    # scaler = MinMaxScaler()
    # X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
    # X_train_val = scaler.transform(X_train_val.reshape(X_train_val.shape[0], -1)).reshape(X_train_val.shape)
    # X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)

    y_train = tf.keras.utils.to_categorical(y_train_res , num_classes=2)
    y_test = tf.keras.utils.to_categorical(y_test , num_classes=2)
    y_train_val = tf.keras.utils.to_categorical(y_train_val , num_classes=2)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    X_train_val = X_train_val.reshape(X_train_val.shape[0], X_train_val.shape[1], X_train_val.shape[2], 1)

    return X_train, X_test, X_train_val, y_train, y_test, y_train_val

def prep_cnn_input_kfold(X_train, X_test, y_train_res, y_test):
    X_train = np.array(X_train) 
    X_test = np.array(X_test)

    # y_train = tf.keras.utils.to_categorical(y_train_res , num_classes=2)
    y_test = tf.keras.utils.to_categorical(y_test , num_classes=2)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    return X_train, X_test, y_train_res, y_test

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def specificity(y_true, y_pred):    
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def build_cnn(inputshape):
    model =  models.Sequential([
    
        layers.Conv2D(32 , (3,3),activation = 'relu',padding='valid', input_shape = inputshape),  
        layers.MaxPooling2D(2, padding='same'),
        layers.Conv2D(128, (3,3), activation='relu',padding='valid'),
        layers.MaxPooling2D(2, padding='same'),
        layers.Dropout(0.7),
        layers.Conv2D(128, (3,3), activation='relu',padding='valid'),
        layers.MaxPooling2D(2, padding='same'),
        layers.Dropout(0.7),
        layers.GlobalAveragePooling2D(),
        layers.Dense(512 , kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation = 'relu'),
        layers.Dropout(0.7),
        # layers.Dense(2 , activation = 'sigmoid')
        layers.Dense(2 , activation = 'softmax')
    ])

    model.compile(loss = 'binary_crossentropy', optimizer = opt, 
                metrics = [tf.keras.metrics.Precision(), tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), get_f1, specificity])
    model.summary()
    return model

def build_resnet(inputshape):
    model_resnet = ResNet50(include_top=True, pooling='avg', weights=None, input_shape=inputshape, classes=2, classifier_activation='softmax')
    # model_resnet = models.Sequential()
    # model_resnet.add(ResNet50(include_top=False, pooling='avg', weights='imagenet', input_shape=inputshape))
    # model_resnet.add(Flatten())
    # model_resnet.add(BatchNormalization())
    # model_resnet.add(Dense(512, activation='relu'))
    # model_resnet.add(Dropout(0.7))
    # model_resnet.add(BatchNormalization())
    # model_resnet.add(Dense(32, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu'))
    # model_resnet.add(Dropout(0.7))
    # model_resnet.add(BatchNormalization())
    # model_resnet.add(Dense(2, activation='softmax'))

    model_resnet.compile(optimizer=opt, loss='binary_crossentropy', 
                        metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), get_f1, specificity])

    return model_resnet

def build_vggish(inputshape):
    model_vggish = models.Sequential()
    axis = -1
    alpha = 0.1
    scale = 16
    reg = None

    def conv_block(model, _scale, axis, _inputshape=inputshape):
        model.add(layers.Conv2D(_scale, (3,3), padding="same", input_shape=_inputshape, kernel_regularizer=reg, activation='relu'))
        model.add(BatchNormalization(axis=axis))
        # conv_2
        model.add(layers.Conv2D(_scale, (3, 3), padding="same", kernel_regularizer=reg, activation='relu'))
        model.add(BatchNormalization(axis=axis))
        # pool_1
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.7))
        return model
    
    model_vggish = conv_block(model_vggish, scale, axis, _inputshape=inputshape)
    model_vggish = conv_block(model_vggish, scale*(2**1), axis)
    model_vggish = conv_block(model_vggish, scale*(2**2), axis)
    model_vggish = conv_block(model_vggish, scale*(2**3), axis)
    model_vggish.add(Flatten())
    # FC1
    model_vggish.add(Dense(256, kernel_regularizer=reg, activation='relu'))
    model_vggish.add(Dropout(0.7))
    # FC2
    model_vggish.add(Dense(256, kernel_regularizer=reg, activation='relu'))
    model_vggish.add(Dropout(0.7))
    
    # classifier
    # model_vggish.add(Dense(2, activation='sigmoid'))
    model_vggish.add(Dense(2, activation='softmax'))

    model_vggish.compile(optimizer=opt, loss='binary_crossentropy', 
                        metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), get_f1, specificity])

    return model_vggish

def build_efficientnet(inputshape):
    model_efficient = EfficientNetB0(weights=None, include_top=True, input_shape=inputshape, classes=2)
    # effnet_layers = EfficientNetB0(weights=None, include_top=False, input_shape=inputshape)
    # model_efficient = models.Sequential()
    # model_efficient.add(effnet_layers)
    # model_efficient.add(Flatten())
    # model_efficient.add(BatchNormalization())
    # model_efficient.add(Dense(512, activation='relu'))
    # model_efficient.add(Dropout(0.7))
    # model_efficient.add(BatchNormalization())
    # model_efficient.add(Dense(32, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu'))
    # model_efficient.add(Dropout(0.7))
    # model_efficient.add(BatchNormalization())
    # model_efficient.add(Dense(2, activation='softmax'))

    model_efficient.compile(optimizer=opt, loss='binary_crossentropy', 
                        metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), get_f1, specificity])
    
    return model_efficient

def build_inceptionResnet(inputshape):
    model_inceptionResnet = InceptionResNetV2(weights=None, include_top=True, input_shape=inputshape, classes=2)

    model_inceptionResnet.compile(optimizer=opt, loss='binary_crossentropy', 
                        metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), get_f1, specificity])

    return model_inceptionResnet

def build_mobilenet(inputshape):
    model_mobilenet = MobileNetV2(weights=None, include_top=True, input_shape=inputshape, classes=2)

    model_mobilenet.compile(optimizer=opt, loss='binary_crossentropy', 
                        metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), get_f1, specificity])

    return model_mobilenet

def build_densenet(inputshape):
    model_densenet = DenseNet121(weights=None, include_top=True, input_shape=inputshape, classes=2)

    model_densenet.compile(optimizer=opt, loss='binary_crossentropy', 
                        metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), get_f1, specificity])

    return model_densenet

def build_xception(inputshape):
    model_xception = Xception(weights=None, include_top=True, input_shape=inputshape, classes=2)

    model_xception.compile(optimizer=opt, loss='binary_crossentropy', 
                        metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.AUC(), tf.keras.metrics.Recall(), get_f1, specificity])
    
    return model_xception

def make_train_plot(feature_type, model_type, history, iteration):
    os.chdir('/home/m13518003/Tugas Akhir/Plots/' + feature_type + '/' + model_type)
    
    if(iteration == 0):
        precision_str = 'precision'
        auc_str = 'auc'
        recall_str = 'recall'
    else:
        precision_str = 'precision_' + str(iteration)
        auc_str = 'auc_' + str(iteration)
        recall_str = 'recall_' + str(iteration)
    
    plt.plot(history.history[precision_str])
    plt.plot(history.history['val_' + precision_str])
    plt.title('model precision')
    plt.ylabel('precision')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    strFile = model_type + ' Precision'
    if os.path.isfile(strFile):
        os.remove(strFile)
    plt.savefig(strFile)
    plt.clf()

    plt.plot(history.history[recall_str])
    plt.plot(history.history['val_' + recall_str])
    plt.title('model recall')
    plt.ylabel('recall')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    strFile = model_type + ' Recall'
    if os.path.isfile(strFile):
        os.remove(strFile)
    plt.savefig(strFile)
    plt.clf()

    plt.plot(history.history['get_f1'])
    plt.plot(history.history['val_get_f1'])
    plt.title('model f1-score')
    plt.ylabel('f1-score')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    strFile = model_type + ' F1 Score'
    if os.path.isfile(strFile):
        os.remove(strFile)
    plt.savefig(strFile)
    plt.clf()

    plt.plot(history.history['specificity'])
    plt.plot(history.history['val_specificity'])
    plt.title('model specificity')
    plt.ylabel('specificity')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    strFile = model_type + ' Specificity'
    if os.path.isfile(strFile):
        os.remove(strFile)
    plt.savefig(strFile)
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    strFile = model_type + ' Loss'
    if os.path.isfile(strFile):
        os.remove(strFile)
    plt.savefig(strFile)
    plt.clf()

    plt.plot(history.history[auc_str])
    plt.plot(history.history['val_' + auc_str])
    plt.title('model auc')
    plt.ylabel('auc')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    strFile = model_type + ' AUC'
    if os.path.isfile(strFile):
        os.remove(strFile)
    plt.savefig(strFile)
    plt.clf()

def kfold_fit(X_train, y_train, model, batch_size):
    k_fold = StratifiedKFold(n_splits=10, random_state=12, shuffle=True)
    val_f1_scores = []
    val_loss_scores = []
    i=1
    for k_train_index, k_test_index in k_fold.split(X_train, y_train):
        print('############# {} of KFold {} #############'.format(i,k_fold.n_splits))
        xtr, xvl = X_train[k_train_index], X_train[k_test_index]
        ytr, yvl = y_train[k_train_index], y_train[k_test_index]
        # y_train = tf.keras.utils.to_categorical(y_train_res, num_classes=2)
        ytr, yvl = tf.keras.utils.to_categorical(y_train[k_train_index], num_classes=2), tf.keras.utils.to_categorical(y_train[k_test_index], num_classes=2)


        history = model.fit(xtr, ytr,
                    epochs=100,
                    batch_size=batch_size)

        scores = model.evaluate(xvl,yvl)
        val_f1_scores.append(scores[4])
        val_loss_scores.append(scores[0])
        # printAndAdd(str(scores), output_lines)
        i+=1

    return val_f1_scores, val_loss_scores