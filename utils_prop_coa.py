#from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout,BatchNormalization,LeakyReLU,GlobalAveragePooling2D
#from keras.optimizers import SGD, Adam, RMSprop
#from keras.models import Model, load_model
#from keras.callbacks import ModelCheckpoint, EarlyStopping
#from keras.preprocessing.image import ImageDataGenerator
#from keras.callbacks import CSVLogger
#from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout,BatchNormalization,LeakyReLU,UpSampling2D,GlobalAveragePooling2D
from tensorflow.keras import applications
from tensorflow.keras import backend as K
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.models import Model, load_model
from glob import glob
from os.path import join,basename 
import csv
from sklearn.model_selection import train_test_split

import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

from glob import glob

import tensorflow as tf
#import utils_prop_coa
#from class_callback import TestCallback

import sys

from classe_cifar_load import *
from classe_cifar_load_100 import *
#from utils_llp_models_new import *
from utils_loss_function import *
from mnist_dataset import *


def handle_one(args):
    print('cifar 10')
    return LOAD_CIFAR10(noise_rate=args.noise_rate,noise_type=args.noise_type)
  

def handle_two(args):
    print('cifar 100')
    return LOAD_CIFAR100(noise_rate=args.noise_rate,noise_type=args.noise_type)
  

def handle_three(args):
    print('Mnist')
    return LOAD_MNIST(noise_rate=args.noise_rate,noise_type=args.noise_type)


  #cifar_dataset=LOAD_CIFAR100(noise_rate=noise_rate)
  #cifar_dataset=LOAD_MNIST(noise_rate=noise_rate)




def make_net_model(input_shape):

    input_shape= Input(input_shape)
    
    #x = Dense(256*256*2,activation=tf.keras.layers.LeakyReLU(alpha=0.2))(input_noise)
   
    #x=layers.Reshape((256,256,2))(x)
    
    x=layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same',activation=tf.keras.layers.LeakyReLU(alpha=0.01))(input_shape)
    x=tf.keras.layers.BatchNormalization(momentum=0.1,epsilon=0.00001)(x)
    x=layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same',activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
    x=tf.keras.layers.BatchNormalization(momentum=0.1,epsilon=0.00001)(x)
    x=layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same',activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
    x=tf.keras.layers.BatchNormalization(momentum=0.1,epsilon=0.00001)(x)
    x=MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
    x=Dropout(0.25)(x)

    x=layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same',activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
    x=tf.keras.layers.BatchNormalization(momentum=0.1,epsilon=0.00001)(x)
    x=layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same',activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
    x=tf.keras.layers.BatchNormalization(momentum=0.1,epsilon=0.00001)(x)
    x=layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same',activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
    x=tf.keras.layers.BatchNormalization(momentum=0.1,epsilon=0.00001)(x)
    x=MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
    x=Dropout(0.25)(x)

    x=layers.Conv2D(512, (3, 3), strides=(1, 1), padding='valid',activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
    x=tf.keras.layers.BatchNormalization(momentum=0.1,epsilon=0.00001)(x)
    x=layers.Conv2D(256, (3, 3), strides=(1, 1), padding='valid',activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
    x=tf.keras.layers.BatchNormalization(momentum=0.1,epsilon=0.00001)(x)
    x=layers.Conv2D(128, (3, 3), strides=(1, 1), padding='valid',activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
    x=tf.keras.layers.BatchNormalization(momentum=0.1,epsilon=0.00001)(x)
    features=tf.keras.layers.GlobalAveragePooling2D()(x)
    #x=tf.keras.layers.Flatten()(x)
    #x = Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)


    #pred = Dense(10, activation='softmax')(features)
    
    #original ok 
    
    logistic = Dense(10)(features)
    pred=tf.nn.softmax(logistic)
    
    # 100 ou 10 depende do dataset!

    #soft=tf.keras.activations.softmax(x, axis=-1)
    custom_model = Model(input_shape,outputs=[pred,logistic])
    
    custom_model.summary()
    return custom_model


def load_net():
    # If you want to use your our weight pre trained##
    #weights_res='/share_alpha/Submarina/pre_treined_features/new_dataset/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    #resnet_50_model = applications.ResNet50(weights='imagenet', include_top=True)#,input_tensor=input_tensor)
    
    #mobile = applications.MobileNet(weights=None,include_top=False,input_shape =(32,32,3))
    #mobile = applications.MobileNet(weights='imagenet')
    
    mobile=applications.densenet.DenseNet121(weights=None,include_top=False,input_shape =(32,32,3))
    mobile.summary()
    return mobile



def f1_score_2(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))      
def custom_loss(weights):    
    weights = K.variable(weights)
    
    def loss(y_true, y_pred):
      # scale predictions so that the class probas of each sample sum to 1
      y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
      # clip to prevent NaN's and Inf's
      y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
      loss = y_true * K.log(y_pred) + (1-y_true) * K.log(1-y_pred)
      loss = loss * weights 
      loss = - K.mean(loss, -1)
      return loss
    return loss
def metrics_f1_recal_pre_mat_calculate(rede_1,rede_2,epoch,datagenerator,max_f1,args,acertos_porcentagem,total_de_img_selecionads,acertos_porcentagem_2,total_de_img_selecionads_2, conjunto='valid'):
    '''
        Calculo das metricas f1, recal, precision e matriz confusão no conjunto treino, 
        valid ou teste.

        Conjunto 'valid' ou 'Treino' 

    '''
    #conjunto
    dc=0
    valid_labels=[]
    predict_valid__all_img=[]
    predict_valid__all_img_2=[]
    valid_num=int(len(datagenerator))
    for dc in range(valid_num):
      sys.stdout.write('\r Avaliando a rede : ' + str(dc) + ' of ' + str(valid_num))
      sys.stdout.flush()

      valid_images_labels_pro=datagenerator.__next__() 

      valid_img=valid_images_labels_pro[0]
            #print(valid_img.shape)
      valid_labels+=[ valid_images_labels_pro[1]  ]

      predict_val = rede_1(valid_img, training=True)[0]

      predict_val_2 = rede_2(valid_img, training=True)[0]

      predict_val = np.rint(predict_val)

      predict_val_2 = np.rint(predict_val_2)

      predict_valid__all_img+=[predict_val]

      predict_valid__all_img_2+=[predict_val_2]
                
    valid_labels=np.concatenate(valid_labels)
    predict_valid__all_img=np.concatenate(predict_valid__all_img)
    predict_valid__all_img_2=np.concatenate(predict_valid__all_img_2)
    #print('Shape label : \n')
                #print(predict_val.shape)
                #print(predict_valid__all_img)

                #valid_labels=np.array(valid_labels)
                #predict_valid__all_img=np.array(predict_valid__all_img)

    print(' \n Calculando Metricas')
                #valid_labels=np.squeeze(valid_labels, axis=0)
                #predict_valid__all_img=np.squeeze(predict_valid__all_img, axis=0)
    print(valid_labels.shape)
    print(predict_valid__all_img.shape)
    print(valid_labels[0])
    print(predict_valid__all_img[0])
    f1=f1_score(valid_labels, predict_valid__all_img,average='macro')
    recal=recall_score(valid_labels, predict_valid__all_img,average='macro')
    precision=precision_score(valid_labels, predict_valid__all_img,average='macro')
    acc=accuracy_score(valid_labels, predict_valid__all_img, normalize=True)

    f1_2=f1_score(valid_labels, predict_valid__all_img_2,average='macro')
    recal_2=recall_score(valid_labels, predict_valid__all_img_2,average='macro')
    precision_2=precision_score(valid_labels, predict_valid__all_img_2,average='macro')
    acc2=accuracy_score(valid_labels, predict_valid__all_img_2, normalize=True)  
    print(' DATASET :', conjunto)
    print(' f1 = {}  , recal = {} , precision = {} , acc ={}'.format( f1,recal,precision,acc))
    print('\n')

    print(' f1_2 = {}  , recal_2 = {} , precision_2 = {}, acc ={}'.format( f1_2,recal_2,precision_2,acc2))
    print('\n')

    matrix_conf=confusion_matrix(valid_labels.argmax(axis=1),predict_valid__all_img.argmax(axis=1))
    print(matrix_conf)
            ### Liberando a memoria
    
    #file1 = open(args.results_path+'/'+args.dataset+'_'+str(args.noise_rate)+'_'+str(args.noise_type)+'.txt',"a")#write mode
    #file1.write(' DATASET : {}'.format (conjunto))
    #file1.write('\n')
    #file1.write(' f1 = {}  , recal = {} , precision = {} , acc ={}'.format( f1,recal,precision,acc))
    #file1.write('\n')
    #file1.write(' f1 = {}  , recal = {} , precision = {}, acc ={}'.format( f1_2,recal_2,precision_2,acc2))
    #file1.write('\n')

    #file1.close()
    if conjunto=='valid':            
        if epoch==0:
            fields=['epoch','f1','recal','precision','acc','relabel_acertos','relabel_total','porcentagem_relabel']
            with open(args.results_path+'/'+args.dataset+'_'+str(args.noise_rate)+'_'+str(args.noise_type)+'valid_1'+'.csv', 'a') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(fields)
                csvfile.close()
            fields=['epoch','f1','recal','precision','acc','relabel_acertos','relabel_total','porcentagem_relabel']
            with open(args.results_path+'/'+args.dataset+'_'+str(args.noise_rate)+'_'+str(args.noise_type)+'valid_2'+'.csv', 'a') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(fields)
                csvfile.close()
        else:
            fields=[epoch,f1,recal,precision,acc,acertos_porcentagem.numpy(),total_de_img_selecionads.numpy(),(acertos_porcentagem/total_de_img_selecionads).numpy()]
            with open(args.results_path+'/'+args.dataset+'_'+str(args.noise_rate)+'_'+str(args.noise_type)+'valid_1'+'.csv', 'a') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(fields)
                csvfile.close()
            fields=[epoch,f1_2,recal_2,precision_2,acc2,acertos_porcentagem_2.numpy(),total_de_img_selecionads_2.numpy(),(acertos_porcentagem_2/total_de_img_selecionads_2).numpy()]
            with open(args.results_path+'/'+args.dataset+'_'+str(args.noise_rate)+'_'+str(args.noise_type)+'valid_2'+'.csv', 'a') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(fields)
                csvfile.close()
    if conjunto=='treino':            
        if epoch==0:
            fields=['epoch','f1','recal','precision','acc','relabel_acertos','relabel_total','porcentagem_relabel']
            with open(args.results_path+'/'+args.dataset+'_'+str(args.noise_rate)+'_'+str(args.noise_type)+'treino_1'+'.csv', 'a') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(fields)
                csvfile.close()
            with open(args.results_path+'/'+args.dataset+'_'+str(args.noise_rate)+'_'+str(args.noise_type)+'treino_2'+'.csv', 'a') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(fields)
                csvfile.close()

        else:
            fields=[epoch,f1,recal,precision,acc,acertos_porcentagem.numpy(),total_de_img_selecionads.numpy(),(acertos_porcentagem/total_de_img_selecionads).numpy()]
            with open(args.results_path+'/'+args.dataset+'_'+str(args.noise_rate)+'_'+str(args.noise_type)+'treino_1'+'.csv', 'a') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(fields)
                csvfile.close()
            fields=[epoch,f1_2,recal_2,precision_2,acc2,acertos_porcentagem_2.numpy(),total_de_img_selecionads_2.numpy(),(acertos_porcentagem_2/total_de_img_selecionads_2).numpy()]
            with open(args.results_path+'/'+args.dataset+'_'+str(args.noise_rate)+'_'+str(args.noise_type)+'treino_2'+'.csv', 'a') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(fields)
                csvfile.close()

    if  f1> max_f1:

        if conjunto=='valid':
            name_gen=args.results_path+'/model_rede1.h5'
            name_disc=args.results_path+'/model_rede2.h5'

            rede_1.save(name_gen)
            rede_2.save(name_disc)
        max_f1=f1
    return  max_f1             
    #valid_labels=[]
    #predict_valid__all_img=[]

