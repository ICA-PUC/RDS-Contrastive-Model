import tensorflow as tf
import numpy as np

#import load_svhn
from tensorflow.keras.models import Model, load_model
import sys

from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout,BatchNormalization,LeakyReLU,UpSampling2D,GlobalAveragePooling2D
import sys
import argparse

#from utils import load_norm, load_image, balance_coords, save_metrics
#from utils import set_logger, save_dict_to_json, Params, check_folder, Results

#from models import cnn, Metrics, f1_mean
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD, Adadelta, Adagrad, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model, load_model
#import keras
#from sklearn.cluster import KMeans
#import gc
#from loss import categorical_focal_loss
from sklearn.metrics import confusion_matrix,f1_score ,recall_score ,precision_score
#from generator_tf2_lam import DataGenerator_lambda
import os

#from utils_llp_models_new import *

from utils_prop_coa import * 
#from class_callback import TestCallback
from classe_cifar_load import *
from classe_cifar_load_100 import *
#from utils_llp_models_new import *
from utils_loss_function import *
from mnist_dataset import *
def main(args):
  '''
  Train Model for tensorflow 2. Used at work:

  
  Retrieve Discard Samples -Contrastive Learning
  '''
  print(args.dataset)
  if args.dataset=='cifar10':
    cifar_dataset=LOAD_CIFAR10(noise_rate=args.noise_rate,noise_type=args.noise_type)
  elif args.dataset=='cifar100':
    cifar_dataset=LOAD_CIFAR100(noise_rate=args.noise_rate,noise_type=args.noise_type)
  elif args.dataset=='mnist':
    cifar_dataset=LOAD_MNIST(noise_rate=args.noise_rate,noise_type=args.noise_type)

  #options = {'cifar10': handle_one(args), 'cifar100': handle_two(args), 'mnist': handle_three(args)}
  print('Criando Pasta para results')
  if not os.path.exists(args.results_path):
    #os.mkdir(args.results_path)
    os.makedirs(args.results_path)
  #cifar_dataset=options[args.dataset]

  batch_size=args.batch_size
  
  #datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)
  #datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.densenet.preprocess_input)
  datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

  datagenerator=datagen.flow(cifar_dataset.train_data,cifar_dataset.train_noisy_labels, batch_size=batch_size,seed=3,shuffle=False)

  datagenerator_corret_label=datagen.flow(cifar_dataset.train_data,cifar_dataset.train_labels, batch_size=batch_size,seed=3,shuffle=False)
  datagenerator_teste=datagen.flow(cifar_dataset.test_data,cifar_dataset.test_labels, batch_size=batch_size,seed=3,shuffle=False)


  # carregando as duas redes Mobile 
  #net1 = load_net()
  #net2 = load_net()

  net1 = make_net_model((32,32,3))
  net2 = make_net_model((32,32,3))


  @tf.function
  def train_step(images,Y_mb,R_T,ratio,Correct_labels,acertos_porcentagem,total_de_img_selecionads,epoch_ammount,acertos_porcentagem_2,total_de_img_selecionads_2):
      #noise = tf.random.normal([num_examples_to_generate, noise_dim,noise_dim,1])
        #noise = tf.random.normal([num_examples_to_generate, noise_dim_Z])
    #global acertos_porcentagem
    
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    
      #generated_images = generator(noise, training=True)
            
      
      #d_net_fake, d_net_fake_feat = discriminator(generated_images, training=True)
      #Previsao da rede
      #tf.print(images.shape)
      batch_size=images.shape[0]
      #tf.print(batch_size)
      rede1_predict_p_f= rede_1(images, training=True)
      rede2_predict_p_f= rede_2(images, training=True)
     
      #d_net_real, d_net_real_feat = rede_1(images, training=True)
      
                    
      #y_prob_mb = tf.cast(y_prob_mb, tf.float32)
      rede1_predict = tf.cast(rede1_predict_p_f[0], tf.float32)
      rede2_predict = tf.cast(rede2_predict_p_f[0], tf.float32)
      
      #rede1_features = tf.cast(rede1_predict_p_f[1], tf.float32)
      
      #rede2_features = tf.cast(rede2_predict_p_f[1], tf.float32)
      #rede2_features=rede2_predict[1]
      
      #tf.print('antes pred',rede1_predict.get_shape().as_list())

      #tf.print('antes features',rede1_features.get_shape().as_list())



      
      
      
      
      vetor_loss_rede1=Loss_cross_entropy(Y_mb,rede1_predict)


      vetor_loss_rede2=Loss_cross_entropy(Y_mb,rede2_predict)
      

      Img_selecionadas_rede1,labels_selecionadas_rede1=Small_Loss_select(images,vetor_loss_rede1,R_T,batch_size,Y_mb)

      Img_selecionadas_rede2,labels_selecionadas_rede2=Small_Loss_select(images,vetor_loss_rede2,R_T,batch_size,Y_mb)

     
      images_discard_rede1,labels_discard_rede1,Correct_labels1=Small_Loss_select_discard(images,vetor_loss_rede1,R_T,batch_size,Y_mb,Correct_labels)

      images_discard_rede2,labels_discard_rede2,Correct_labels2=Small_Loss_select_discard(images,vetor_loss_rede2,R_T,batch_size,Y_mb,Correct_labels)
      


      #### GET LABEL BY DINO APROACH

      ##### Primeiro Treinando o KNN
      
      #model_knn=KNN_model(labels_selecionadas_rede1,labels_selecionadas_rede2,features_selecionadas_rede1,features_selecionadas_rede2)
      #### Get Predict ####

      #pseudo_label_1,images_discard_rede1,acertos_porcentagem,total_de_img_selecionads=predict_KNN(model_knn,images_discard_rede1,Correct_labels1,acertos_porcentagem,total_de_img_selecionads,labels_discard_rede1,features_discard_rede_1)
      #pseudo_label_2,images_discard_rede2,acertos_porcentagem_2,total_de_img_selecionads_2=predict_KNN(model_knn,images_discard_rede2,Correct_labels2,acertos_porcentagem_2,total_de_img_selecionads_2,labels_discard_rede2,features_discard_rede_2)
      #(images_discard_rede1)
      
      #=model_knn.predict(images_discard_rede2)

      #tf.print('antes pred',images_discard_rede1.get_shape().as_list())
      #tf.print('antes pred',images_discard_rede2.get_shape().as_list())

      img_rede1_aug_1,img_rede1_aug_2,img_rede1_aug_3,img_rede1_aug_4,img_rede1_aug_5,img_rede1_aug_6,img_rede1_aug_7,img_rede1_aug_8=augmentations(images_discard_rede1)
      img_rede2_aug_1,img_rede2_aug_2,img_rede2_aug_3,img_rede2_aug_4,img_rede2_aug_5,img_rede2_aug_6,img_rede2_aug_7,img_rede2_aug_8=augmentations(images_discard_rede2)

      
      # A rede 2 ira atribuir os pseudo labels das imgs descartadas da rede 1, apos dataaugmentation.
      #

      pseudo_label_1,images_discard_rede1,acertos_porcentagem,total_de_img_selecionads=predict_aug_images(rede_2,rede_1,img_rede1_aug_1,img_rede1_aug_2,img_rede1_aug_3,img_rede1_aug_4,img_rede1_aug_5,img_rede1_aug_6,img_rede1_aug_7,img_rede1_aug_8,images_discard_rede1,Correct_labels1,acertos_porcentagem,total_de_img_selecionads,labels_discard_rede1,args.aug_time,args.threshold)
      pseudo_label_2,images_discard_rede2,acertos_porcentagem_2,total_de_img_selecionads_2=predict_aug_images(rede_1,rede_2,img_rede2_aug_1,img_rede2_aug_2,img_rede2_aug_3,img_rede2_aug_4,img_rede2_aug_5,img_rede2_aug_6,img_rede2_aug_7,img_rede2_aug_8,images_discard_rede2,Correct_labels2,acertos_porcentagem_2,total_de_img_selecionads_2,labels_discard_rede2,args.aug_time,args.threshold)

     
      #d_net_real, d_net_real_feat = rede_1(images, training=True)
      #tf.print(rede1_predict)
                    
      
      

      rede1_predict_discard_image= rede_1(images_discard_rede1, training=True)[0]
      rede2_predict_discard_image= rede_2(images_discard_rede2, training=True)[0]
      


      #calculando a proporcoa no conjunto de small loss
      
      Loss_rede1_pseudo_label= Final_loss(rede1_predict_discard_image,pseudo_label_1)
      #tf.print(rede1_predict_discard_image)
      #tf.print(pseudo_label_1)
      #Loss_rede1_pseudo_label=Loss_cross_entropy_exp(pseudo_label_1,rede1_predict_discard_image)
      #.25,2 Funcionou ...
      #0.9,-0.2 nan...
      
      #tf.print(rede2_predict_discard_image.shape)
      #tf.print(pseudo_label_1.shape)
      Loss_rede2_pseudo_label= Final_loss(rede2_predict_discard_image,pseudo_label_2)

      #Loss_rede2_pseudo_label=Loss_cross_entropy_exp(pseudo_label_2,rede2_predict_discard_image)
      #tf.print(Loss_rede2_pseudo_label)
      #Loss_rede1_pseudo_label=discriminator_loss(rede1_predict_discard_image,pseudo_label_2)
      #tf.print('valor da loss',Loss_rede1_pseudo_label)

      #Loss_rede2_pseudo_label=discriminator_loss(rede2_predict_discard_image,pseudo_label_1)
      #tf.print('valor da loss 2',Loss_rede2_pseudo_label)
      # O treinamento da rede é realizado de forma invertida
      # Ou seja as imgs com menos Loss na rede1 sao utilizadas na rede 2 e vice e versa
      
      #print(Img_selecionadas_rede2.shape)
      rede1_predict_small_loss= rede_1(Img_selecionadas_rede2, training=True)[0]
      rede2_predict_small_loss= rede_2(Img_selecionadas_rede1, training=True)[0]

      # Atencao novamente a previsao invertida. rede 1 com labels da rede 2, pois a img selecionadas da rede 2
      # que sao utilizadas no treino da rede 1 e vice e versa
     

     
      #Atencao o Label aqui deve ser referente as imgs selecionadas. No caso os labels invertidos
      Loss_final_rede1= Final_loss_with_pseudo_labels(rede1_predict_small_loss,labels_selecionadas_rede2,Loss_rede1_pseudo_label,R_T,ratio,epoch_ammount)
      
      Loss_final_rede2= Final_loss_with_pseudo_labels(rede2_predict_small_loss,labels_selecionadas_rede1,Loss_rede2_pseudo_label,R_T,ratio,epoch_ammount)
      
      ################################## ############################
      ############################################################

      ####################### IMPLEMENTACAO DINO#####################################
      # Mais facil colocar todas as imgs , Limpas e noise , porem posso escolher apenas um conjunto...

      dino_aug_1,dino_aug_2=augmentations_dino(images)

      rede1_predict_dino_aug1= rede_1(dino_aug_1, training=True)[1]
      rede1_predict_dino_aug2= rede_1(dino_aug_2, training=True)[1]

      rede2_predict_dino_aug1= rede_2(dino_aug_1, training=True)[1]
      rede2_predict_dino_aug2= rede_2(dino_aug_2, training=True)[1]


      rede1_predict_dino_aug1=temp_softmax(rede1_predict_dino_aug1)
      rede1_predict_dino_aug2=temp_softmax(rede1_predict_dino_aug2)

      rede2_predict_dino_aug1=temp_softmax(rede2_predict_dino_aug1)
      rede2_predict_dino_aug2=temp_softmax(rede2_predict_dino_aug2)



      # Aqui rede1_predict_dino_aug1 esta como label e rede2_predict_dino_aug2 como previsto
      Loss_1_dino= Final_loss(rede2_predict_dino_aug2,rede1_predict_dino_aug1)
      # Aqui rede2_predict_dino_aug1 esta como label e rede1_predict_dino_aug2 como previsto
      Loss_2_dino= Final_loss(rede1_predict_dino_aug2,rede2_predict_dino_aug1)
      
      Loss_dino_final=(Loss_1_dino+Loss_2_dino)/2
      #print(Loss_final_rede1)
      #disc_loss = discriminator_loss_weights(d_net_real,y_prob_mb)
      #disc_loss = discriminator_loss_weights_kl(d_net_real,y_prob_mb)
      #gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
      Loss_final_rede1=Loss_final_rede1+0.2*Loss_dino_final
      Loss_final_rede2=Loss_final_rede2+0.2*Loss_dino_final
    gradients_of_rede1 = disc_tape.gradient(Loss_final_rede1, rede_1.trainable_variables)

        #generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    rede_1_optimizer.apply_gradients(zip(gradients_of_rede1, rede_1.trainable_variables))

    gradients_of_rede2 = gen_tape.gradient(Loss_final_rede2, rede_2.trainable_variables)

        #generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    rede_2_optimizer.apply_gradients(zip(gradients_of_rede2, rede_2.trainable_variables))

    return acertos_porcentagem,total_de_img_selecionads,acertos_porcentagem_2,total_de_img_selecionads_2
    #gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        #generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    #discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

####################################################################################
#PARAMETROS TREINO
##################################################################
  # Change here parameters
  lr = 0.001
  #batch_size = 64
  ratio=args.noise_rate
  rede_1_optimizer = tf.keras.optimizers.Adam(lr)
  rede_2_optimizer = tf.keras.optimizers.Adam(lr)
  #patience epochs 200
  Tk=10
  #rede_1 = create_custom_model(net1,rede_1_optimizer)
  #rede_2 = create_custom_model(net2,rede_2_optimizer)
  rede_1=net1
  rede_2=net2
  epoch_ammount=200
  max_f1_treino=0
  max_f1_valid=0
  for epoch in range(epoch_ammount):
    #nmb = trainx.shape[0] // mb_size
       
    passos_per_epoch=int(len(datagenerator))
    acertos_porcentagem=tf.Variable(0,dtype=tf.float64)
    total_de_img_selecionads=tf.Variable(0,dtype=tf.float64)

    acertos_porcentagem_2=tf.Variable(0,dtype=tf.float64)
    total_de_img_selecionads_2=tf.Variable(0,dtype=tf.float64)
        
    print('TREINO:::')
    sys.stdout.write('\r TRAIN TIME EPOCA INICIALIZANDO : ' + str(epoch) + ' of ' + str(epoch_ammount))
    sys.stdout.flush()
    print(' \n \n')
        
    mb=0
    V=np.array([(ratio*(epoch+1))/Tk,ratio])
    R_T=1-np.amin(V)
    print('R_T :',R_T)
    
    for mb in range(passos_per_epoch):
      sys.stdout.write('\rUpdated record: ' + str(mb) + ' of ' + str(passos_per_epoch))
      sys.stdout.flush()
                   
      

      images_labels_pro=datagenerator.__next__()
      correct_labels_images=datagenerator_corret_label.__next__() 

      X_mb=images_labels_pro[0]
     
      Y_mb=images_labels_pro[1]
    
      #y_prob_mb =  proportion_compute(Y_mb)

      Correct_labels=correct_labels_images[1]
    
     
      acertos_porcentagem,total_de_img_selecionads,acertos_porcentagem_2,total_de_img_selecionads_2=train_step(X_mb,Y_mb,R_T,ratio,Correct_labels,acertos_porcentagem,total_de_img_selecionads,epoch_ammount,acertos_porcentagem_2,total_de_img_selecionads_2)

      if mb== passos_per_epoch-1:
        tf.print('--------------------------------------')
        tf.print('porcentagem de acertos!',acertos_porcentagem/total_de_img_selecionads)
        tf.print(' Acertos',acertos_porcentagem)
        tf.print(' Total',total_de_img_selecionads)
        tf.print('--------------------------------------')
        tf.print('--------------------------------------')
        tf.print('porcentagem de acertos!',acertos_porcentagem_2/total_de_img_selecionads_2)
        tf.print(' Acertos',acertos_porcentagem_2)
        tf.print(' Total',total_de_img_selecionads_2)
        tf.print('--------------------------------------')
        
  
    print('Avaliando a rede \n')
    max_f1_treino=metrics_f1_recal_pre_mat_calculate(rede_1,rede_2,epoch,datagenerator,max_f1_treino,args,acertos_porcentagem,total_de_img_selecionads,acertos_porcentagem_2,total_de_img_selecionads_2,conjunto='treino')
    max_f1_valid=metrics_f1_recal_pre_mat_calculate(rede_1,rede_2,epoch,datagenerator_teste,max_f1_valid,args,acertos_porcentagem,total_de_img_selecionads,acertos_porcentagem_2,total_de_img_selecionads_2, conjunto='valid')
    
    

if __name__ == '__main__':

  """ 
  Codigo piloto Implementação do Retrieve Discard Sampels (RDS) Contrastive
  
  
  """
  parser = argparse.ArgumentParser(description='Codigo RDS')
  parser.add_argument('--noise_type', type=str, dest='noise_type', help='symmetric ou pairflip',default='symmetric')
  parser.add_argument('--dataset', type=str, dest='dataset', help=' Dataset cifar10 cifar100 ou mnist',default='cifar10')
  parser.add_argument('--batch_size', type=int, dest='batch_size', help='batch_size int',default=128)
  parser.add_argument('--noise_rate', type=float, dest='noise_rate',default=0.2, help='Noise rate 0.01-0.99')
  parser.add_argument('--results', type=str, dest='results_path',default='models', help='Caminho de onde salvar o modelo e logs de saida')
  parser.add_argument('--aug_time', type=int, dest='aug_time',default=1, help='Quantidade de aug')
  parser.add_argument('--threshold', type=float, dest='threshold',default=0.8, help='threshold')
  
  args = parser.parse_args()

  main(args)

 
