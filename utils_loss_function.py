from numpy import NaN
import tensorflow as tf
import numpy as np

#import load_svhn
from tensorflow.keras.models import Model, load_model

from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout,BatchNormalization,LeakyReLU,UpSampling2D,GlobalAveragePooling2D

from functools import partial
from sklearn.neighbors import KNeighborsClassifier
#tf.config.run_functions_eagerly(True)
def proportion_compute_tensorflow_mode(label_input):
  
    #x = tf.constant([[1,0,0,0,0],[1,0,0,0,0],[0,0,0,1,0],[1,0,0,0,1]])
    label_input=tf.cast(label_input, tf.float32)
    prob_result=tf.reduce_mean(label_input, 0)
  
    
    return prob_result
def proportion_compute(label_input):

    batch_size = len(label_input)
    prob_result = label_input.sum(axis = 0)/batch_size
    return  prob_result
def proportion_compute_lambda(label_input):

    prob_seq1=np.array([0.75, 0 , 0 , 0.25])
    prob_seq2=np.array([0,0.35, 0.45, 0.2])

    prob_result_1=label_input[0]*prob_seq1
    prob_result_2=label_input[1]*prob_seq2

    prob_result_all=prob_result_1+prob_result_2
    return prob_result_all


def discriminator_loss(d_net_real,y_prob):
    ''' 
    Esta funcao calcula a Loss da proporcao pela  cross entropy da proporcao
    '''

    D_loss_prop =  tf.reduce_mean(-tf.reduce_sum(y_prob*(tf.math.log(tf.reduce_mean(d_net_real, [0]) + 1e-7))))
  

    return D_loss_prop

def Small_Loss_select(images,vetor_loss_rede,ratio,batch_size,Y_mb):

    ''' 
    Esta funcao seleciona, com base no fator
     R(t)"ratio" ,as imagems que tivera as menores perdas no passo atual da rede

    '''

    num_instance = int(batch_size * ratio)
    #tf.print(vetor_loss_rede)

    new_index_images_labels=tf.argsort(vetor_loss_rede)[:num_instance]

    labels_small_loss_select=tf.gather(Y_mb , new_index_images_labels)
        
    images_small_loss_select=tf.gather(images , new_index_images_labels)
    
    #features_small_loss_select=tf.gather(rede_features,new_index_images_labels)

    return images_small_loss_select,labels_small_loss_select


def Small_Loss_select_rt1_rt2(images,vetor_loss_rede,batch_size,Y_mb,RT1,RT2):

    ''' 
    Esta funcao seleciona, com base no fator
     R(t)"ratio" RT1 e RT2 ,as imagems que tivera as menores perdas no passo atual da rede

    '''

    num_instance_high = int(batch_size * RT2)
    num_instance_low= int(batch_size * RT1)
    #tf.print(vetor_loss_rede)

    new_index_images_labels=tf.argsort(vetor_loss_rede)[num_instance_low:num_instance_high]

    labels_small_loss_select=tf.gather(Y_mb , new_index_images_labels)
        
    images_small_loss_select=tf.gather(images , new_index_images_labels)
    

    return images_small_loss_select,labels_small_loss_select

def sum_prop_discard(all_vector_prop,vector_prop_to_sum):
 
    if all_vector_prop==None:
        all_vector_prop=vector_prop_to_sum
        all_vector_prop=tf.expand_dims( all_vector_prop, 0, name=None)
  
    else:
      
        vector_prop_to_sum=tf.expand_dims( vector_prop_to_sum, 0, name=None)
     
        all_vector_prop=tf.concat([all_vector_prop, vector_prop_to_sum], 0)
      
   
    return all_vector_prop

def calculate_std_mean(all_vector_prop):
    all_vector_prop=tf.cast(all_vector_prop, tf.float32)
    tf.print('sahpe final',tf.shape(all_vector_prop))
    tf.print('erro e desvio padrao')
    print('media')
    tf.print(tf.math.reduce_mean(all_vector_prop, 0))
    print('std')
    tf.print(tf.math.reduce_std(all_vector_prop, 0))

def KNN_model(labels_selecionadas_rede1,labels_selecionadas_rede2,features_selecionadas_rede1,features_selecionadas_rede2,n_neighbors=5):
    model_knn = KNeighborsClassifier(n_neighbors=5)
    ### conc labels
    #labels_selecionadas_rede1
    #tf.print('labels shape knn model',labels_selecionadas_rede1.get_shape().as_list())
    #tf.print('features shape knn model',features_selecionadas_rede1.get_shape().as_list())

    all_labels=tf.concat([labels_selecionadas_rede1,labels_selecionadas_rede2], 0)
    all_features=tf.concat([features_selecionadas_rede1,features_selecionadas_rede2], 0)

    #tf.print('all labels CONC shape knn model',all_labels.get_shape().as_list())

    #tf.print('all_features CONC shape knn model',all_features.get_shape().as_list())
    model_knn.fit(all_features.numpy(),all_labels.numpy())

    return model_knn

def Small_Loss_select_discard(images,vetor_loss_rede,ratio,batch_size,Y_mb,Correct_labels):

    ''' 
    Esta funcao seleciona, com base no fator
     R(t)"ratio" ,as imagems que tivera as menores perdas no passo atual da rede

    '''

    num_instance = int(batch_size * ratio)
    #tf.print(vetor_loss_rede)

    new_index_images_labels_discard=tf.argsort(vetor_loss_rede)[num_instance:]
    labels_small_loss_select_discard=tf.gather(Y_mb , new_index_images_labels_discard)
        
    images_small_loss_select_discard=tf.gather(images , new_index_images_labels_discard)

    

    Correct_labels=tf.gather(Correct_labels , new_index_images_labels_discard)
    return images_small_loss_select_discard,labels_small_loss_select_discard,Correct_labels



def Final_loss(rede_predict_small_loss,labels_selecionadas_rede):
    '''
    Loss final do modelo. Junta a Loss das imgs com menores perdas com a loss da proporcao
    '''
    cce_final = tf.keras.losses.CategoricalCrossentropy()
    lamb=0.45
    final_loss=cce_final(labels_selecionadas_rede,rede_predict_small_loss)#+rede_proporcao_loss
    #final_loss=rede_proporcao_loss
    return final_loss


def Final_loss_with_pseudo_labels(rede_predict_small_loss,labels_selecionadas_rede,loss_pesudo,R_T,ratio,epoch_ammount):
    '''
    Loss final do modelo. Junta a Loss das imgs com menores perdas com a loss da proporcao
    '''
    cce_final = tf.keras.losses.CategoricalCrossentropy()
    lamb=0.5
    #lamb=0.25
    final_loss=cce_final(labels_selecionadas_rede,rede_predict_small_loss)#+rede_proporcao_loss
    #if epoch_ammount>=20:
        #tf.print('i a here in final loss....')
    final_loss=final_loss+lamb*loss_pesudo
    
    #final_loss=rede_proporcao_loss
    return final_loss


def Final_loss_2(rede_predict_small_loss,rede_proporcao_loss,labels_selecionadas_rede):
    #cce_final = tf.keras.losses.CategoricalCrossentropy()
    
    lamb=0.4
    #final_loss=tf.math.reduce_mean(-labels_selecionadas_rede * tf.log(rede_predict_small_loss + 1e-6))
    #final_loss=cce_final(labels_selecionadas_rede,rede_predict_small_loss)#+lamb*rede_proporcao_loss



    final_loss=tf.math.reduce_mean(-labels_selecionadas_rede*tf.math.log(tf.reduce_mean(rede_predict_small_loss, [0]) + 1e-7))
    #rede_predict_small_loss=rede_predict_small_loss+1e-7
    #final_loss=tf.math.reduce_mean(-labels_selecionadas_rede*tf.math.log(rede_predict_small_loss))
    
    return final_loss

def generator_loss(d_net_real_feat,d_net_fake_feat):

    G_loss = tf.reduce_mean(tf.square(tf.reduce_mean(d_net_real_feat, axis=0)-tf.reduce_mean(d_net_fake_feat, axis=0)))

    return G_loss

def Loss_cross_entropy(y_true,y_pred):
    cce = tf.keras.losses.CategoricalCrossentropy(
    reduction=tf.keras.losses.Reduction.NONE)


    return cce(y_true, y_pred)

def Loss_cross_entropy_exp(y_true,y_pred):
    cce = tf.keras.losses.CategoricalCrossentropy(
    reduction=tf.keras.losses.Reduction.NONE)
    #print(tf.math.exp(tf.math.pow(
    #y_pred, 5, name=None)))
    exp_pred=tf.math.exp(tf.math.pow(
    y_pred, 1.5, name=None))
    
    fator_exp=tf.reduce_mean(exp_pred, 1)
    #print(b)
    #cce=tf.math.exp(y_pred**3.8)*cce(y_true, y_pred)
    return tf.reduce_mean(fator_exp*cce(y_true,y_pred))




def Focal_rds_2(y_true, y_pred,alpha,gamma):
    cross_entropy=Final_loss(y_pred,y_true)
    #tf.print('ce',cross_entropy)
    probs=y_pred
    #tf.print('tr',y_true)
    #tf.print('y_pred',y_pred)
    alpha = tf.where(tf.equal(y_true, 1.0), alpha, (1.0 -alpha))
    #tf.print('alpha',alpha)
    
    pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
    #tf.print('pt',pt)
    loss = alpha * tf.pow(1.0 - pt,gamma) * cross_entropy
    #tf.print('ls',loss)
    #tf.print('final',tf.reduce_sum(loss, axis=-1))
    
    final=tf.math.reduce_mean(loss, axis=-1)
    #tf.print(final)

    shape=tf.math.reduce_mean(loss, axis=-1).get_shape().as_list()
    #tf.print('shape',shape)


    is_empty = tf.equal(tf.size(final), 0)

    #tf.print('is_empty',is_empty)
    
    if is_empty==False:
        final_rds=tf.math.reduce_mean(final)
    else:
        final_rds=0.
    #tf.print(tf.reduce_sum(loss, axis=-1).get_shape().as_list())
    #tf.print('final',final_rds)
    if tf.math.is_inf(final_rds):
        final_rds=0.
        tf.print('infinito infelizmente')
    return final_rds


def Focal_rds(y_true,y_pred,alpha,gamma): 
#focal_loss_fixed(y_true, y_pred):
    """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
    epsilon = 1.e-9
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)
    tf.print('true',y_true)
    tf.print('y_pred',y_pred)

    model_out = tf.add(y_pred, epsilon)
    ce = tf.multiply(y_true, -tf.math.log(model_out))
    tf.print('ce',ce)
    weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
    tf.print('weight',weight)
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    tf.print('fl',fl)
    fl=tf.where(tf.math.is_nan(fl), tf.zeros_like(fl), fl)
    tf.print('fl no nan',fl)
    reduced_fl = tf.reduce_max(fl, axis=1)
    tf.print('reduced_fl',reduced_fl)
    tf.print('media',tf.reduce_mean(reduced_fl))
    return tf.reduce_mean(reduced_fl)

#def Focal_rds(y_true,y_pred,alpha,gamma):
    


 #   epsilon = 1.e-9
    #y_true = tf.convert_to_tensor(y_true, tf.float32)
    #y_pred = tf.convert_to_tensor(y_pred, tf.float32)

 #   model_out =tf.add(y_pred, epsilon)
    
    #tf.add(y_pred, epsilon)
 ##   ce = tf.multiply(y_true, -tf.math.log(model_out))
 ##   tf.print('ce',ce)
  #  tf.print('sub',tf.subtract(1., model_out))
  #  tf.print('power', tf.pow(tf.subtract(1., model_out), gamma))
  #  weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
  #  tf.print(weight)
  #  fl = tf.multiply(alpha, tf.multiply(weight, ce))
  #  tf.print('fl',fl)
  #  reduced_fl = tf.reduce_max(fl, axis=1)
  #  tf.print('reduced max',reduced_fl)
  #  focal=tf.reduce_mean(reduced_fl)
    
    

   ## tf.print('focal',focal)
    #return focal

def Loss_cross_entropy_2(y_true,y_pred):
    y_pred_=y_pred+ 1e-7
    cce_2=tf.math.reduce_mean(-y_true*tf.math.log(y_pred_),1)
    print(cce_2.shape)
    return cce_2
    #y_prob*(tf.math.log(tf.reduce_mean(d_net_real, [0]) + 1e-7))))

def Loss_cross_entropy_3(y_true,y_pred):
    
    cce=tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    return cce

def Final_loss_3(rede_predict_small_loss,rede_proporcao_loss,labels_selecionadas_rede):
    '''
    Loss final do modelo. Junta a Loss das imgs com menores perdas com a loss da proporcao
    '''
    #cce_final = tf.keras.losses.CategoricalCrossentropy()
    final_loss=tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_selecionadas_rede, logits=rede_predict_small_loss))
    lamb=0.1
    #final_loss=cce_final(labels_selecionadas_rede,rede_predict_small_loss)#+lamb*rede_proporcao_loss
    return final_loss

    #tf.nn.softmax_cross_entropy_with_logits

    #tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

def discriminator_loss_weights(d_net_real,y_prob):
    
    class_weights=tf.cast(tf.constant([5, 1, 1.4, 0.7, 2]), tf.float64)
   

    D_loss_prop =  tf.reduce_mean(-tf.reduce_sum(class_weights*y_prob*(tf.math.log(tf.reduce_mean(d_net_real, [0]) + 1e-7))))
   

    return D_loss_prop

def discriminator_loss_weights_kl(d_net_real,y_prob):
    replace_0_to_1 = tf.cast(tf.constant([1, 1, 1, 1, 1]), tf.float64)
  

    condition = tf.equal(y_prob, 0)
    #tf.print(condition)
    y_prob_for_division=tf.where(condition, replace_0_to_1,y_prob)
    #tf.print(y_prob_for_division)
    D_loss_prop =  tf.reduce_mean(-tf.reduce_sum(y_prob*(tf.math.log((tf.reduce_mean(d_net_real, [0])/y_prob_for_division) + 1e-7))))
    
    return D_loss_prop


def augmentations(batch_images):
    
  
    aug_1=tf.image.adjust_brightness(batch_images,0.5)
    aug_2= tf.image.adjust_saturation(batch_images,20)
    aug_3=tf.image.flip_up_down(batch_images)
  
    aug_4= tf.image.flip_left_right(batch_images)
    
    aug_5=tf.image.adjust_brightness(tf.image.flip_left_right(batch_images),0.5)
    aug_6=tf.image.adjust_saturation(tf.image.flip_up_down(batch_images),20)
    aug_7=tf.image.random_saturation(tf.image.adjust_brightness(batch_images,0.9),2,15)
    aug_8=tf.image.rot90(tf.image.adjust_brightness(batch_images,0.2), k=1, name=None)

    return aug_1,aug_2,aug_3,aug_4,aug_5,aug_6,aug_7,aug_8

def augmentations_dino(batch_images):
    rand_1=np.random.randint(8, size=1)[0]
    rand_2=np.random.randint(8, size=1)[0]
    
    

    aug_dic=dict({0:partial(tf.image.adjust_brightness,delta=0.5),1:partial(tf.image.adjust_saturation,saturation_factor=20),
    2:partial(tf.image.flip_up_down),3:partial(tf.image.flip_left_right),4:partial(tf.image.rot90,k=1) ,5:partial(tf.image.adjust_saturation,saturation_factor=10),
    6:partial(tf.image.adjust_brightness,delta=0.1),7: partial(tf.image.adjust_saturation,saturation_factor=50)})
   
    dino_aug_1=aug_dic.get(rand_1)(batch_images)
    dino_aug_2=aug_dic.get(rand_2)(batch_images)

    return dino_aug_1,dino_aug_2

def temp_softmax(logistic,temp=0.25):
    logistic_temp=logistic/temp

    soft_temp=tf.nn.softmax(logistic_temp)
    return soft_temp


def predict_aug_images(rede,rede_ajuste,aug_1,aug_2,aug_3,aug_4,aug_5,aug_6,aug_7,aug_8,images_discard,Correct_labels,acertos_porcentagem,total_de_img_selecionads,labels_discard_rede,aug_time,threshold):
    
    predict_aug1= tf.cast( rede(aug_1, training=True)[0],tf.float32)
    predict_aug2= tf.cast( rede(aug_2, training=True)[0],tf.float32)
    predict_aug3= tf.cast( rede(aug_3, training=True)[0],tf.float32)
    predict_aug4=  tf.cast( rede(aug_4, training=True)[0],tf.float32)
    predict_aug5=  tf.cast( rede(aug_5, training=True)[0],tf.float32)
    predict_aug6=  tf.cast( rede(aug_6, training=True)[0],tf.float32)
    predict_aug7=  tf.cast( rede(aug_7, training=True)[0],tf.float32)
    predict_aug8=  tf.cast( rede(aug_8, training=True)[0],tf.float32)
    predict_aug9=  tf.cast( rede(images_discard, training=True)[0],tf.float32)

    predict_aug1_aj= tf.cast( rede_ajuste(aug_1, training=True)[0],tf.float32)
    predict_aug2_aj= tf.cast( rede_ajuste(aug_2, training=True)[0],tf.float32)
    predict_aug3_aj= tf.cast( rede_ajuste(aug_3, training=True)[0],tf.float32)
    predict_aug4_aj=  tf.cast( rede_ajuste(aug_4, training=True)[0],tf.float32)
    predict_aug5_aj=  tf.cast( rede_ajuste(aug_5, training=True)[0],tf.float32)
    predict_aug6_aj=  tf.cast( rede_ajuste(aug_6, training=True)[0],tf.float32)
    predict_aug7_aj=  tf.cast( rede_ajuste(aug_7, training=True)[0],tf.float32)
    predict_aug8_aj=  tf.cast( rede_ajuste(aug_8, training=True)[0],tf.float32)
    predict_aug9_aj=  tf.cast( rede_ajuste(images_discard, training=True)[0],tf.float32)
    

    #all_predics=(predict_aug1+predict_aug2+predict_aug3+predict_aug4+predict_aug5+predict_aug6+predict_aug7+predict_aug8+predict_aug9)/9
    #all_predics=(predict_aug1+predict_aug2+predict_aug3+predict_aug9)/4
    if aug_time==1:
        all_predics=(predict_aug9*4)/4
    elif aug_time==2:
        all_predics=(predict_aug9+predict_aug1)/2
    elif aug_time==3:
        all_predics=(predict_aug9+predict_aug1+predict_aug2)/3
    elif aug_time==4:
        all_predics=(predict_aug9+predict_aug1+predict_aug2+predict_aug3)/4
    elif aug_time==8:
        all_predics=(predict_aug9+predict_aug1+predict_aug2+predict_aug3+predict_aug4+predict_aug5+predict_aug6+predict_aug7)/8

    # rint com trashold
    all_predics=tf.cast( tf.where(all_predics > threshold, 1, 0),tf.float32)
    #all_predics=tf.math.rint(all_predics, name=None)
    #rint com trashold

    #all_predics_aj=(predict_aug1_aj+predict_aug2_aj+predict_aug3_aj+predict_aug4_aj+predict_aug5_aj+predict_aug6_aj+predict_aug7_aj+predict_aug8_aj+predict_aug9_aj)/9

    #all_predics_aj=(predict_aug1_aj+predict_aug2_aj+predict_aug3_aj+predict_aug9_aj)/4
    if aug_time==1:
        all_predics_aj=(predict_aug9_aj*4)/4
    elif aug_time==2:
        all_predics_aj=(predict_aug9_aj+predict_aug1_aj)/2
    elif aug_time==3:
        all_predics_aj=(predict_aug9_aj+predict_aug1_aj+predict_aug2_aj)/3
    elif aug_time==4:
        all_predics_aj=(predict_aug9_aj+predict_aug1_aj+predict_aug2_aj+predict_aug3_aj)/4
    elif aug_time==8:
        all_predics_aj=(predict_aug9_aj+predict_aug1_aj+predict_aug2_aj+predict_aug3_aj+predict_aug4_aj+predict_aug5_aj+predict_aug6_aj+predict_aug7_aj)/8
    

    # rint com trashold
    all_predics_aj=tf.cast( tf.where(all_predics_aj >threshold
, 1, 0),tf.float32)
    ##all_predics_aj=tf.math.rint(all_predics_aj, name=None)
    #rint com trashold
    

    equal_values_rede1_rede2 = tf.where(tf.math.equal(all_predics,all_predics_aj),all_predics,2)

    row_wise_sum_rede1_rede2 = tf.reduce_sum(tf.abs(equal_values_rede1_rede2),1)

    #final_equal_index=tf.where(tf.reduce_any(tf.math.not_equal(all_predics, all_predics_aj),axis=1))
    
    final_equal_index = tf.where(row_wise_sum_rede1_rede2<=1)

    labels_discard_rede=tf.gather_nd(labels_discard_rede,final_equal_index)



    

    Correct_labels=tf.gather_nd(Correct_labels,final_equal_index)

    

    images_select_pseudo_labels=tf.gather_nd(images_discard,final_equal_index)
    
    pseudo_label=tf.gather_nd(all_predics,final_equal_index)
    
    

    row_wise_sum = tf.reduce_sum(tf.abs(pseudo_label),1)
    select_one_sum_indexes = tf.where(tf.equal(row_wise_sum,1))

    pseudo_label=tf.gather_nd(pseudo_label,select_one_sum_indexes)

    images_select_pseudo_labels=tf.gather_nd(images_select_pseudo_labels,select_one_sum_indexes)
    
    
    Correct_labels=tf.gather_nd(Correct_labels,select_one_sum_indexes)

    labels_discard_rede=tf.gather_nd(labels_discard_rede,select_one_sum_indexes)

    # selecionar os labels que sao diferentes dos anotados que estão possivelmente errados"
    final_not_equal_index=tf.where(tf.reduce_any(tf.math.not_equal(pseudo_label, labels_discard_rede),axis=1))

    pseudo_label_final=tf.gather_nd(pseudo_label,final_not_equal_index)

    images_select_pseudo_labels_final=tf.gather_nd(images_select_pseudo_labels,final_not_equal_index)
    
    
    Correct_labels_final=tf.gather_nd(Correct_labels,final_not_equal_index)

    labels_discard_rede_final=tf.gather_nd(labels_discard_rede,final_not_equal_index)

    #Correct_labels_final=tf.cast(Correct_labels_final,tf.float32)

    is_empty = tf.equal(tf.size(pseudo_label), 0)


    
    if is_empty==False:
        #tf.print()
        #tf.print(Correct_labels_final,summarize=12)
        #tf.print(pseudo_label_final,summarize=12)
        c=tf.math.equal(Correct_labels_final, pseudo_label_final)
        count_true_false=tf.reduce_all(c, axis=1)
        acertos=tf.math.count_nonzero(count_true_false)
        
        acertos_porcentagem=acertos_porcentagem+tf.cast(acertos,tf.float64)
        total_de_img_selecionads=total_de_img_selecionads+tf.cast(tf.shape(pseudo_label_final)[0],tf.float64)
        #tf.print('Acertos ',acertos_porcentagem)

        #tf.print('total geral',total_de_img_selecionads)

        #tf.print('pseudo_label',pseudo_label_final,summarize=12)
        #tf.print('Correct_labels',Correct_labels_final,summarize=12)
        #tf.print('labels_discard_rede',labels_discard_rede_final,summarize=12)
    
    #else:
    #    tf.print('pseudo_label df',pseudo_label)
    ##    tf.print('Correct_labels df',Correct_labels)
    #    tf.print('labels_discard_rede df',labels_discard_rede)
    #    tf.print('pseudo_label',pseudo_label_final)
    #    tf.print('Correct_labels',Correct_labels_final)
    #    tf.print('labels_discard_rede',labels_discard_rede_final)
        
        


        
        

   
    return pseudo_label_final,images_select_pseudo_labels_final,acertos_porcentagem,total_de_img_selecionads

def predict_KNN(model_knn,images_discard_rede,Correct_labels,acertos_porcentagem,total_de_img_selecionads,labels_discard_rede,features_discard_rede):
    # calcula os pseudo labels
   
    
    #tf.print('Correct_labels_final',Correct_labels.get_shape().as_list())
    pseudo_label=model_knn.predict(features_discard_rede)
    #tf.print('shape pseudo pred model knn',pseudo_label.shape)

    final_not_equal_index=tf.where(tf.reduce_any(tf.math.not_equal(pseudo_label, labels_discard_rede),axis=1))

    pseudo_label_final=tf.gather_nd(pseudo_label,final_not_equal_index)

    images_select_pseudo_labels_final=tf.gather_nd(images_discard_rede,final_not_equal_index)
    
    
    Correct_labels_final=tf.gather_nd(Correct_labels,final_not_equal_index)

    labels_discard_rede_final=tf.gather_nd(labels_discard_rede,final_not_equal_index)

    #Correct_labels_final=tf.cast(Correct_labels_final,tf.float32)

    is_empty = tf.equal(tf.size(pseudo_label), 0)

    if is_empty==False:
        #tf.print()
        #tf.print(Correct_labels_final,summarize=12)
        #tf.print(pseudo_label_final,summarize=12)
        c=tf.math.equal(Correct_labels_final, pseudo_label_final)
        count_true_false=tf.reduce_all(c, axis=1)
        acertos=tf.math.count_nonzero(count_true_false)
        
        acertos_porcentagem=acertos_porcentagem+tf.cast(acertos,tf.float64)
        total_de_img_selecionads=total_de_img_selecionads+tf.cast(tf.shape(pseudo_label_final)[0],tf.float64)

    return pseudo_label_final,images_select_pseudo_labels_final,acertos_porcentagem,total_de_img_selecionads
