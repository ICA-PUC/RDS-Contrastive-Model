

import os
import os.path
import numpy as np

#if sys.version_info[0] == 2:
#    import cPickle as pickle
#else:
import pickle

#import torch.utils.data as data
#from utils_add_noise import download_url, check_integrity, noisify
from numpy.testing import assert_array_almost_equal
from tensorflow.keras.utils import to_categorical


class LOAD_CIFAR10():
    ''' 
    
    '''

    def __init__(self, diretorio='cifar-10-batches-py', 
                 noise_type="symmetric", noise_rate=0.45, random_state=0):
        self.nb_classes=10
        self.diretorio=diretorio
        self.noise_type=noise_type
        self.noise_rate=noise_rate
        "symmetric or pairflip"
        ######### Carregando o dataset original do CIFAR-10 ############
        print('Carregando CIFAR-10 da pasta: ',self.diretorio)

        self.listOfFiles=[]
        for (dirpath, dirnames, filenames) in os.walk(self.diretorio):
            [self.add_list_file(dirpath,file) if file.split('_')[0]=='data'else print('ok') for file in filenames]
        print('Quantidade de arquivos',len(self.listOfFiles))
        
        
        
        
        self.train_data = []
        self.train_labels = []
        for file in self.listOfFiles:
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='latin1')
            self.train_data.append(dict['data'])
            if 'labels' in dict:
                self.train_labels += dict['labels']
            else:
                self.train_labels += dict['fine_labels']
            fo.close()
        self.train_data = np.concatenate(self.train_data)
        self.train_data = self.train_data.reshape((50000, 32, 32,3))
        #self.train_data = self.train_data.transpose((0, 2, 3, 1))# convert to hwc ?
        
        self.train_labels=np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])

        self.test_data=[]
        self.test_labels=[]

        with open('cifar-10-batches-py/test_batch', 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        self.test_data.append(dict['data'])
        if 'labels' in dict:
            self.test_labels += dict['labels']
        else:
            self.test_labels += dict['fine_labels']
            fo.close()
        self.test_data = np.concatenate(self.test_data)
        self.test_data = self.test_data.reshape((10000, 32, 32,3))
        #self.test_data = self.test_data.transpose((0, 2, 3, 1))# convert to hwc ?
        self.test_labels=to_categorical(self.test_labels,num_classes=10)
        
        self.train_noisy_labels,self.actual_noise_rate = self.noisify(train_labels=self.train_labels, noise_type=self.noise_type, noise_rate=self.noise_rate, random_state=0, nb_classes=self.nb_classes)
        
        self.train_noisy_labels=to_categorical(self.train_noisy_labels,num_classes=10)
        self.train_labels=to_categorical(self.train_labels,num_classes=10)
    def add_list_file(self,dirpath,file):
    #print(file)
    # print(file.split('.')[-1])
        self.listOfFiles+=[os.path.join(dirpath,file)]




    def noisify(self,dataset='mnist', nb_classes=10, train_labels=None, noise_type=None, noise_rate=0, random_state=0):
        if noise_type == 'pairflip':
            train_noisy_labels, actual_noise_rate = self.noisify_pairflip(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
        if noise_type == 'symmetric':
            train_noisy_labels, actual_noise_rate = self.noisify_multiclass_symmetric(train_labels, noise_rate, random_state=0, nb_classes=nb_classes)
        return train_noisy_labels, actual_noise_rate

    
    def noisify_multiclass_symmetric(self,y_train, noise, random_state=None, nb_classes=10):
        """mistakes:
            flip in the symmetric way
        """
        P = np.ones((nb_classes, nb_classes))
        n = noise
        P = (n / (nb_classes - 1)) * P

        if n > 0.0:
            # 0 -> 1
            P[0, 0] = 1. - n
            for i in range(1, nb_classes-1):
                P[i, i] = 1. - n
            P[nb_classes-1, nb_classes-1] = 1. - n

            y_train_noisy = self.multiclass_noisify(y_train, P=P,
                                            random_state=random_state)
            actual_noise = (y_train_noisy != y_train).mean()
            assert actual_noise > 0.0
            print('Actual noise %.2f' % actual_noise)
            y_train = y_train_noisy
        print (P)
        return y_train, actual_noise
    
    def multiclass_noisify(self,y, P, random_state=0):
        """ Flip classes according to transition probability matrix T.
        It expects a number between 0 and the number of classes - 1.
        """
        print (np.max(y), P.shape[0])
        assert P.shape[0] == P.shape[1]
        assert np.max(y) < P.shape[0]

        # row stochastic matrix
        assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
        assert (P >= 0.0).all()

        m = y.shape[0]
        print (m)
        new_y = y.copy()
        flipper = np.random.RandomState(random_state)

        for idx in np.arange(m):
            i = y[idx]
            # draw a vector with only an 1
            flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
            new_y[idx] = np.where(flipped == 1)[0]

        return new_y



    def noisify_pairflip(self,y_train, noise, random_state=None, nb_classes=10):
        """mistakes:
            flip in the pair
        """
        P = np.eye(nb_classes)
        n = noise

        if n > 0.0:
            # 0 -> 1
            P[0, 0], P[0, 1] = 1. - n, n
            for i in range(1, nb_classes-1):
                P[i, i], P[i, i + 1] = 1. - n, n
            P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

            y_train_noisy = self.multiclass_noisify(y_train, P=P,
                                            random_state=random_state)
            actual_noise = (y_train_noisy != y_train).mean()
            assert actual_noise > 0.0
            print('Actual noise %.2f' % actual_noise)
            y_train = y_train_noisy
        print (P)

        return y_train, actual_noise

