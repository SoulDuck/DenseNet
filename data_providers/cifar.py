import tempfile
import os
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np


from base_provider import ImageDataSet , DataProvider
from downloader import download_data_url



def augment_image(image , pad):

    init_shape = image.shape
    new_shape = [init_shape[0] +pad*2 , init_shape[1]+pad*2 , init_shape[2]]
    zeros_padded=np.zeros(new_shape)
    zeros_padded[pad:init_shape[0]+pad , pad: init_shape[1] +pad , : ] = image

    init_x = np.random.randint(0, pad*2)
    init_y = np.random.randint(0 , pad*2)

    cropped = zeros_padded[init_x: init_x + init_shape[0]  , init_y : init_y + init_shape[1] , :]
    flip= random.getrandbits(1)
    if flip:
        cropped = cropped[: , ::-1 , :]
    return cropped



def augment_all_images(initial_images , pad):
    new_images = np.zeros(initial_images.shape)
    for i in range(initial_images.shape[0]):
        new_images[i] = augment_image(initial_images[i] , pad=4)
        return new_images



class CifarDataProvider(DataProvider):
    def __init__ (self , save_path=None , validation_set = None , validation_split=None , \
                  shuffle=None , normalization =None  ,one_hot =True , augmentation=False ,**kwargs):
        print 'CifarDataProvider(DataProvider)'

        print '\tsave_path:' ,save_path
        print '\tvalidation_set:',validation_set
        print '\tvalidation_split:',validation_split
        print '\tshuffle:',shuffle
        print '\tnormalization:',normalization
        print '\tone_hot:',one_hot

        self.augmentation=augmentation

        self._save_path = save_path
        self.one_hot = one_hot
        download_data_url(self.data_url , self._save_path)
        print self
        train_fnames , test_fnames  = self.get_filenames(self._save_path)
        images , labels = self.read_cifar(train_fnames)
        if validation_set is not None and validation_split is not None:

            split_idx = int(images.shape[0] * (1 -validation_split))
            print 'split idx : ', split_idx
            self.train = CifarDataSet(
                images = images[:split_idx],
                labels =labels[:split_idx],
                n_classes=self.n_classes,
                shuffle=shuffle,
                normalization=normalization,
                augmentation = self.augmentation)
            self.validation = CifarDataSet(
                images=images[split_idx:],
                labels=labels[split_idx:],
                n_classes=self.n_classes,
                shuffle=shuffle,
                normalization=normalization,
                augmentation=self.augmentation)

        else:
            self.train = CifarDataSet(
                images=images,
                labels=labels,
                n_classes=self.n_classes,
                shuffle=shuffle,
                normalization=normalization,
                augmentation=self.augmentation)


        images , labels = self.read_cifar(test_fnames)
        self.test = CifarDataSet(
            images = images,
            labels = labels,
            shuffle=None,
            n_classes=None,
            normalization=normalization,
            augmentation=False)

        if validation_set and not validation_split:
            self.validation = self.test



    @property
    def save_path(self ):

        if self._save_path is None:
            self._save_path = os.path.join(tempfile.gettempdir() ,'cifar%d'  %self.n_classes)
        return self._save_path

    @property
    def data_url(self):
        self.n_classes
        url = 'http://www.cs.toronto.edu/~kriz/cifar-%d-python.tar.gz' % self.n_classes

        return url

    @property
    def get_filenames(self):
        raise NotImplementedError
    @property
    def data_shape(self):
        return (32, 32, 3)

    @property
    def n_classes(self):
        return self._n_classes

    def read_cifar(self , filenames):
        print 'read_cifar'
        if self.n_classes ==10:
            labels_key = b'labels'
        elif self.n_classes ==100:
            lables_keys = b'fine_labels'
        images_res=[]
        labels_res=[]

        print '\t',filenames
        for fname in filenames:
            with open(fname , 'rb') as f:
                images_and_labels=pickle.load(file=f)

            images = images_and_labels[b'data']
            labels = images_and_labels[b'labels']
            images = images.reshape(-1,3,32,32)
            images = images.swapaxes(1,3).swapaxes(1,2)
            images_res.append(images)
            labels_res.append(labels)
        images_res = np.vstack(images_res)
        labels_res = np.hstack(labels_res)
        print '\tlabel shape :', np.shape(images_res)
        print '\timage shape :', np.shape(labels_res)
        if self.one_hot:
            labels_res = self.labels_to_one_hot(labels_res)
        return images_res , labels_res


    @property
    def n_classes(self):
        return self._n_classes




class CifarDataSet(ImageDataSet):
    def __init__(self, images , labels , n_classes , shuffle , normalization , augmentation):

        if shuffle is None:
            self.shuffle_every_epoch=None
        elif shuffle == 'once_prior_train':
            self.shuffle_every_epoch=False
            images , labels = self.shuffle_images_and_labels(images , labels)
        elif shuffle == 'every_epoch':
            self.shuffle_every_epoch=True
        else:
            raise Exception("Unknown type of shuffling")
        self.images = images
        self.labels = labels
        self.n_classes = n_classes
        self.augmentation = augmentation
        self.normalization = normalization
        self.images = self.normalize_images(images  , self.normalization)
        self.start_new_epoch()

    def start_new_epoch(self):
        self._batch_counter=0

        if self.shuffle_every_epoch:
            images , labels = self.shuffle_images_and_labels(self.images , self.labels)
        else:
            images , labels = self.images , self.labels

        if self.augmentation:
            images = augment_all_images(images , pad=4)
        self.epoch_images = images
        self.epoch_labels = labels

    @property
    def num_examples(self):
        return self.labels.shape[0]

    def next_batch(self, batch_size):
        start = self._batch_counter * batch_size
        end = (self._batch_counter +1 ) * batch_size
        self._batch_counter +=1
        images_slice = self.epoch_images[start : end ]

        labels_slice = self.epoch_labels[start : end ]

        if images_slice.shape[0] != batch_size:
            self.start_new_epoch()
            return self.next_batch(batch_size)
        else:
            return images_slice , labels_slice

class Cifar10DataProvider(CifarDataProvider):
    _n_classes = 10
    data_augmentation = False

    def get_filenames(self, save_path):
        sub_save_path = os.path.join(save_path  , 'cifar-10-batches-py')
        train_filenames= [os.path.join(sub_save_path , 'data_batch_%d' % i) for i in range(1,6)]
        test_filenames = [os.path.join(sub_save_path , 'test_batch')]
        return train_filenames , test_filenames



if __name__ == '__main__':
    test_filenames , test_filenames = Cifar10DataProvider()



