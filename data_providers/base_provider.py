# -*- coding: utf-8 -*-
import numpy as np

class DataSet:
    @property
    def num_examples(self):
        raise NotImplementedError

    def nex_batch(self , batch_size):
        raise NotImplementedError



class ImageDataSet(DataSet):
    #여기서 어떤 기능을 하는거지???

    def _measure_mean_and_std(self):
        means=[]
        stds=[]

        for ch in range(self.images.shape[-1]):
            means.append(np.mean(self.images[:,:,:,ch]))
            stds.append(np.std(self.images[:,:,:,ch]))
        self._mean = means
        self._stds = stds

    @property
    def images_means(self):
        if not hasattr(self , '_means'):
            self._measure_mean_and_std()
        return self._mean

    @property
    def images_stds(self):
        if not hasattr(self , '_stds'):
            self._measure_mean_and_std()
        return self._stds


    def shuffle_images_and_labels(self , images , labels):
        rand_indexes = np.random.permutation(images.shape[0])
        shuffled_images = images[rand_indexes]
        shuffle_labels = labels[rand_indexes]

        return shuffled_images , shuffle_labels


    def normalize_images(self,images , normalization_type):

        if normalization_type == 'divide_255':
            images = images/255.

        elif normalization_type == 'divide_256':
            images = images / 256.

        elif normalization_type == 'by_channels':
            print 'normalization start type : by channels'
            images = images.astype('float64')
            for i in range(images.shape[-1]):
                images[:,:,:,i] = ((images[:,:,:,i] - self.images_means[i])  /self.images_stds[i])
        else:
            raise Exception("Unkown type of normalization")
        return images





    def normalization_by_channels(self , initial_images):
        new_images = np.zeros(initial_images.shape)
        for i in range(initial_images.shape[0]):
            new_images[i] = self.normalization_by_channels(initial_images[i])
        return new_images


    def normalize_image_by_channel(self , image):
        new_image = np.zeros(image.shape)
        for chanel in range(3):
            mean = np.mean(image[:,:,chanel])
            std = np.std(image[:,:,chanel])
            new_image[:,:,chanel] = (image[:,:,chanel] - mean) /std
        return new_image




class DataProvider:
    @property
    def data_shape(self):
        raise NotImplementedError
    def n_classes(self):
        raise NotImplementedError

    def labels_to_one_hot(self , labels):
        new_labels=np.zeros((labels.shape[0] , self.n_classes))
        new_labels[range(labels.shape[0]) , labels] = np.ones(labels.shape)
        return new_labels

    def labels_from_one_hot(self, labels):
        return np.argmax(labels , axis=1)




