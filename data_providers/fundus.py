import tensorflow as tf
from base_provider import *
import sys
import os , glob
import random

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
        new_images[i] = augment_all_images(initial_images[i] , pad=4)
        return new_images



class FundusDataProvider(DataProvider):
    def __init__ (self , save_path=None , validation_set = None , validation_split=None , \
                  shuffle=None , normalization =None  ,one_hot =True , augmentation=False ,**kwargs):

        print "Debug start | FundusDataProvider | __init__ "
        print '\tsave_path:' ,save_path
        print '\tvalidation_set:',validation_set
        print '\tvalidation_split:',validation_split
        print '\tshuffle:',shuffle
        print '\tnormalization:',normalization
        print '\tone_hot:',one_hot

        self._save_path=save_path
        self.validation_set = validation_set
        self.validation_split=validation_split
        self.shuffle=shuffle
        self.normalization=normalization
        self.one_hot=one_hot
        self.augmentation=augmentation

        self.train_filenames=self.get_filenames(self._save_path)
        images , labels , fnames=self.read_fundus()

        print "Debug end | FundusDataProvider | __init__ "
    @property
    def data_shape(self):
        return (300,300,3)
    @property
    def n_classes(self):
        return self._n_classes


    def read_fundus(self):
        print 'Debug start | Class FundusDataProvider | def read_fundus'
        ret_images=[]
        ret_labels=[]
        ret_fnames=[]
        for f in self.train_filenames:
            images , labels , fnames=self.reconstruct_tfrecord_rawdata(f)
            ret_images.extend(images)
            ret_labels.extend(labels)
            ret_fnames.extend(fnames)

        print 'total # images ',np.shape(ret_images)
        print 'total # labels ',np.shape(ret_labels)
        print 'total # fnames ',np.shape(ret_fnames)
        return ret_images , ret_labels , ret_fnames
        print 'Debug end | Class FundusDataProvider | def read_fundus'
    @property
    def get_filenames(self):
        raise NotImplementedError

    def reconstruct_tfrecord_rawdata(self , tfrecord_path):
        debug_flag_lv0 = True
        debug_flag_lv1 = True
        if __debug__ == debug_flag_lv0:
            print 'debug start | batch.py | class tfrecord_batch | reconstruct_tfrecord_rawdata '

        print 'now Reconstruct Image Data please wait a second'
        reconstruct_image = []
        # caution record_iter is generator

        record_iter = tf.python_io.tf_record_iterator(path=tfrecord_path)
        n = len(list(record_iter))
        record_iter = tf.python_io.tf_record_iterator(path=tfrecord_path)

        print 'The Number of Data :', n
        ret_img_list = []
        ret_lab_list = []
        ret_filename_list = []
        for i, str_record in enumerate(record_iter):
            msg = '\r -progress {0}/{1}'.format(i, n)
            sys.stdout.write(msg)
            sys.stdout.flush()

            example = tf.train.Example()
            example.ParseFromString(str_record)

            height = int(example.features.feature['height'].int64_list.value[0])
            width = int(example.features.feature['width'].int64_list.value[0])
            raw_image = (example.features.feature['raw_image'].bytes_list.value[0])
            label = int(example.features.feature['label'].int64_list.value[0])
            filename = (example.features.feature['filename'].bytes_list.value[0])
            image = np.fromstring(raw_image, dtype=np.uint8)
            image = image.reshape((height, width, -1))
            ret_img_list.append(image)
            ret_lab_list.append(label)
            ret_filename_list.append(filename)
        ret_img = np.asarray(ret_img_list)
        ret_lab = np.asarray(ret_lab_list)
        if debug_flag_lv1 == True:
            print ''
            print 'images shape : ', np.shape(ret_img)
            print 'labels shape : ', np.shape(ret_lab)
            print 'length of filenames : ', len(ret_filename_list)
        return ret_img, ret_lab, ret_filename_list


class FundusDataSet(ImageDataSet):
    def __init__(self , images , labels , fnames , shuffle , normalization , augmentation):
        if self.shuffle is None:
            self.shuffle_every_epoch=None
        elif self.shuffle is 'once_prior_train':
            self.shuffle_every_epoch=False
        elif  self.shuffle is 'every_epoch':
            self.shuffle_every_epoch=True
        else:
            raise Exception('Unknown type of shuffling')

        self.images = images
        self.labels = labels
        self.fnames = fnames
        self.shuffle = shuffle
        self.normalization = normalization
        self.augmentataion = augmentation

        self.normalization_images(images , self.normalization)
        self.start_new_epoch()

    def start_new_epoch(self):
        self._batch_counter=0
        if self.shuffle_every_epoch:
            result= self.shuffle_args(self.images , self.labels , self.fnames)
            images, labels, fnames = result
        else:
            images, labels, fnames = self.images , self.labels , self.fnames

        if self.augmentataion:
            images = augment_all_images(images , pad=4)
        self.epoch_images = images
        self.epoch_lables = labels
        self.epoch_fnames = fnames
    @property
    def num_examples(self):
        return self.labels.shape[0]

    def next_batch(self , batch_size):
        start = self._batch_counter * batch_size
        end = (self._batch_counter + 1) * batch_size
        self._batch_counter += 1
        images_slice = self.epoch_images[start: end]
        labels_slice = self.epoch_labels[start: end]
        fnames_slice = self.epoch_fnames[start: end]

        if images_slice.shape[0] != batch_size:
            self.start_new_epoch()
            return self.next_batch(batch_size)
        else:
            return images_slice, labels_slice , fnames_slice


class Fundus_NvsAbN_DataProvider(FundusDataProvider):
    print '#Debug | Fundus_NvsAbN_DataProvider'
    _n_classes=2
    def get_filenames(self , save_path):
        train_filenames = []
        print '\t#Debug | def get_filenames'
        save_path=os.path.join(save_path , 'fundus_tfrecords')
        os.path.join(save_path, '*.tfrecord')
        tfrecord_paths=glob.glob(os.path.join(save_path,'*.tfrecord'))
        for tp in tfrecord_paths:
            train_filenames.append(tp)
        print train_filenames
        return train_filenames

