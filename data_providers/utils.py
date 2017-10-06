from cifar import Cifar10DataProvider
from fundus import Fundus_NvsAbN_DataProvider

def get_data_provider_by_name(name , train_params):
    print 'debug | utils.py | get_data_provider_by_name'
    print 'name : ', name
    print "##train params##"
    print '\t', train_params
    if name == 'C10':
        return Cifar10DataProvider(**train_params)
    elif name == 'Fundus':
        return Fundus_NvsAbN_DataProvider(**train_params)

    else:
        print "Sorry , data provider for '%s' dataset was not implemented yet" %name
        exit()


if __name__ == '__main__':
    print 'a'
    train_params_fundus = {
        'save_path': '../dataset',
        'batch_size': 60,
        'n_epoch': 300,
        'initial_learning_rate': 0.1,
        'reduce_lr_epoch_1': 100,
        'reduce_lr_epoch_2': 200,
        'validation_set': True,
        'validation_split': None,
        'suffle': True,
        'normalization': 'divide_255',
    }
    get_data_provider_by_name('Fundus' , train_params_fundus)