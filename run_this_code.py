#-*- coding:utf-8 -*-
import model
import argparse
from data_providers import utils
train_params_cifar ={
    'save_path':'./dataset',
    'batch_size': 64,
    'n_epochs': 300,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 150,  # epochs * 0.5
    'reduce_lr_epoch_2': 225,  # epochs * 0.75
    'validation_set': True,
    'validation_split': None,  # None or float
    'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
    'normalization': 'by_channels',  # None, divide_256, divide_255, by_channels

}

train_params_svhn = {
    'batch_size': 64,
    'n_epochs': 40,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 20,
    'reduce_lr_epoch_2': 30,
    'validation_set': True,
    'validation_split': None,  # you may set it 6000 as in the paper
    'shuffle': True,  # shuffle dataset every epoch or not
    'normalization': 'divide_255',
}




def get_train_params_by_name(name):
    if name in ['C10','C100','C10+','C100+']:
        return train_params_cifar
    if name == 'SVHN':
        return train_params_svhn


if '__main__' == __name__:
    parser  = argparse.ArgumentParser()
    parser.add_argument('--train' , action='store_true' , help='Train the model')
    parser.add_argument('--test' , action='store_true')
    parser.add_argument('--model_type' , '-m' , type=str , choices=['DenseNet' , 'DenseNet-BC'] , default='DenseNet')
    parser.add_argument('--growth_rate' , '-k' , type=int , choices=[12,24,40] , default=12 , help='growth rate for every layer')
    parser.add_argument('--depth' , '-d' , type=int , choices=[40 , 100 , 190 , 250] , default=40 , help ='Depth of every network')
    parser.add_argument('--dataset', '-ds' , type=str , choices=['C10', 'C10+', 'C100' , 'C100+' , 'SVHN' ] , default='C10')
    parser.add_argument('--total_blocks' , '-tb' , type=int , default=3 , metavar='')
    parser.add_argument('--keep_prob' , '-kp' , type=float , metavar='')
    parser.add_argument('--weight_decay' , '-wd' , type=float , default=1e-4 , metavar='')
    parser.add_argument('--nesterov_momentum' , '-nm' , type=float , default=0.9 , metavar='' )
    parser.add_argument('--reduction' , '-red' , type=float , default=0.5  , metavar='')

    parser.add_argument('--logs', dest='should_save_logs' ,action='store_true')
    parser.add_argument('--no_logs', dest='should_save_logs', action='store_false' )
    parser.set_defaults(should_save_logs=True)

    parser.add_argument('--saves' , dest='should_save_model' , action = 'store_true' )
    parser.add_argument('--no-saves' , dest='should_save_model', action ='store_false')
    parser.set_defaults(should_save_model=True)

    parser.add_argument('--renew_logs', dest='renew_logs' , action='store_true')
    parser.add_argument('--not-renew-logs' , dest='renew_logs' , action='store_false')
    parser.set_defaults(renew_logs=True)


    args=parser.parse_args()
    if not args.keep_prob:
        if args.dataset in ['C10' , 'C100' , 'SVHN']:
            args.keep_prob=0.8
        else:
            args.keep_prob=1.0
    if args.model_type == 'Densenet':
        args.bc_mode= False
        args.reduction = 1.0
    elif args.model_type=='DenseNet-BC':
        args.bc_mode=True


    model_params = vars(args)
    ### 임시적으로 사용함 ###
    args.train=True

    ### 임시적으로 사용함 ###


    if not args.train and not args.test:
        print("you should train or test ur network , plz check param")
        exit()
    train_params = get_train_params_by_name(args.dataset)

    for k,v in model_params.items():
        print "\t %s : %s "%(k,v)
    for k ,v in train_params.items():
        print "\t %s : %s " % (k, v)


    print("prepare training data..")
    data_provider = utils.get_data_provider_by_name(args.dataset , train_params)

    print data_provider
    print "initialize_model..."

    print model_params
    model=model.DenseNet(data_provider=data_provider , **model_params)


    if args.train:
        print "Data provider train images :" , data_provider.train.num_examples
        model.train_all_epochs(train_params)

    if args.test:
        if not args.train:
            args.load_model()
        print "Data provider test images : " , data_provider.test.num_examples
        print "Testing..."

        loss , accuracy = model.test(data_provider.test , batch_size =200)
        print "mean cross_entropy: %f , mean accuracy : %f" %(loss , accuracy)



