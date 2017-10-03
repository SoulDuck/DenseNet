from  .cifar import Cifar10DataProvider

def get_data_provider_by_name(name , train_params):
    if name == 'C10':
        print "##train params##"
        print 'def :get_data_provider_by_name'
        print '\t',train_params
        return Cifar10DataProvider(**train_params)
    else:
        print "Sorry , data provider for '%s' dataset was not implemented yet" %name
        exit()