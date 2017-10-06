#-*- coding:utf-8 -*-R
from data_providers import utils
import numpy as np
import time
"""
class Foo:
    def __init__(self, a=5, b=10):
        self.a = a
        self.b = b
        self.c=10


InstanceOfFoo = Foo()
print(vars(InstanceOfFoo))
"""

"""
class ParentOne:
    def __init__(self):
        print 'a'
    def func(self):
        print("ParentOne의 함수 호출!")

class ParentTwo:
    def __init__(self):
        print 'b'
    def func(self):
        print("ParentTwo의 함수 호출!")

class Child(ParentTwo ,ParentOne):
    def childFunc(self):
        ParentOne.func(self)
        ParentTwo.func(self)


c=Child()
c.childFunc()
"""



a=np.zeros([12000,300,300,3])
b=np.zeros([12000,300,300,3])
start_time=time.time()
c=np.vstack([a,b])
print time.time()-start_time
start_time=time.time()
c_1=np.concatenate([a,b] , axis=0)
print time.time()-start_time
d=np.ones(300)
e=np.ones(300)
f=np.hstack([d,e])

print np.shape(c)
print np.shape(f)
print np.shape(c_1)