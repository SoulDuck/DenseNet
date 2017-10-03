from data_providers import utils


class Foo:
    def __init__(self, a=5, b=10):
        self.a = a
        self.b = b
        self.c=10


InstanceOfFoo = Foo()
print(vars(InstanceOfFoo))