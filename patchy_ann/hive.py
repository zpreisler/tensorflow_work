#!/usr/bin/env python
class hive:
    """
    Class for reading simulation data and generating input/output for tensorflow
    """
    def __init__(self,files,offset=0):
        from glob import glob
        from numpy import array,fromfile,append,column_stack
        """
        Input files [files can be e.g. '*.conf']
        """
        self.files=glob(files)
        self.attr=['en','mu','rho','epsilon']
        self.__alloc_attr__()
        self.__get_attr__()

    def __get_attr__(self):
        from numpy import fromfile,append
        for f in self.files[:]:
            name=f[:f.rfind('.')+1]
            for a in self.attr:
                attr='__collective_'+a
                t=fromfile(name+a)
                q=getattr(self,attr)
                t=append(q,t)
                setattr(self,attr,t)

    def __alloc_attr__(self):
        for a in self.attr:
            attr='__collective_'+a
            setattr(self,attr,[])

a=hive('data/*.conf')

print(a.__collective_mu)

for x in a.__collective_mu:
    print(x)
