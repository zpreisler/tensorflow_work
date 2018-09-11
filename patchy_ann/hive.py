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

        for f in self.files:
            name=f[:f.rfind('.')+1]
            for a in self.attr:
                t=fromfile(name+a)
                attr='__collective_'+a
                t=append(getattr(self,attr),t)
                setattr(self,attr,t)

    def __alloc_attr__(self):
        for a in self.attr:
            attr='__collective_'+a
            setattr(self,attr,[])

a=hive('data/*.conf')

print(a.__collective_mu)
