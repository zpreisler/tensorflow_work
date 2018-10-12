class network:
    def __init__(self,inputs,name='network'):
        print("Initialize %s"%name)
        self.build_graph(inputs,name=name)

    def build_graph(self,inputs,name='network'):
        import tensorflow as tf
        with tf.variable_scope(name):
            self.dense_1=tf.layers.dense(inputs=inputs,
                    units=5,
                    activation=tf.nn.tanh,
                    name='d1')

            self.dense_2=tf.layers.dense(inputs=self.dense_1,
                    units=5,
                    activation=tf.nn.tanh,
                    name='d2')

            self.dense_3=tf.layers.dense(inputs=self.dense_2,
                    units=1,
                    activation=tf.nn.tanh,
                    name='output_layer')

            self.output_layer=self.dense_3

from myutils import configuration,data
class data_feeder(configuration):
    def __init__(self,files,add_data=[],delimiter=[':']):
        from glob import glob
        print("Data Feeder")
        files=glob(files)
        configuration.__init__(self,files,delimiter=delimiter,add_data=add_data)

        self.dsort(key=lambda x: float(*x['epsilon']))

    def __is_data__(self,name):
        return any([isinstance(x[name],data) for x in self.dconf])

    def get(self,name):
        if self.__is_data__(name):
            return [x[name].data.mean() for x in self.dconf]
        else:
            return [float(*x[name]) for x in self.dconf]

    def feed(self,names=[]):
        from numpy import array
        d=[]
        for x in self.dconf:
            v=[x[name] for name in names]
            dd=[]
            for t in v:
                if isinstance(t,data):
                    dd+=[t.data.mean()]
                else:
                    dd+=[float(*t)]
            d+=[dd]
        return array(d)

    def feed_data(self,names=[],dnames=[]):
        from numpy import array
        d=[]
        for x in self.dconf:
            dd=[]
            v=[x[name] for name in names]
            w=[x[name].data for name in dnames]
            
            print(w) 
            #for i,j in zip(*w):
            #    print(i,j)

            #for i,j in zip(*[w[0].data,w[1].data]):
            #   yy print(i,j)
            #print(w)
            dd+=v

            d+=[dd]

        return array(d)




