class network(object):
    def __init__(self,inputs,name='network'):
        import tensorflow as tf
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

class flow(object):
    def __init__(self,name='flow'):
        import tensorflow as tf
        """
        dataset
        """
        self.get_data()

        print("Initialize Flow")
        self.rate=tf.placeholder(tf.float64,1)

        self.define_optimizer()

    def get_data(self):
        self.c=data_feeder('eos/fluid7?.conf',add_data=['.en','.rho'])
        self.data=self.c.feed(['epsilon','pressure','.en','.rho'])
        self._data_=self.c.feed_data(['epsilon','pressure'],['.en','.rho'])
        print(self._data_)

    def define_optimizer(self,name="AdamOptimizer"):
        import tensorflow as tf
        self.optimizer=tf.train.AdamOptimizer(learning_rate=self.rate)

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
        from numpy import array,append,vstack,concatenate,ones,hstack,full
        #n=names+dnames
        #print(n)
        #d=self.conf
        #t=[x for x in n if isinstance(x,data) is True]
        #print(t)

        d=[]
        for x in self.dconf:
            t=[]
            w=[x[name].data for name in dnames if isinstance(x[name],data) is True]
            m=min([len(y) for y in w])

            for name in names:
                if isinstance(x[name],data) is False:
                    v=float(*x[name])
                    t+=[full(m,v)]

            w=array(t+w).transpose()
            d+=[w]

        d=concatenate(d)
        return d
