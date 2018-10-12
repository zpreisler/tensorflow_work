class network(object):
    def __init__(self,inputs,name='network'):
        import tensorflow as tf
        print("Initialize %s"%name)
        self.build_graph(inputs,name=name)

    def build_graph(self,inputs,name='network'):
        import tensorflow as tf
        with tf.variable_scope(name):
            self.dense_1=tf.layers.dense(inputs=inputs,
                    units=6,
                    activation=tf.nn.tanh,
                    name='d1')

            self.dense_2=tf.layers.dense(inputs=self.dense_1,
                    units=4,
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
        self.define_dataset()

        self.nn=network(self.input_layer,name="Kagami")

        print("Initialize Flow")
        self.rate=tf.placeholder(tf.float64)
        self.define_loss()
        self.define_optimizer()
        self.define_training()

    def get_data(self):
        self.c=data_feeder('eos/fluid*.conf',add_data=['.en','.rho'])
        self.data=self.c.feed(['epsilon','pressure','.en','.rho'])
        self._data_=self.c.feed_data(['epsilon','pressure','.en','.rho'])

    def define_dataset(self,n_eval=1024):
        import tensorflow as tf
        from numpy import linspace
        """
        dataset
        """
        self.get_data()
        inputs=self.data[:,0].reshape(-1,1)
        outputs=self.data[:,3].reshape(-1,1)

        dataset=tf.data.Dataset.from_tensor_slices( 
                {'inputs': inputs,
                    'outputs': outputs}
                )
        train_dataset=dataset.repeat().shuffle(256).batch(256)

        inputs=linspace(inputs.min(),inputs.max(),n_eval).reshape(-1,1)
        dataset=tf.data.Dataset.from_tensor_slices( 
                {'inputs': inputs,
                    'outputs': inputs}
                )
        eval_dataset=dataset.repeat().batch(n_eval)

        iterator=tf.data.Iterator.from_structure(train_dataset.output_types,
                train_dataset.output_shapes)
        next_element=iterator.get_next()

        self.init_train_op=iterator.make_initializer(train_dataset)
        self.init_eval_op=iterator.make_initializer(eval_dataset)

        self.input_layer=next_element['inputs']
        self.output_layer=next_element['outputs']

    def define_loss(self):
        import tensorflow as tf
        self.loss=tf.reduce_mean(
                tf.nn.l2_loss(
                    self.nn.output_layer-self.output_layer
                    ))

    def define_training(self):
        self.train=self.optimizer.minimize(self.loss)

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

    def feed_data(self,names=[]):
        from numpy import array,append,vstack,concatenate,ones,hstack,full
        d=[]
        for x in self.dconf:
            t=[]
            w=[x[name].data for name in names if isinstance(x[name],data) is True]
            m=min([len(y) for y in w])

            for name in names:
                if isinstance(x[name],data) is False:
                    v=float(*x[name])
                    t+=[full(m,v)]

            w=array(t+w).transpose()
            d+=[w]

        d=concatenate(d)
        return d
