class network(object):
    def __init__(self,inputs,outputs,name='network'):
        import tensorflow as tf
        print("Initialize %s"%name)
        self.rate=tf.placeholder(tf.float64)
        self.inputs=inputs
        self.outputs=outputs

        self.build_graph(inputs,output_dim=outputs.shape[-1],name=name)

        self.define_loss(outputs)
        self.define_optimizer()
        self.define_training()

    def build_graph(self,inputs,output_dim,name='network'):
        import tensorflow as tf
        with tf.variable_scope(name):
            self.dense_1=tf.layers.dense(inputs=inputs,
                    units=16,
                    activation=tf.nn.tanh,
                    name='d1')

            self.dense_2=tf.layers.dense(inputs=self.dense_1,
                    units=8,
                    activation=tf.nn.tanh,
                    name='d2')

            self.dense_3=tf.layers.dense(inputs=self.dense_2,
                    units=output_dim,
                    name='output_layer')

            self.output_layer=self.dense_3

    def define_loss(self,output_layer):
        import tensorflow as tf
        self.loss=tf.reduce_mean(
                tf.nn.l2_loss(
                    self.output_layer-output_layer
                    )
                )

    def define_optimizer(self,name="optimizer"):
        import tensorflow as tf
        self.optimizer=tf.train.AdamOptimizer(learning_rate=self.rate)

    def define_training(self):
        self.train=self.optimizer.minimize(self.loss)

class flow(object):
    def __init__(self,name='flow'):
        import tensorflow as tf

        self.c=data_feeder(files='eos/fluid*.conf',
                add_data=['.en','.rho'])

        self.data=self.c.feed(['epsilon','pressure','.en','.rho'])
        self.data_all=self.c.feed_data(['epsilon','pressure','.en','.rho'])

        inputs=self.data[:,:2] #epsilon,pressure
        outputs=self.data[:,2:] #en,rho

        next_element,self.init_train_op,self.init_eval_op=self.data_pipeline(inputs=inputs,
                outputs=outputs)

        """
        Network
        """
        self.nn=network(next_element['inputs'],next_element['outputs'],name="Kagami")

    def data_pipeline(self,inputs=None,outputs=None,batch=2048,n=1024):
        import tensorflow as tf
        from numpy import linspace,zeros,array
        """
        train dataset
        """
        length=len(inputs)
        if batch>length:
            batch=length

        dataset=tf.data.Dataset.from_tensor_slices( 
                {'inputs': inputs,
                    'outputs': outputs}
                )
        train_dataset=dataset.repeat().shuffle(length).batch(batch)

        iterator=tf.data.Iterator.from_structure(
                train_dataset.output_types,
                train_dataset.output_shapes)
        next_element=iterator.get_next()
        init_train_op=iterator.make_initializer(train_dataset)

        """
        eval dataset
        """

        x=[]
        for k in inputs.transpose():
            x+=[linspace(k.min(),k.max(),n)]
        x=array(x).transpose()
        z=zeros((n,outputs.shape[-1]))

        dataset=tf.data.Dataset.from_tensor_slices( 
                {'inputs': x,
                    'outputs': z}
                )
        eval_dataset=dataset.batch(n)

        init_eval_op=iterator.make_initializer(eval_dataset)

        return next_element,init_train_op,init_eval_op

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
