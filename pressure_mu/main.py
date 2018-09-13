#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

class IteratorInitHook(tf.train.SessionRunHook):
    def after_create_session(self, session, coord):
        print("Initializing iterator")
        session.run(self.iterator_init_op)

class network:
    def __init__(self,input_layer=None):

        self.input_layer=input_layer
        self.build_graph()

        with tf.variable_scope('training_counters'):
            self.global_step=tf.train.get_or_create_global_step()

        self.loss=None
        self.optimizer=None
        self.minimize=None

    def build_graph(self):
        import tensorflow as tf

        with tf.variable_scope('model'):
            dense=tf.layers.dense(inputs=self.input_layer,
                    units=8,
                    activation=tf.nn.tanh)

            dense2=tf.layers.dense(inputs=dense,
                    units=8,
                    activation=tf.nn.tanh)

            self.output_layer=tf.layers.dense(inputs=dense2,units=2)

    def define_loss(self,output_target):
        import tensorflow as tf
        self.loss=tf.reduce_mean(
                tf.nn.l2_loss(self.output_layer-output_target)
                )

    def define_optimizer(self):
        import tensorflow as tf
        self.optimizer=tf.train.AdamOptimizer(1e-3)

    def define_minimize(self):
        import tensorflow as tf
        self.minimize=self.optimizer.minimize(self.loss)

def model_fn(features,labels,mode):
    import tensorflow as tf
    model=network(features);

    model.define_loss(labels)
    model.define_optimizer()
    model.define_minimize()

    return tf.estimator.EstimatorSpec(loss=model.loss,
            train_op=model.minimize,
            mode=mode)

def get_input_fn(data):
    from numpy import column_stack,array

    input_data=column_stack((data._collective_mu,data._collective_epsilon))
    output_data=column_stack((data._collective_rho,data._collective_en))

    init_hook=IteratorInitHook()

    def input_fn():
        import tensorflow as tf

        dataset=tf.data.Dataset.from_tensor_slices((input_data,output_data))
        train_dataset=dataset.repeat().batch(256)

        iterator=tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)

        print(dataset)

        init_op=iterator.make_initializer(train_dataset)
        init_hook.iterator_init_op=init_op

        X,y=iterator.get_next()

        print(X,y)

        return X,y

    return input_fn,init_hook

def main(argv):
    import tensorflow as tf
    from hive import hive
    from pprint import pprint
    from numpy import column_stack,array

    data=hive("/home/zdenek/Projects/tensorflow/patchy_ann/data_4_p/*.conf")

    print(data.length)

    input_data=column_stack((data._collective_mu,data._collective_epsilon))
    output_data=column_stack((data._collective_rho,data._collective_en))

    dataset=tf.data.Dataset.from_tensor_slices((input_data,output_data))
    train_dataset=dataset.repeat().batch(32)

    iterator=tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
    next_element=iterator.get_next()
    train_init_op=iterator.make_initializer(train_dataset)

    input_layer=next_element[0]
    output_target=next_element[1]

    #model=network(input_layer);

    #model.define_loss(output_target)
    #model.define_optimizer()
    #model.define_minimize()

    #init_vars=tf.group(tf.global_variables_initializer())

    #with tf.Session() as session:
    #    session.run(init_vars)
    #    session.run(train_init_op)
        #a,b=session.run(next_element)

    #    for i in range(20):
    #        a,b=session.run([model.loss,model.minimize])
    #        print(a)

    #print("""###########""")

    train_input_fn,train_init_hook=get_input_fn(data)

    #print(train_input_fn)

    #print("""###########""")

    estimator=tf.estimator.Estimator(model_fn=model_fn, model_dir='log')
    
    train_spec=tf.estimator.TrainSpec(input_fn=train_input_fn,hooks=[train_init_hook])

    tf.logging.set_verbosity(tf.logging.INFO)

    estimator.train(input_fn=train_input_fn,hooks=[train_init_hook],steps=40)


    #init_vars=tf.group(tf.global_variables_initializer())

    #with tf.Session() as session:
    #    summary=tf.summary.FileWriter("summary",session.graph)

    #    session.run(init_vars)
    #    session.run(train_init_op)

    #    session.run(model.minimize)

    #    summary.close()

if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
