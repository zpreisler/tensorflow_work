#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
def model_fn(features,labels,mode):
    import tensorflow as tf
    from model.dense import network
    model=network(features);

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions={'input': model.input_layer,
                'output': model.output_layer}
        predictions=model.output_layer
        return tf.estimator.EstimatorSpec(mode=mode,
                predictions=predictions)

    if mode == tf.estimator.ModeKeys.TRAIN:

        model.define_loss(labels)
        model.define_optimizer()
        model.define_minimize()

        return tf.estimator.EstimatorSpec(loss=model.loss,
                train_op=model.minimize,
                mode=mode)

def get_input_fn(data):
    from numpy import column_stack,array
    import tensorflow as tf
    from model.dense import IteratorInitHook

    input_data=column_stack((data._collective_pressure,data._collective_epsilon))
    output_data=column_stack((data._collective_rho,data._collective_en))

    init_hook=IteratorInitHook()

    def input_fn():

        dataset=tf.data.Dataset.from_tensor_slices((input_data,output_data))
        train_dataset=dataset.shuffle(2048).repeat().batch(64)

        iterator=tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)

        print(dataset)

        init_op=iterator.make_initializer(train_dataset)
        init_hook.iterator_init_op=init_op

        return iterator.get_next()

    def predict_fn():

        dataset=tf.data.Dataset.from_tensor_slices((input_data))
        train_dataset=dataset.batch(1)

        iterator=tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)

        print(dataset)

        init_op=iterator.make_initializer(train_dataset)
        init_hook.iterator_init_op=init_op

        return iterator.get_next()

    return input_fn,predict_fn,init_hook

def main(argv):
    import tensorflow as tf
    from hive import hive
    from pprint import pprint
    from numpy import column_stack,array

    data=hive("data/p*.conf")

    print(data.length)

    train_input_fn,predict_input_fn,train_init_hook=get_input_fn(data)
    estimator=tf.estimator.Estimator(model_fn=model_fn, model_dir='log')

    tf.logging.set_verbosity(tf.logging.INFO)

    estimator.train(input_fn=train_input_fn,hooks=[train_init_hook],steps=10000)

    p=estimator.predict(input_fn=predict_input_fn,hooks=[train_init_hook])
    b=[ a[0] for a in p ]
    
    from matplotlib.pyplot import plot,show,figure
    from mpl_toolkits.mplot3d import Axes3D

    fig=figure()
    ax=fig.add_subplot(111,projection='3d')

    ax.scatter(data._collective_pressure,data._collective_epsilon,data._collective_rho,'bo',alpha=0.1)
    ax.scatter(data._collective_pressure,data._collective_epsilon,b,'g',alpha=0.1)

    show()

if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
