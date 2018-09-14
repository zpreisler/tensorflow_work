#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def main(argv):
    import tensorflow as tf
    from hive import hive
    from pprint import pprint
    from numpy import column_stack,array
    from model.dense import model_fn
    from model.data_pipeline import get_input_fn,get_mu_input_fn

    """
    interpolate from pressure
    """

    data_p=hive("data/p*.conf")

    p_train_input_fn,p_train_init_hook=get_input_fn(data_p)
    estimator=tf.estimator.Estimator(model_fn=model_fn, model_dir='logp')

    tf.logging.set_verbosity(tf.logging.INFO)

    estimator.train(input_fn=p_train_input_fn,hooks=[p_train_init_hook],steps=20000)

    """
    interpolate from chemical potential
    """

    data_mu=hive("data/gc*.conf")

    mu_train_input_fn,mu_predict_input_fn,mu_train_init_hook=get_mu_input_fn(data_mu)
    mu_estimator=tf.estimator.Estimator(model_fn=model_fn, model_dir='logmu')

    tf.logging.set_verbosity(tf.logging.INFO)

    mu_estimator.train(input_fn=mu_train_input_fn,hooks=[mu_train_init_hook],steps=20000)

if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
