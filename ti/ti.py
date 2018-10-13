#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def main(argv):
    from pprint import pprint
    from matplotlib.pyplot import figure,show,plot,xlabel,ylabel
    from model.model import network,data_feeder,flow
    fl=flow()

    rho=fl.c.get('.rho')
    en=fl.c.get('.en')
    epsilon=fl.c.get('epsilon')

    init_vars=tf.group(tf.global_variables_initializer())

    saver=tf.train.Saver()

    with tf.Session() as session: 
        session.run(init_vars)
        session.run(fl.init_train_op)

        try:
            saver.restore(session,"log/last.ckpt")
        except tf.errors.NotFoundError:
            pass

        for i in range(25000):
            l,_=session.run([fl.nn.loss,fl.nn.train],feed_dict={fl.nn.rate: 1e-3})
            if i%500 is 0:
                print(i,l)

        saver.save(session,"log/last.ckpt")

        session.run(fl.init_eval_op)
        a,b=session.run([fl.nn.inputs,fl.nn.output_layer])
        print(a,b)

    figure()
    plot(epsilon,rho,"-",alpha=0.5)
    plot(fl.data_all[:,0],fl.data_all[:,3],',',alpha=0.1)
    
    plot(a[:,0],b[:,1])
    xlabel(r"$\beta$")
    ylabel(r"$\rho$")

    figure()
    plot(epsilon,en,"-",alpha=0.5)
    plot(fl.data_all[:,0],fl.data_all[:,2],',',alpha=0.1)
    plot(a[:,0],b[:,0])

    xlabel(r"$\beta$")
    ylabel(r"$\rho$")
    show()

if __name__=="__main__":
    tf.app.run()
