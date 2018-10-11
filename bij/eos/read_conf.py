#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
def feeder(d):
    from numpy import array
    dd=[]
    for a in d:
        _eps=float(*a['epsilon'])
        for _rho in a['.rho'].data:
            dd+=[[_eps,_rho]]
    return array(dd)

def net(x):
    with tf.variable_scope("net"):
        p1=tf.layers.dense(inputs=x,units=6,activation=tf.nn.tanh,name="p_1")
        p2=tf.layers.dense(inputs=p1,units=6,activation=tf.nn.tanh,name="p_2")

        out=tf.layers.dense(inputs=p2,units=1,name="p_out")

        return out

def main(argv):
    from glob import glob
    from myutils import configuration
    from pprint import pprint
    from matplotlib.pyplot import figure,show,plot,xlabel,ylabel
    from numpy import array
    print("Reading configurations")

    c=configuration(glob('fluid*.conf'))

    c.data('.en')
    c.data('.rho')
    c.dsort()

    d=c.dconf
    rho=[x['.rho'].data.mean() for x in d]
    epsilon=[float(*x['epsilon']) for x in d]

    dd=feeder(d)
    fd=dd.transpose()

    q=[]
    for e in fd[0]:
        q+=[[e]]
    q=array(q)

    w=[]
    for e in fd[1]:
        w+=[[e]]
    w=array(w)

    print("q:",q,q.shape)

    dataset0=tf.data.Dataset.from_tensor_slices(q)
    print(dataset0.output_types)
    print(dataset0.output_shapes)

    dataset1=tf.data.Dataset.from_tensor_slices(w)
    print(dataset1.output_types)
    print(dataset1.output_shapes)

    dataset=tf.data.Dataset.zip((dataset0,dataset1))

    train_dataset=dataset.repeat().batch(256)

    iterator=tf.data.Iterator.from_structure(train_dataset.output_types,
            train_dataset.output_shapes)
    next_element=iterator.get_next()

    input_layer=next_element[0]
    output_layer=next_element[1]

    init_op=iterator.make_initializer(train_dataset)

    out_rho=net(input_layer)

    loss=tf.reduce_mean(tf.nn.l2_loss((out_rho-output_layer)))
    optimizer=tf.train.AdamOptimizer(learning_rate=1e-3,name="Adam")

    train=optimizer.minimize(loss)
    init_vars=tf.group(tf.global_variables_initializer())

    with tf.Session() as session: 
        session.run(init_op)
        session.run(init_vars)

        #print(next_element)
        #b=session.run(next_element)
        #print(b)

        for i in range(1000):
            l,_=session.run([loss,train])
            print(l)

    figure()
    e2,r2=dd.transpose()
    plot(epsilon,rho,"-",alpha=0.5)
    plot(e2,r2,',',alpha=0.1)
    xlabel(r"$\beta$")
    ylabel(r"$\rho$")
    show()


if __name__=="__main__":
    tf.app.run()
