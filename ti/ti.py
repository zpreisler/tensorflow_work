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

def feeder2(d):
    from numpy import array
    dd=[]
    for a in d:
        _eps=float(*a['epsilon'])
        _rho=a['.rho'].data.mean()
        dd+=[[_eps,_rho]]
    return array(dd)

def main(argv):
    from glob import glob
    from myutils import configuration
    from pprint import pprint
    from matplotlib.pyplot import figure,show,plot,xlabel,ylabel
    from numpy import array,linspace
    from model.model import network,data_feeder
    print("Reading configurations")

    c=data_feeder('eos/fluid8?.conf',add_data=['.en','.rho'])

    d=c.dconf

    rho=c.get('.rho')
    epsilon=c.get('epsilon')

    a=c.feed(['epsilon','.rho','.en'])
    b=c.feed_data(['epsilon'],['.rho','.en'])
    #print(a)
    print(b)

    dd=feeder2(d)
    fd=dd.transpose()

    print("dd",dd.shape)
    print(dd[:,0])
    e=fd[0]
    e=e.reshape(1,-1)
    print(e.shape)

    q=[]
    for e in fd[0]:
        q+=[[e]]
    q=array(q)

    w=[]
    for e in fd[1]:
        w+=[[e]]
    w=array(w)

    #print("q:",q,q.shape)

    t=linspace(1,8,2048)
    t=t.reshape(2048,1)
    #print("t:",t,t.shape)

    dataset=tf.data.Dataset.from_tensor_slices({'a': q,'b': w})
    vdataset=tf.data.Dataset.from_tensor_slices({'a': t, 'b': t})

    train_dataset=dataset.repeat().shuffle(len(q)).batch(256)
    v_dataset=vdataset.batch(2048)

    iterator=tf.data.Iterator.from_structure(train_dataset.output_types,
            train_dataset.output_shapes)
    next_element=iterator.get_next()

    init_op=iterator.make_initializer(train_dataset)
    v_init_op=iterator.make_initializer(v_dataset)

    input_layer=next_element['a']
    output_layer=next_element['b']

    nn=network(input_layer,name='t')
    out_rho=nn.output_layer


    rate=tf.placeholder(tf.float64)

    loss=tf.reduce_mean(tf.nn.l2_loss((out_rho-output_layer)))
    optimizer=tf.train.AdamOptimizer(learning_rate=rate,name="Adam")

    train=optimizer.minimize(loss)
    init_vars=tf.group(tf.global_variables_initializer())

    saver=tf.train.Saver()

    with tf.Session() as session: 
        session.run(init_vars)
        session.run(init_op)

        try:
            saver.restore(session,"log/last.ckpt")
        except tf.errors.NotFoundError:
            pass

        for i in range(5):
            l,_=session.run([loss,train],feed_dict={rate: 1e-2})
            if i%500 is 0:
                print(i,l)

        for i in range(25):
            l,_=session.run([loss,train],feed_dict={rate: 1e-4})
            if i%500 is 0:
                print(i,l)

        saver.save(session,"log/last.ckpt")

        session.run(v_init_op)
        a,b=session.run([input_layer,out_rho])
        #print(a,b)

    figure()
    e2,r2=dd.transpose()
    plot(epsilon,rho,"-",alpha=0.5)
    plot(e2,r2,',',alpha=0.1)
    
    plot(a,b)

    xlabel(r"$\beta$")
    ylabel(r"$\rho$")
    show()

if __name__=="__main__":
    tf.app.run()
