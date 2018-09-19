#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def fce_p(x):
    return tf.log(x)

def fce_mu(x):
    return 2.0*x

def projection(X):
    with tf.variable_scope("projection"):
        d1=tf.layers.dense(inputs=X,units=8,activation=tf.nn.tanh)
        d2=tf.layers.dense(inputs=d1,units=8,activation=tf.nn.tanh)

        out=tf.layers.dense(inputs=d2,units=1)

        return out

def bijection(X):
    with tf.variable_scope("bijection"):
        #rho1=fce_p(X)
        rho1=projection(X)
        
        d1=tf.layers.dense(inputs=X,units=8,activation=tf.nn.tanh)
        d2=tf.layers.dense(inputs=d1,units=8,activation=tf.nn.tanh)

        mu=tf.layers.dense(inputs=d2,units=1)
        rho2=fce_mu(mu)

        return rho1,rho2,mu

def main(argv):
    from numpy import linspace,array,reshape,random,log
    from matplotlib.pyplot import show,plot,figure,xlabel,ylabel
    print("""Bijection""")

    X=tf.placeholder(tf.float32,[None,1])
    Y=tf.placeholder(tf.float32,[None,1])

    """projection"""
    fp=projection(X)
    projection_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="projection")

    loss_fp=tf.reduce_mean(tf.nn.l2_loss((fp-Y)))
    optimizer_fp=tf.train.AdamOptimizer(learning_rate=1e-3)

    train_fp=optimizer_fp.minimize(loss_fp,var_list=projection_vars)

    """bijection"""
    rho1,rho2,mu=bijection(X)

    bijection_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="bijection/den")

    loss=tf.reduce_mean(tf.nn.l2_loss((rho1-rho2)))
    optimizer=tf.train.AdamOptimizer(learning_rate=1e-3)

    train=optimizer.minimize(loss,var_list=bijection_vars)

    init_vars=tf.group(tf.global_variables_initializer())

    session=tf.Session()

    session.run(init_vars)

    print("""projection:""",projection_vars)
    print("""bijection:""",bijection_vars)

    for i in range(10000):

        x=random.uniform(1.1,10,64)
        X_batch=reshape(x,(-1,1))
        Y_batch=reshape(log(x),(-1,1))

        _,fp_loss=session.run([train_fp,loss_fp],feed_dict={X: X_batch, Y: Y_batch})

    print("[fp_loss]",fp_loss)

    x=linspace(1.1,10,64)
    X_batch=reshape(x,(-1,1))
    Y_batch=reshape(log(x),(-1,1))

    y=session.run(fp,feed_dict={X: X_batch})

    figure()
    plot(x,log(x))
    plot(x,y,"go--",markersize=2.0)

    session.run(bijection_vars)

    for i in range(1000):

        x=random.uniform(1.1,10,64)
        X_batch=reshape(x,(-1,1))

        _,m,r1,r2,rho_loss=session.run([train,mu,rho1,rho2,loss],feed_dict={X: X_batch})

        if i %1000 == 0: 
            #print("x_batch:",X_batch)
            #print("mu:",m)
            #print("rho1:",r1)
            #print("rho2:",r2)
            print("[rho_loss]",rho_loss)

    X_batch=reshape(linspace(1.1,10,64),(-1,1))
    m,r1,r2=session.run([mu,rho1,rho2],feed_dict={X: X_batch})

    x=reshape(X_batch,(1,-1))[0]
    r1=reshape(r1,(1,-1))[0]
    r2=reshape(r2,(1,-1))[0]
    m=reshape(m,(1,-1))[0]

    figure()
    plot(x,log(x))
    plot(x,r1,"o--",markersize=1.0)
    xlabel(r"pressure $p$")
    ylabel(r"$\rho$")

    figure()
    plot(m,2.0*m)
    plot(m,r2,"o--",markersize=1.0)
    xlabel(r"$\mu$")
    ylabel(r"$\rho$")

    figure()
    plot(x,0.5*log(x),markersize=1.0)
    plot(x,m,"ro--",markersize=1.0)
    xlabel(r"$p$")
    ylabel(r"chemical potential $\mu$")

    show()

if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
