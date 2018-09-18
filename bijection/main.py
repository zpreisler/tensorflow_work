#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def fce_p(x):
    return tf.log(x)

def fce_mu(x):
    return x**3

def bijection(X):
    with tf.variable_scope("bijection"):
        rho1=fce_p(X)
        
        d1=tf.layers.dense(inputs=X,units=8,activation=tf.nn.tanh)
        d2=tf.layers.dense(inputs=d1,units=8,activation=tf.nn.tanh)

        mu=tf.layers.dense(inputs=d2,units=1)
        rho2=fce_mu(mu)

        return rho1,rho2,mu

def main(argv):
    from numpy import linspace,array,reshape,random,log
    print("""Bijection""")

    X=tf.placeholder(tf.float32,[None,1])

    rho1,rho2,mu=bijection(X)

    loss=tf.reduce_mean(tf.nn.l2_loss((rho1-rho2)))

    optimizer=tf.train.AdamOptimizer(learning_rate=1e-3)
    train=optimizer.minimize(loss)
    init_vars=tf.group(tf.global_variables_initializer())

    session=tf.Session()
    session.run(init_vars)

    for i in range(20001):

        X_batch=reshape(random.uniform(1.1,10,64),(-1,1))

        _,m,r1,r2,rho_loss=session.run([train,mu,rho1,rho2,loss],feed_dict={X: X_batch})

        if i %10000 == 0: 
            print("x_batch:",X_batch)
            print("mu:",m)
            print("rho1:",r1)
            print("rho2:",r2)
            print("rho_loss:",rho_loss)

    X_batch=reshape(linspace(1.1,10,100),(100,1))
    m,r1,r2=session.run([mu,rho1,rho2],feed_dict={X: X_batch})

    from matplotlib.pyplot import show,plot,figure,xlabel,ylabel
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
    plot(m,(m**3.0))
    plot(m,r2,"o--",markersize=1.0)
    xlabel(r"$\mu$")
    ylabel(r"$\rho$")

    figure()
    plot(x,m,"ro",markersize=1.0)
    xlabel(r"$p$")
    ylabel(r"chemical potential $\mu$")

    show()

if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
