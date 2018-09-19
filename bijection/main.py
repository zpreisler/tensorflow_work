#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def fce_p(x):
    return tf.log(x)

def fce_mu(x):
    return 0.5*x
    #return 2.0*x**2

def proj_p(X):

    with tf.variable_scope("proj_p"):
        p1=tf.layers.dense(inputs=X,units=4,activation=tf.nn.tanh,name="p_1")
        p2=tf.layers.dense(inputs=p1,units=4,activation=tf.nn.tanh,name="p_2")

        out=tf.layers.dense(inputs=p2,units=1,name="p_out")

        return out

def proj_mu(X):

    with tf.variable_scope("proj_mu"):
        m1=tf.layers.dense(inputs=X,units=5,activation=tf.nn.tanh,name="m_1")
        m2=tf.layers.dense(inputs=m1,units=5,activation=tf.nn.tanh,name="m_2")

        out=tf.layers.dense(inputs=m2,units=1,name="m_out")

        return out

def bijection(X):

    with tf.variable_scope("proj_p",reuse=True):
        p1=tf.layers.dense(inputs=X,units=4,activation=tf.nn.tanh,name="p_1")
        p2=tf.layers.dense(inputs=p1,units=4,activation=tf.nn.tanh,name="p_2")

        rho1=tf.layers.dense(inputs=p2,units=1,name="p_out")

    with tf.variable_scope("bijection"):
        
        d1=tf.layers.dense(inputs=X,units=4,activation=tf.nn.tanh,name="b_1")
        d2=tf.layers.dense(inputs=d1,units=4,activation=tf.nn.tanh,name="b_2")

        mu=tf.layers.dense(inputs=d2,units=1,name="b_m")

    with tf.variable_scope("proj_mu",reuse=True):
        m1=tf.layers.dense(inputs=mu,units=5,activation=tf.nn.tanh,name="m_1")
        m2=tf.layers.dense(inputs=m1,units=5,activation=tf.nn.tanh,name="m_2")

        #rho2=fce_mu(mu)
        rho2=tf.layers.dense(inputs=m2,units=1,name="m_out")

        return rho1,rho2,mu

def main(argv):
    from numpy import linspace,array,reshape,random,log,sqrt
    from matplotlib.pyplot import show,plot,figure,xlabel,ylabel
    print("""Bijection""")

    X=tf.placeholder(tf.float32,[None,1])
    Y=tf.placeholder(tf.float32,[None,1])

    """projection"""
    out_p=proj_p(X)
    out_mu=proj_mu(X)

    """bijection"""
    rho1,rho2,mu=bijection(X)

    loss_proj_p=tf.reduce_mean(tf.nn.l2_loss((out_p-Y)))
    loss_proj_mu=tf.reduce_mean(tf.nn.l2_loss((out_mu-Y)))
    loss_bij=tf.reduce_mean(tf.nn.l2_loss((rho1-rho2)))

    vars_proj_p=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="proj_p")
    vars_proj_mu=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="proj_mu")
    vars_bij=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="bijection")

    optimizer_proj_p=tf.train.AdamOptimizer(learning_rate=1e-3,name="Adam_p")
    optimizer_proj_mu=tf.train.AdamOptimizer(learning_rate=1e-3,name="Adam_mu")
    optimizer_bij=tf.train.AdamOptimizer(learning_rate=1e-3,name="Adam_bij")

    train_proj_p=optimizer_proj_p.minimize(loss_proj_p,var_list=vars_proj_p)
    train_proj_mu=optimizer_proj_mu.minimize(loss_proj_mu,var_list=vars_proj_mu)
    train_bij=optimizer_bij.minimize(loss_bij,var_list=vars_bij)

    with tf.Session() as session:
        tf.global_variables_initializer().run(session=session)    
        writer=tf.summary.FileWriter("log",session.graph)

        print("""projection p:""")
        for v in vars_proj_p:
            print(v)
        print("""projection mu:""")
        for v in vars_proj_mu:
            print(v)
        print("""bijection:""")
        for v in vars_bij:
            print(v)

        for i in range(40000):

            x=random.uniform(1.1,10,256)
            X_batch=reshape(x,(-1,1))
            Y_batch=reshape(log(x),(-1,1))

            _,proj_p_loss=session.run([train_proj_p,loss_proj_p],feed_dict={X: X_batch, Y: Y_batch})

        print("[proj_p_loss]",proj_p_loss)

        xx=linspace(1.1,10,64)
        x=random.uniform(1.1,10,256)
        X_batch=reshape(xx,(-1,1))
        y=session.run(out_p,feed_dict={X: X_batch})

        figure()
        plot(xx,log(xx),"k:")
        plot(xx,y,"ro",alpha=0.33)

        y=session.run(rho1,feed_dict={X: X_batch})
        plot(xx,y,"gs",alpha=0.33)

        for i in range(40000):

            x=random.uniform(0.1,5,256)
            X_batch=reshape(x,(-1,1))
            Y_batch=reshape(fce_mu(x),(-1,1))

            _,proj_mu_loss=session.run([train_proj_mu,loss_proj_mu],feed_dict={X: X_batch, Y: Y_batch})

        print("[proj_mu_loss]",proj_mu_loss)

        for i in range(20000):

            x=random.uniform(1.1,10,256)
            X_batch=reshape(x,(-1,1))

            _,m,r1,r2,rho_loss=session.run([train_bij,mu,rho1,rho2,loss_bij],feed_dict={X: X_batch})

            if i %1000 == 0: 
                print("[rho_loss]",rho_loss)

        X_batch=reshape(linspace(1.1,10,64),(-1,1))
        r1,r2,m=session.run([rho1,rho2,mu],feed_dict={X: X_batch})

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
    plot(m,fce_mu(m))
    plot(m,r2,"o--",markersize=1.0)
    xlabel(r"$\mu$")
    ylabel(r"$\rho$")

    figure()
    plot(x,2*log(x),markersize=1.0)
    plot(x,m,"ro--",markersize=1.0)
    xlabel(r"$p$")
    ylabel(r"chemical potential $\mu$")

    show()

if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
