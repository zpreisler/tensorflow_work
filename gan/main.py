#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def sample_Z(m,n):
    from numpy import random
    return random.uniform(-1,1,size=[m,n])

def linear_Z(m):
    from numpy import linspace
    from numpy import column_stack
    t=linspace(-1,1,m)
    return column_stack((t,t))

def sample_X(m):
    xy=[]
    from numpy import random
    from numpy import column_stack
    #x=random.random_sample((m,))-0.5
    x=random.uniform(-0.5,0.5,m)
    y=x**2
    return column_stack((x,y))

def generator(Z,reuse=False):
    with tf.variable_scope("GAN/Generator",reuse=reuse):
        d1=tf.layers.dense(inputs=Z,units=16,activation=tf.nn.leaky_relu)
        d2=tf.layers.dense(inputs=d1,units=16,activation=tf.nn.leaky_relu)
        output_layer=tf.layers.dense(d1,units=2)

    return output_layer
        
def discriminator(X,reuse=False):
    with tf.variable_scope("GAN/Discriminator",reuse=reuse):
        d1=tf.layers.dense(inputs=X,units=16,activation=tf.nn.leaky_relu)
        d2=tf.layers.dense(inputs=d1,units=16,activation=tf.nn.leaky_relu)
        d3=tf.layers.dense(inputs=d2,units=2)
        output_layer=tf.layers.dense(inputs=d3,units=1)

    return output_layer,d3

def main(argv):
    print("Generative Adversarial Network")

    generator_sample=generator(Z)
    real_logits,real_rep=discriminator(X)
    fake_logits,fake_rep=discriminator(generator_sample,reuse=True)

    generator_loss=tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,
                labels=tf.ones_like(fake_logits))
            )

    discriminator_loss=tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits,
                labels=tf.ones_like(real_logits))+
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,
                labels=tf.zeros_like(fake_logits))
            )

    generator_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
    discriminator_vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

    generator_optimizer=tf.train.AdamOptimizer(learning_rate=1e-4)
    discriminator_optimizer=tf.train.AdamOptimizer(learning_rate=1e-4)

    train_generator=generator_optimizer.minimize(generator_loss,var_list=generator_vars)
    train_discriminator=discriminator_optimizer.minimize(discriminator_loss,var_list=discriminator_vars)

    batch_size=256
    steps=10000
    nsteps=24

    session=tf.Session()
    tf.global_variables_initializer().run(session=session)

    from matplotlib.pyplot import plot,show,figure,close,savefig

    count=0
    for i in range(steps):
        X_batch=sample_X(batch_size)
        Z_batch=sample_Z(batch_size,2)

        for _ in range(nsteps):
            _,dloss=session.run([train_discriminator,discriminator_loss],feed_dict={X: X_batch,Z: Z_batch})

        dreal,dfake=session.run([real_rep,fake_rep],feed_dict={X: X_batch, Z: Z_batch})

        for _ in range(nsteps):
            _,gloss=session.run([train_generator,generator_loss],feed_dict={Z: Z_batch})

        greal,gfake=session.run([real_rep,fake_rep],feed_dict={X: X_batch, Z: Z_batch})

        if i%50 == 0:
            print(dloss,gloss)
            
            Z_linear=linear_Z(batch_size)
            g=session.run(generator_sample,feed_dict={Z: Z_linear})

            figure()
            plot(X_batch[:,0],X_batch[:,1],'.',markersize=1.0)
            plot(g[:,0],g[:,1],'.',markersize=1.0)
            savefig("g_%03d.png"%(count))
            close()
            

            figure()
            plot(dreal[:,0],dreal[:,1],'.',markersize=1.0)
            plot(greal[:,0],greal[:,1],'.',markersize=1.0)
            plot(dfake[:,0],dfake[:,1],'.',markersize=1.0)
            plot(gfake[:,0],gfake[:,1],'.',markersize=1.0)
            savefig("fig_%03d.png"%(count))
            count+=1
            close()
    

if __name__=="__main__":
    import tensorflow as tf
    tf.app.run()
