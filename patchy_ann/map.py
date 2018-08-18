#!/usr/bin/env python

def f(x,y):
    return x**2+y**2

class read_data:
    def __init__(self,files):
        from glob import glob
        from numpy import array,fromfile,append,column_stack
        self.files=glob(files)
        self.collective_mu=array([])
        self.collective_epsilon=array([])
        self.collective_en=array([])
        self.collective_rho=array([])
    
        offset=20

        for f in self.files:
            name=f[:f.rfind('.')]
            mu=fromfile(name+".mu")
            epsilon=fromfile(name+".epsilon")
            en=fromfile(name+".en")
            rho=fromfile(name+".rho")

            self.collective_mu=append(self.collective_mu,mu[offset:])
            self.collective_epsilon=append(self.collective_epsilon,epsilon[offset:])
            self.collective_en=append(self.collective_en,en[offset:])
            self.collective_rho=append(self.collective_rho,rho[offset:])

        self.x=column_stack((self.collective_epsilon,self.collective_mu))

def main():
    print("map")
    from numpy import linspace,meshgrid,reshape,column_stack,array,zeros,ones
    from numpy.random import uniform,randint,normal
    from matplotlib.pyplot import contourf,show,figure,savefig,xlabel,ylabel,subplots_adjust,plot,colorbar

    from tensorflow.train import Saver

    import tensorflow as tf

    d1=50
    d2=50

    x=linspace(-10,15,d1)
    y=linspace(0,15,d2)

    Y,X=meshgrid(y,x)

    feed_c=[[a,b] for a in x for b in y]
    feed_z=[f(a,b) for a,b in feed_c]
    Z=reshape(feed_z,(d1,d2))

    feed_z=reshape(feed_z,(-1,1))
    feed_z=array(feed_z,dtype='float32')

    """
    true dataset
    """

    #figure()

    #contourf(X,Y,Z,12,interpolation=None)

    #xlabel(r"$x$")
    #ylabel(r"$y$")
    #subplots_adjust(bottom=0.2,left=0.2)

    #savefig("_map0.png")


    """
    training data
    """

    #data=read_data("/home/zdenek/Projects/tensorflow/patchy_ann/data/*.conf")
    data=read_data("/home/zdenek/tmp/t/y/ww*.conf")

    d_in=10000

    x_in=uniform(-5,5,d_in)
    y_in=uniform(-5,5,d_in)

    x_in=data.collective_mu
    y_in=data.collective_epsilon


    feed_c_in=array(column_stack((x_in,y_in)),dtype='float32')
    #feed_z_out=[a for a in feed_c_in]
    feed_z_out=data.collective_rho

    print(len(data.collective_en))

    feed_z_out=reshape(feed_z_out,(-1,1))
    feed_z_out=array(feed_z_out,dtype='float32')

    """
    train dataset
    """

    t_dataset=tf.data.Dataset.from_tensor_slices((feed_c_in,feed_z_out))
    train_dataset=t_dataset.repeat().shuffle(len(data.collective_rho)).batch(512)

    """
    Iterator
    """
    iterator=tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
    next_element=iterator.get_next()
    train_init_op=iterator.make_initializer(train_dataset)

    tr_dataset=tf.data.Dataset.from_tensor_slices((feed_c,feed_z))
    true_dataset=tr_dataset.batch(d1*d2)

    validate_init_op=iterator.make_initializer(true_dataset)

    input_layer=next_element[0]
    output_layer=next_element[1]

    """
    ANN
    """
    
    #dense1=tf.layers.dense(inputs=input_layer,units=8,activation=tf.nn.tanh,bias_initializer=tf.truncated_normal_initializer())
    #dense2=tf.layers.dense(inputs=dense1,units=9,activation=tf.nn.tanh,bias_initializer=tf.truncated_normal_initializer())

    dense1a=tf.layers.dense(inputs=input_layer,units=8,activation=tf.nn.tanh)
    dense1b=tf.layers.dense(inputs=input_layer,units=2)
    dense1=tf.concat((dense1a,dense1b),axis=1)

    dense2a=tf.layers.dense(inputs=dense1,units=9,activation=tf.nn.tanh)
    dense2b=tf.layers.dense(inputs=dense1,units=2)
    dense2=tf.concat((dense2a,dense2b),axis=1)

    y_out=tf.layers.dense(inputs=dense2,units=1)

    cross_entropy=tf.reduce_mean(tf.nn.l2_loss((output_layer-y_out)))
    #cross_entropy=tf.reduce_mean((output_layer-y_out)**2)
    optimizer=tf.train.AdamOptimizer(0.01)
    minimize=optimizer.minimize(cross_entropy)

    init_vars=tf.group(tf.global_variables_initializer())

    saver=Saver()

    sess = tf.Session()

    try:
        saver.restore(sess,"/home/zdenek/tmp/rho.ckpt")
    except ValueError:
        sess.run(init_vars)
        pass

    sess.run(train_init_op)

    """
    Training
    """

    for i in range(14000):
        a,b=sess.run([cross_entropy,minimize])
        if i%400 == 0:
            print(i,a,b)


            #sess.run(validate_init_op)
            #y_predicted=sess.run(y_out)
            #Z_predicted=reshape(y_predicted,(d1,d2))

            #figure()
            #contourf(X,Y,Z_predicted,12,interpolation=None)
            #xlabel(r"$x %d$"%i)
            #ylabel(r"$y$")

            #sess.run(train_init_op)

    """
    Evaluation
    """
    sess.run(validate_init_op)
    y_predicted=sess.run(y_out)
    Z_predicted=reshape(y_predicted,(d1,d2))

    figure()
    contourf(X,Y,Z_predicted,16,interpolation=None)
    xlabel(r"$x$")
    ylabel(r"$y$")

    #plot(data.collective_epsilon[data.collective_rho>0.1],data.collective_mu[data.collective_rho>0.1],"b.",alpha=0.1)
    #plot(data.collective_mu[data.collective_rho>0.7],data.collective_epsilon[data.collective_rho>0.7],"g.",alpha=0.001)
    #plot(data.collective_mu[data.collective_en>0.9],data.collective_epsilon[data.collective_en>0.9],"g.",alpha=0.002)
    #plot(data.collective_mu[data.collective_en>1.45],data.collective_epsilon[data.collective_en>1.45],"r.",alpha=0.001)
    #plot(data.collective_mu,data.collective_epsilon,"k.",alpha=0.001)
    colorbar()

    subplots_adjust(bottom=0.2,left=0.2)

    savefig("_map.png")

    saver.save(sess,"/home/zdenek/tmp/rho.ckpt")

    show()


if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl-C")
