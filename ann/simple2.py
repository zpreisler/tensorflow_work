#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
class landscape:
    """
    Generate random landscape
    """
    def __init__(self,n=64):
        from numpy.random import rand
        from numpy import sqrt
        """
        n: number of gaussians
        """
        self.n=n
        v=rand(n,3)
        v[:,0]=v[:,0]*50-40
        v[:,1]=v[:,1]*5-2
        v[:,2]=v[:,2]*2.0*sqrt(10.0)
        self.v=v

    def y(self,x,s=0.0):  
        from numpy.random import uniform,normal
        from numpy import array
        y=0
        for h,mu,sigma in self.v:
            y+=h*self.gaussian(x,mu,sigma)
        noise=normal(0.0,s,len(y))
        a=y/self.n+noise
        return array(a,dtype='float32')

    def gaussian(self,x,mu,sigma):
        from numpy import exp,sqrt,pi
        return 1.0/sqrt(2*pi*sigma**2)*exp(-(x-mu)**2/(2.0*sigma**2))

class data:
    def __init__(self,x):
        self.x=x

    def add_column(self,v):
        from numpy import full,column_stack,array
        a=full(len(self.x),v)
        self.x2=array(column_stack((self.x,a)),dtype='float32')

def main():
    print("linear fit")
    import tensorflow as tf
    from tensorflow.contrib.tensor_forest.python import tensor_forest
    from tensorflow.python.ops import resources

    from numpy.random import uniform
    from numpy import arange,linspace,array,sqrt,reshape,log,cast,full,column_stack
    from matplotlib.pyplot import figure,plot,show,xlabel,ylabel,legend,savefig,subplots_adjust

    d=20000
    """
    first dataset
    """
    rr1=landscape()

    x1=uniform(-1,2,d)

    x1_data=data(x1)
    x1_data.add_column(1.0)

    y1=rr1.y(x1_data.x,0.333)

    feed_x1=reshape(x1_data.x2,(-1,2))
    feed_y1=reshape(y1,(-1,1))

    dataset1=tf.data.Dataset.from_tensor_slices((feed_x1,feed_y1))

    """
    second dataset
    """
    rr2=landscape()

    x2=uniform(-1,2,d)

    x2_data=data(x2)
    x2_data.add_column(2.0)

    y2=rr2.y(x2_data.x,0.333)

    feed_x2=reshape(x2_data.x2,(-1,2))
    feed_y2=reshape(y2,(-1,1))

    print(feed_x2)

    dataset2=tf.data.Dataset.from_tensor_slices((feed_x2,feed_y2))

    dataset=dataset1.concatenate(dataset2)

    print(dataset)
    train_dataset=dataset.shuffle(2*d).repeat().batch(4000)

    d=1000

    """
    first dataset
    """

    xo1=linspace(-1,2,d)

    xo1_data=data(xo1)
    xo1_data.add_column(1.0)

    yo1=rr1.y(xo1_data.x)

    feed_x1_true=reshape(xo1_data.x2,(-1,2))
    feed_y1_true=reshape(yo1,(-1,1))

    true_dataset1=tf.data.Dataset.from_tensor_slices((feed_x1_true,feed_y1_true))
    validate_dataset1=true_dataset1.repeat().batch(d)

    """
    second dataset
    """

    xo2=linspace(-1,2,d)

    xo2_data=data(xo2)
    xo2_data.add_column(2.0)

    yo2=rr2.y(xo2_data.x)

    feed_x2_true=reshape(xo2_data.x2,(-1,2))
    feed_y2_true=reshape(yo2,(-1,1))

    true_dataset2=tf.data.Dataset.from_tensor_slices((feed_x2_true,feed_y2_true))
    validate_dataset2=true_dataset2.repeat().batch(d)


    """
    Iterator
    """

    iterator=tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
    next_element=iterator.get_next()
    train_init_op=iterator.make_initializer(train_dataset)

    validate_init_op1=iterator.make_initializer(validate_dataset1)
    validate_init_op2=iterator.make_initializer(validate_dataset2)

    input_layer=next_element[0]
    output_layer=next_element[1]

    """
    ANN
    """
    
    dense1=tf.layers.dense(inputs=input_layer,units=28,activation=tf.nn.tanh,bias_initializer=tf.truncated_normal_initializer())
    dense2=tf.layers.dense(inputs=dense1,units=29,activation=tf.nn.tanh,bias_initializer=tf.truncated_normal_initializer())
    y_out=tf.layers.dense(inputs=dense2,units=1)

    cross_entropy=tf.reduce_mean(tf.nn.l2_loss(output_layer-y_out))
    optimizer=tf.train.AdamOptimizer(0.01)
    minimize=optimizer.minimize(cross_entropy)

    init_vars=tf.group(tf.global_variables_initializer())

    sess = tf.Session()

    sess.run(train_init_op)
    sess.run(init_vars)

    print(sess.run(next_element[0]))
    print(sess.run(next_element[1]))

    """
    Training
    """

    for i in range(20000):
        a,b=sess.run([cross_entropy,minimize])
        if i%200 == 0:
            print(i,a,b)

    """
    Evaluation
    """

    sess.run(validate_init_op1)
    y1_predicted=sess.run(y_out)

    sess.run(validate_init_op2)
    y2_predicted=sess.run(y_out)

    figure()
    plot(x1,y1,".",alpha=0.01)
    plot(x2,y2,".",alpha=0.01)

    plot(xo1,yo1,"-",alpha=0.5,label=r"true")
    plot(xo2,yo2,"-",alpha=0.5,label=r"true$_2$")

    plot(xo1,y1_predicted,"--",alpha=0.5,label="predicted")
    plot(xo2,y2_predicted,":",alpha=0.5,label="predicted$_2$")

    xlabel(r"$x$")
    ylabel(r"$y$")

    legend(frameon=False,loc='best')
    subplots_adjust(bottom=0.2,left=0.2)
    savefig("_simple2.png")

    show()

if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl-C")
