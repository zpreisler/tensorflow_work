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
        y=0
        for h,mu,sigma in self.v:
            y+=h*self.gaussian(x,mu,sigma)
        noise=normal(0.0,s,len(y))
        return y/self.n+noise

    def gaussian(self,x,mu,sigma):
        from numpy import exp,sqrt,pi
        return 1.0/sqrt(2*pi*sigma**2)*exp(-(x-mu)**2/(2.0*sigma**2))

def main():
    print("linear fit")
    import tensorflow as tf
    from tensorflow.contrib.tensor_forest.python import tensor_forest
    from tensorflow.python.ops import resources

    from numpy.random import uniform
    from numpy import arange,linspace,array,sqrt,reshape,log,cast
    from matplotlib.pyplot import figure,plot,show,xlabel,ylabel,legend,savefig,subplots_adjust

    rr=landscape()

    x=uniform(-1,2,20000)
    y=rr.y(x,0.333)
    #y=rr.y(x,0.01)

    x=array(x,dtype='float32')
    y=array(y,dtype='float32')

    feed_x=reshape(x,(-1,1))
    feed_y=reshape(y,(-1,1))

    dataset=tf.data.Dataset.from_tensor_slices((feed_x,feed_y))
    print(dataset)
    train_dataset=dataset.repeat().batch(2000)

    d=1000
    xo=linspace(-1,2,d)
    yo=rr.y(xo)

    xo=array(xo,dtype='float32')
    yo=array(yo,dtype='float32')

    feed_x_true=reshape(xo,(-1,1))
    feed_y_true=reshape(yo,(-1,1))

    true_dataset=tf.data.Dataset.from_tensor_slices((feed_x_true,feed_y_true))
    validate_dataset=true_dataset.repeat().batch(d)

    """
    Iterator
    """

    iterator=tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
    next_element=iterator.get_next()
    train_init_op=iterator.make_initializer(train_dataset)
    validate_init_op=iterator.make_initializer(validate_dataset)

    input_layer=next_element[0]
    output_layer=next_element[1]

    """
    ANN
    """
    
    dense1=tf.layers.dense(inputs=input_layer,units=8,activation=tf.nn.tanh,bias_initializer=tf.truncated_normal_initializer())
    dense2=tf.layers.dense(inputs=dense1,units=9,activation=tf.nn.tanh,bias_initializer=tf.truncated_normal_initializer())
    y_out=tf.layers.dense(inputs=dense2,units=1)

    #cross_entropy=tf.reduce_mean(tf.squared_difference(output_layer,y_out))
    cross_entropy=tf.reduce_mean(tf.nn.l2_loss(output_layer-y_out))
    optimizer=tf.train.AdamOptimizer(0.01)
    minimize=optimizer.minimize(cross_entropy)

    init_vars=tf.group(tf.global_variables_initializer())

    sess = tf.Session()

    sess.run(train_init_op)
    sess.run(init_vars)

    """
    Training
    """

    for i in range(10000):
        a,b=sess.run([cross_entropy,minimize])
        if i%200 == 0:
            print(i,a,b)

    """
    Evaluation
    """

    sess.run(validate_init_op)
    y_predicted=sess.run(y_out)

    figure()
    plot(x,y,".",alpha=0.005)
    plot(xo,yo,"-",alpha=0.5,label=r"true")
    plot(xo,y_predicted,"--",alpha=0.5,label="predicted")
    xlabel(r"$x$")
    ylabel(r"$y$")
    legend(frameon=False,loc='best')
    subplots_adjust(bottom=0.2,left=0.2)
    savefig("_simple.png")
    show()

if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl-C")
