#!/usr/bin/env python

def f(x,y):
    return x**2+y**2

def main():
    print("map")
    from numpy import linspace,meshgrid,reshape,column_stack,array,zeros,ones
    from numpy.random import uniform,randint,normal
    from matplotlib.pyplot import contourf,show,figure

    import tensorflow as tf

    d1=20
    d2=20

    x=linspace(-5,5,d1)
    y=linspace(-5,5,d2)

    Y,X=meshgrid(y,x)

    feed_c=[[a,b] for a in x for b in y]
    feed_z=[f(a,b) for a,b in feed_c]
    Z=reshape(feed_z,(d1,d2))

    feed_z=reshape(feed_z,(-1,1))
    feed_z=array(feed_z,dtype='float32')

    """
    true dataset
    """


    figure()

    contourf(X,Y,Z,12,interpolation=None)

    d_in=10000

    x_in=uniform(-5,5,d_in)
    y_in=uniform(-5,5,d_in)


    feed_c_in=array(column_stack((x_in,y_in)),dtype='float32')
    feed_z_out=[f(a,b)+normal(0.0,0.1,1) for a,b in feed_c_in]

    feed_z_out=reshape(feed_z_out,(-1,1))
    feed_z_out=array(feed_z_out,dtype='float32')

    """
    train dataset
    """

    t_dataset=tf.data.Dataset.from_tensor_slices((feed_c_in,feed_z_out))
    train_dataset=t_dataset.repeat().batch(2000)

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
    
    dense1=tf.layers.dense(inputs=input_layer,units=28,activation=tf.nn.tanh,bias_initializer=tf.truncated_normal_initializer())
    dense2=tf.layers.dense(inputs=dense1,units=29,activation=tf.nn.tanh,bias_initializer=tf.truncated_normal_initializer())

    y_out=tf.layers.dense(inputs=dense2,units=1)

    cross_entropy=tf.reduce_mean(tf.nn.l2_loss((output_layer-y_out)))
    #cross_entropy=tf.reduce_mean((output_layer-y_out)**2)
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

    Z_predicted=reshape(y_predicted,(d1,d2))

    figure()
    contourf(X,Y,Z_predicted,12,interpolation=None)

    show()


if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl-C")
