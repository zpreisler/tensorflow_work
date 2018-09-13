#!/usr/bin/env python
class read_data:
    def __init__(self,files,offset=200):
        from glob import glob
        from numpy import array,fromfile,append,column_stack
        self.files=glob(files)
        self._collective_mu=array([])
        self._collective_epsilon=array([])
        self._collective_pressure=array([])
        self._collective_en=array([])
        self._collective_rho=array([])

        for f in self.files:
            name=f[:f.rfind('.')]
            mu=fromfile(name+".mu")
            epsilon=fromfile(name+".epsilon")
            pressure=fromfile(name+".pressure")
            en=fromfile(name+".en")
            rho=fromfile(name+".rho")

            self._collective_mu=append(self._collective_mu,mu[offset:])
            self._collective_epsilon=append(self._collective_epsilon,epsilon[offset:])
            self._collective_pressure=append(self._collective_pressure,pressure[offset:])
            self._collective_en=append(self._collective_en,en[offset:])
            self._collective_rho=append(self._collective_rho,rho[offset:])

        self.x=column_stack((self._collective_epsilon,self._collective_pressure))

def network(input_layer):
    import tensorflow as tf
    dense1a=tf.layers.dense(inputs=input_layer,units=7,activation=tf.nn.tanh)
    dense1b=tf.layers.dense(inputs=input_layer,units=2)
    dense1=tf.concat((dense1a,dense1b),axis=1)

    dense2a=tf.layers.dense(inputs=dense1,units=8,activation=tf.nn.tanh)
    dense2b=tf.layers.dense(inputs=dense1,units=2)
    dense2=tf.concat((dense2a,dense2b),axis=1)

    con=tf.concat((dense1,dense2),axis=1)

    return tf.layers.dense(inputs=con,units=2)

def main():
    print("map")
    from numpy import linspace,meshgrid,reshape,column_stack,array,zeros,ones
    from numpy.random import uniform,randint,normal
    from matplotlib.pyplot import tricontourf,contourf,show,figure,savefig,xlabel,ylabel,subplots_adjust,plot,colorbar,contour,clabel,subplots,title,xlim,ylim

    from tensorflow.train import Saver

    import tensorflow as tf

    d1=100
    d2=100

    x=linspace(0.0001,20,d1)
    y=linspace(0.0001,20,d2)
    Y,X=meshgrid(y,x)

    feed_c=[[a,b] for a in x for b in y]
    feed_z=zeros(d1*d1*2,dtype='float32')
    feed_z=reshape(feed_z,(-1,2))
    
    """
    training data
    """

    data=read_data("/home/zdenek/Projects/tensorflow/patchy_ann/data_4_p/*.conf")

    from hive import hive
    data2=hive("/home/zdenek/Projects/tensorflow/patchy_ann/data_4_p/*.conf")

    print(data._collective_mu[:10])
    print(data2._collective_mu[:10])

    y_in=data._collective_pressure*data._collective_epsilon
    x_in=1.0/(data._collective_epsilon)

    feed_c_in=array(column_stack((x_in,y_in)),dtype='float32')

    x_out=data._collective_rho
    y_out=data._collective_en

    feed_c_out=array(column_stack((x_out,y_out)),dtype='float32')

    feed_z_out=reshape(feed_c_out,(-1,2))
    feed_z_out=array(feed_z_out,dtype='float32')

    print(len(data._collective_rho))

    """
    train dataset
    """

    t_dataset=tf.data.Dataset.from_tensor_slices((feed_c_in,feed_z_out))
    train_dataset=t_dataset.repeat().shuffle(len(data._collective_en)).batch(512)

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
    y_evaluated=network(input_layer)

    cross_entropy=tf.reduce_mean(tf.nn.l2_loss((output_layer-y_evaluated)))
    optimizer=tf.train.AdamOptimizer(3e-3)
    minimize=optimizer.minimize(cross_entropy)

    init_vars=tf.group(tf.global_variables_initializer())

    saver=Saver()

    sess = tf.Session()

    try:
        saver.restore(sess,"/home/zdenek/tmp/prho.ckpt")
    except ValueError:
        sess.run(init_vars)
        pass

    sess.run(train_init_op)

    """
    Training
    """

    for i in range(2000):
        a,_=sess.run([cross_entropy,minimize])
        if i%400 == 0:
            print(i,a)

    """
    Evaluation
    """

    sess.run(validate_init_op)
    y_predicted=sess.run(y_evaluated)
    rho=reshape(y_predicted[:,0],(d1,d2))
    en=reshape(y_predicted[:,1],(d1,d2))


    fig,ax=subplots()

    contourf(X,Y,rho,interpolation=None,vmin=0,levels=[0.0,0.1,0.6,0.7,0.8,0.9,1.1],cmap="YlGn")
    cbar=colorbar()
    cbar.ax.set_ylabel(r"density $\rho$") 
    clines=contour(X,Y,rho,interpolation=None,vmin=0,levels=[0.0,0.1,0.6,0.7,0.8,0.9,1.1],colors='k',hold='on',linewidths=0.33)
    cbar.add_lines(clines)

    cc=contour(X,Y,en,interpolation=None,vmin=0.5,vmax=2.0,levels=[0.5,1.0,1.5,1.8,1.9,2.0],cmap="YlOrRd")
    clabel(cc,inline=1,fontsize=10,fmt="%.1lf")
    xlabel(r"pressure $p$")
    ylabel(r"$\epsilon$")

    xlim([0.0001,20])
    ylim([0.0001,20])

    title("energy per particle $U/N$")
    #plot(data.collective_epsilon[data.collective_en>1.9],data.collective_pressure[data.collective_en>1.9],"r.")

    subplots_adjust(bottom=0.17,left=0.17)
    savefig("_map.png")
    savefig("_map.pdf")

    saver.save(sess,"/home/zdenek/tmp/prho.ckpt")

    show()


if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl-C")
