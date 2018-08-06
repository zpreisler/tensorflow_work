#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def gaussian(x,mu,sigma):
    from numpy import exp,sqrt,pi
    return 1.0/sqrt(2*pi*sigma**2)*exp(-(x-mu)**2/(2.0*sigma**2))

def landscape(x,v):
    y=0
    for h,mu,sigma in v:
        #print(h,mu,sigma)
        y+=h*gaussian(x,mu,sigma)
    return y/len(v)

def landscape_params(n=100):
    from numpy.random import rand
    from numpy import sqrt
    h=rand(100,3)
    h[:,0]=h[:,0]*50-40
    h[:,1]=h[:,1]*5-2
    h[:,2]=h[:,2]*2.0*sqrt(10.0)+1e-3
    return h

def main():
    print("linear fit")
    import tensorflow as tf
    from tensorflow.contrib.tensor_forest.python import tensor_forest
    from tensorflow.python.ops import resources

    from numpy.random import rand,uniform,normal
    from numpy import arange,linspace,array,sqrt,reshape,log
    from matplotlib.pyplot import figure,plot,show

    h=landscape_params()

    x=linspace(-1,2,100,dtype='float32')
    x=uniform(-1,2,900000)
    x=array(x,dtype='float32')
    #y=landscape(x,h)+uniform(-0.05,0.05,100000)
    y=landscape(x,h)+normal(0.0,1e-1,900000)
    y=array(y,dtype='float32')
    #y=log(x)

    xx=reshape(x,(-1,1))
    yy=reshape(y,(-1,1))

    dataset=tf.data.Dataset.from_tensor_slices((xx,yy))
    train_dataset=dataset.repeat().batch(1024)

    xq=linspace(-1,2,1000,dtype='float32')
    yq=landscape(xq,h)

    xxq=reshape(xq,(-1,1))
    yyq=reshape(yq,(-1,1))

    qdataset=tf.data.Dataset.from_tensor_slices((xxq,yyq))
    validate_dataset=qdataset.repeat().batch(1000)

    print(dataset)

    iterator=tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
    next_element=iterator.get_next()
    train_init_op=iterator.make_initializer(train_dataset)
    validate_init_op=iterator.make_initializer(validate_dataset)

    input_layer=next_element[0]
    output_layer=next_element[1]

    num_classes=1
    num_features=1
    num_trees=37
    max_nodes=871

    hparams=tensor_forest.ForestHParams(
            num_classes=num_classes,
            num_features=num_features,
            num_trees=num_trees,
            max_nodes=max_nodes,
            regression=True).fill()

    forest_graph=tensor_forest.RandomForestGraphs(hparams)

    train_op=forest_graph.training_graph(input_layer,output_layer)
    loss_op=forest_graph.training_loss(input_layer,output_layer)

    infer_op,_,_= forest_graph.inference_graph(input_layer)

    init_vars=tf.group(tf.global_variables_initializer(),resources.initialize_resources(resources.shared_resources()))

    sess = tf.Session()

    sess.run(train_init_op)
    sess.run(init_vars)

    for i in range(333):
        a,b=sess.run([train_op,loss_op])
        print(a,b)

    sess.run(validate_init_op)
    yt=sess.run(infer_op)
    print(yt)
    from matplotlib.pyplot import show,figure,plot
    figure()
    #plot(x,y,".",alpha=0.005)
    plot(xq,yq,"-",alpha=0.5)
    plot(xq,yt,"--",alpha=0.5)
    show()

if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrl-C")
