def get_input_fn(data):
    from numpy import column_stack,array
    import tensorflow as tf
    from model.dense import IteratorInitHook

    input_data=array(column_stack((data._collective_pressure,data._collective_epsilon)))
    output_data=array(column_stack((data._collective_rho,data._collective_en)))

    init_hook=IteratorInitHook()

    def input_fn():

        dataset=tf.data.Dataset.from_tensor_slices((input_data,output_data))
        train_dataset=dataset.shuffle(len(input_data)).repeat().batch(256)

        iterator=tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)

        print(dataset)

        init_op=iterator.make_initializer(train_dataset)
        init_hook.iterator_init_op=init_op

        return iterator.get_next()

    return input_fn,init_hook

def get_predict_input_fn(data):
    from numpy import column_stack,array
    import tensorflow as tf
    from model.dense import IteratorInitHook

    init_hook=IteratorInitHook()

    def predict_fn():
        from numpy import linspace,ndarray,meshgrid

        dataset=tf.data.Dataset.from_tensor_slices((data))
        predict_dataset=dataset.batch(1)

        iterator=tf.data.Iterator.from_structure(predict_dataset.output_types,predict_dataset.output_shapes)

        init_op=iterator.make_initializer(predict_dataset)
        init_hook.iterator_init_op=init_op

        return iterator.get_next()

    return predict_fn,init_hook

def get_mu_input_fn(data):
    from numpy import column_stack,array
    import tensorflow as tf
    from model.dense import IteratorInitHook

    input_data=column_stack((data._collective_mu,data._collective_epsilon))
    output_data=column_stack((data._collective_rho,data._collective_en))

    init_hook=IteratorInitHook()

    def input_fn():

        dataset=tf.data.Dataset.from_tensor_slices((input_data,output_data))
        train_dataset=dataset.shuffle(len(input_data)).repeat().batch(256)

        iterator=tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)

        print(dataset)

        init_op=iterator.make_initializer(train_dataset)
        init_hook.iterator_init_op=init_op

        return iterator.get_next()

    return input_fn,init_hook
