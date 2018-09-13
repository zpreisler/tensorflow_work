import tensorflow as tf
class IteratorInitHook(tf.train.SessionRunHook):
    def after_create_session(self, session, coord):
        print("Initializing iterator")
        session.run(self.iterator_init_op)

class network:
    def __init__(self,input_layer=None):

        self.input_layer=input_layer
        self.build_graph()

        with tf.variable_scope('training_counter'):
            self.global_step=tf.train.get_or_create_global_step()

        self.loss=None
        self.optimizer=None
        self.minimize=None

    def build_graph(self):
        import tensorflow as tf

        with tf.variable_scope('model'):
            dense=tf.layers.dense(inputs=self.input_layer,
                    units=8,
                    activation=tf.nn.tanh)

            dense2=tf.layers.dense(inputs=dense,
                    units=8,
                    activation=tf.nn.tanh)

            self.output_layer=tf.layers.dense(inputs=dense2,units=2)

    def define_loss(self,output_target):
        import tensorflow as tf
        self.loss=tf.reduce_mean(
                tf.nn.l2_loss(self.output_layer-output_target)
                )

    def define_optimizer(self):
        import tensorflow as tf
        self.optimizer=tf.train.AdamOptimizer(1e-2)

    def define_minimize(self):
        import tensorflow as tf
        self.minimize=self.optimizer.minimize(self.loss,global_step=self.global_step)
