import tensorflow as tf
from dataset import input_fn

# MODEL NAME
MODEL_NAME = 'FCN'

# MODEL HYPERPARAMETERS
learning_rate = 0.00015
n_hidden_neurons = []
num_hidden_layers = len(n_hidden_neurons)
num_classes = 2

# TRAINING HYPERPARAMETRS
batch_size = 1024
num_epochs = 10
input_config = {
    'label_feature': 'av_training_set',
    'label_map': {'SCR1': 0, 'PC': 1, 'NTP': 0, 'INV': 0, 'AFP': 0, 'INJ1': 1},
    'features': {
      'global_view': {
        'length': 2001,
        'is_time_series': True
        },
      'local_view': {
        'length': 201,
        'is_time_series': True
        }
      }
    }

# SESSION CONF
cpu_session = tf.ConfigProto(
    device_count={'CPU': 1, 'GPU': 0},
    allow_soft_placement=False,
    log_device_placement=False
)
gpu_session = tf.ConfigProto(
    device_count={'CPU': 1, 'GPU': 1},
    allow_soft_placement=False,
    log_device_placement=False
)


class FCNetwork:
    def __init__(self, next_element):
        self.global_x = next_element['time_series_features']['global_view']
        self.local_x = next_element['time_series_features']['local_view']
        if 'labels' in next_element:
            self.y = next_element['labels']['one_hot']
            self.y_cls = next_element['labels']['class']

    def create_graph(self):
        with tf.variable_scope('global_view'):
            with tf.variable_scope('layers'):
                self.g_net = self.global_x
                for l in range(num_hidden_layers):
                    self.g_net = tf.layers.dense(
                        self.g_net,
                        n_hidden_neurons[l],
                        activation=tf.nn.relu,
                        kernel_initializer=tf.initializers.he_uniform(),
                        name='g_net-' + str(l)
                    )
        with tf.variable_scope('local_view'):
            with tf.variable_scope('layers'):
                self.l_net = self.local_x
                for l in range(num_hidden_layers):
                    self.l_net = tf.layers.dense(
                        self.l_net,
                        n_hidden_neurons[l],
                        activation=tf.nn.relu,
                        kernel_initializer=tf.initializers.he_uniform(),
                        name='l_net-' + str(l)
                    )
        with tf.variable_scope('concat'):
            self.concat_tensor = tf.concat(
                [self.g_net, self.l_net], axis=1, name='concat_tensor')
        with tf.variable_scope('fc'):
            self.net = tf.layers.dense(
                self.concat_tensor,
                self.concat_tensor.shape[1],
                activation=tf.nn.relu,
                kernel_initializer=tf.initializers.he_uniform(),
                name='fc')
        with tf.variable_scope('logits'):
            self.logits = tf.layers.dense(
                self.net,
                num_classes,
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='logits')
        with tf.variable_scope('predict'):
            self.y_pred = tf.nn.softmax(self.logits, name='y_pred')
            self.y_pred_cls = tf.argmax(self.y_pred, axis=1, name='y_pred_cls')
        with tf.variable_scope('optimizations'):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.y, logits=self.logits, name='cross_entropy')
            self.loss = tf.reduce_mean(self.cross_entropy, name='loss')
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate).minimize(self.loss, name='optimizer')
        with tf.variable_scope('metrics'):
            self.accuracy, self.acc_update_op = tf.metrics.accuracy(
                labels=self.y_cls, predictions=self.y_pred_cls, name='accuracy')

    def create_sess(self):
        sess = tf.Session()
        sess.run(tf.tables_initializer())
        init = tf.global_variables_initializer()
        sess.run(init)
        self.sess = sess
        self.create_summary()

    def train(self, train_init_op, val_init_op=None):
        print('\nTraining')
        for epoch in range(num_epochs):
            print('-' * 58)
            print("\nEpoch: ", epoch)
            # Training
            self.sess.run(train_init_op)
            self.sess.run(tf.local_variables_initializer())
            batch_num = 0
            while True:
                try:
                    summary, _, acc, loss = self.sess.run(
                        [self.merged, self.optimizer, self.acc_update_op, self.loss])
                    # print(
                    #     "Batch", batch_num,
                    #     "\tAccuracy: ", acc,
                    #     "\tLoss: ", loss
                    # )
                    batch_num += 1
                except tf.errors.OutOfRangeError:
                    break
            # Train Summary
            self.train_summary_writer.add_summary(summary, epoch)
            # total, count = self.sess.run(tf.local_variables())
            # self.train_accuracy = total / count
            self.train_accuracy = acc

            if val_init_op is not None:
                # Validation
                self.sess.run(val_init_op)
                self.sess.run(tf.local_variables_initializer())
                batch_num = 0
                self.val_accuracy = 0
                while True:
                    try:
                        summary, acc = self.sess.run([self.merged, self.acc_update_op])
                        batch_num += 1
                    except tf.errors.OutOfRangeError:
                        break
                # Val Summary
                self.val_summary_writer.add_summary(summary, epoch)
                # total, count = self.sess.run(tf.local_variables())
                # self.val_accuracy = total / count
                self.val_accuracy = acc

            # Epoch results
            print('-' * 58)
            print("Training Accuracy: ", self.train_accuracy)
            if val_init_op is not None:
                print("Validation Accuracy: ", self.val_accuracy)

            # Save latest Checkpoint in 2 formats
            # model_epoch.ckpt and overwrite model.ckpt
            self.saver.save(
                self.sess,
                './models/' + MODEL_NAME + '/model_' + str(epoch) + '.ckpt')
            self.saver.save(
                self.sess,
                './models/' + MODEL_NAME + '/model.ckpt')
            print('Model Saved')
        print('-' * 58)
        self.sess.close()

    def test(self, test_init_op, labels=False):
        print('\nTesting')
        self.create_sess()
        self.saver.restore(
            self.sess,
            './models/' + MODEL_NAME + '/model.ckpt')

        # Testing
        self.sess.run(test_init_op)
        self.sess.run(tf.local_variables_initializer())
        batch_num = 0
        self.test_accuracy = 0
        predictions = []
        while True:
            try:
                if labels:
                    predictions, acc = self.sess.run(
                        [self.y_pred_cls, self.acc_update_op])
                else:
                    pred = self.sess.run([self.y_pred_cls])
                    predictions.extend(pred)
                batch_num += 1
            except tf.errors.OutOfRangeError:
                break

        print("Predictions: ", predictions)
        if labels:
            self.test_accuracy = acc
            print("Testing Accuracy: ", self.test_accuracy)

    def create_summary(self):
        # Model Saver
        self.saver = tf.train.Saver()

        # Summary Writer
        self.train_summary_writer = tf.summary.FileWriter(
            "./models/" + MODEL_NAME + "/log/train", self.sess.graph)
        self.val_summary_writer = tf.summary.FileWriter(
            "./models/" + MODEL_NAME + "/log/val")
        tf.summary.scalar('summary_loss', self.loss)
        tf.summary.scalar('summary_acc', self.accuracy)
        self.merged = tf.summary.merge_all()


def dataset():
    with tf.device('/cpu:0'):
        with tf.variable_scope('Dataset'):
            train_dataset = input_fn('train', input_config, batch_size)
            val_dataset = input_fn('val', input_config, batch_size)
            test_dataset = input_fn('test', input_config, batch_size)

            iterator = tf.data.Iterator.from_structure(
                train_dataset.output_types,
                train_dataset.output_shapes)
            next_element = iterator.get_next()

            train_init_op = iterator.make_initializer(train_dataset)
            val_init_op = iterator.make_initializer(val_dataset)
            test_init_op = iterator.make_initializer(test_dataset)
            return train_init_op, val_init_op, test_init_op, next_element


if __name__ == '__main__':
    train_init_op, val_init_op, test_init_op, next_element = dataset()
    fc_network = FCNetwork(next_element)
    fc_network.create_graph()
    fc_network.create_sess()
    fc_network.train(train_init_op, val_init_op)
    fc_network.test(test_init_op, True)
