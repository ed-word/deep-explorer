import tensorflow as tf

# MODEL NAME
MODEL_NAME = 'FDFWD_TN'
hparams = {}


class FDFWD_TN:
    def __init__(self, next_element, hyperparam):
        global hparams
        hparams = hyperparam
        self.global_x = next_element['time_series_features']['global_view']
        self.local_x = next_element['time_series_features']['local_view']
        if 'labels' in next_element:
            self.y = next_element['labels']['one_hot']
            self.y_cls = next_element['labels']['class']
        # 2001, 201
        self.global_size = int(self.global_x.shape[1])
        self.local_size = int(self.local_x.shape[1])

        self.global_time_steps = int(
            self.global_size / hparams['global_num_steps_per_tstep'])
        self.local_time_steps = int(
            self.local_size / hparams['local_num_steps_per_tstep'])

    def create_graph(self):
        if hparams['include_global']:
            with tf.variable_scope('global_view'):
                with tf.variable_scope('time_layers'):
                    self.g_layers = []
                    dynamic_batchsize = tf.shape(self.global_x)[0]
                    self.prev_out = tf.zeros(
                        [dynamic_batchsize, hparams['global_num_out_per_tsetp']], tf.float32
                    )
                    for g in range(self.global_time_steps):
                        curr_x = tf.slice(
                                    self.global_x,
                                    [0, g*hparams['global_num_steps_per_tstep']],
                                    [-1, hparams['global_num_steps_per_tstep']]
                                )
                        curr_input = tf.concat([self.prev_out, curr_x], axis=1)

                        layer = tf.layers.dense(
                            curr_input,
                            hparams['global_num_out_per_tsetp'],
                            activation=tf.nn.relu,
                            kernel_initializer=tf.initializers.he_uniform(),
                            name='g_net' + str(g)
                        )
                        self.g_layers.append(layer)
                        if hparams['overlap']:
                            ovlap = tf.slice(
                                    self.global_x,
                                    [0, (g+1)*hparams['global_num_steps_per_tstep'] - hparams['global_num_overlap_prevstep']],
                                    [-1, hparams['global_num_overlap_prevstep']]
                                )
                            self.prev_out = tf.concat([layer, ovlap], axis=1)
                        else:
                            self.prev_out = layer
                    self.g_net = layer
                with tf.variable_scope('fc'):
                    self.g_net = tf.layers.dense(
                        self.g_net,
                        hparams['time_layer_out_num'],
                        activation=tf.nn.relu,
                        kernel_initializer=tf.initializers.he_uniform(),
                        name='fc')

        with tf.variable_scope('local_view'):
            with tf.variable_scope('time_layers'):
                self.l_layers = []
                dynamic_batchsize = tf.shape(self.local_x)[0]
                self.prev_out = tf.truncated_normal(
                    [dynamic_batchsize, hparams['local_num_out_per_tsetp']],
                    dtype=tf.float32
                )
                for l in range(self.local_time_steps):
                    curr_x = tf.slice(
                                self.local_x,
                                [0, l*hparams['local_num_steps_per_tstep']],
                                [-1, hparams['local_num_steps_per_tstep']]
                            )
                    curr_input = tf.concat([self.prev_out, curr_x], axis=1)

                    layer = tf.layers.dense(
                        curr_input,
                        hparams['local_num_out_per_tsetp'],
                        activation=tf.nn.relu,
                        kernel_initializer=tf.initializers.he_uniform(),
                        name='l_net' + str(l)
                    )
                    self.l_layers.append(layer)
                    if hparams['overlap']:
                        ovlap = tf.slice(
                                self.local_x,
                                [0, (l+1)*hparams['local_num_steps_per_tstep'] - hparams['local_num_overlap_prevstep']],
                                [-1, hparams['local_num_overlap_prevstep']]
                            )
                        self.prev_out = tf.concat([layer, ovlap], axis=1)
                    else:
                        self.prev_out = layer
                self.l_net = layer
            with tf.variable_scope('fc'):
                    self.l_net = tf.layers.dense(
                        self.l_net,
                        hparams['time_layer_out_num'],
                        activation=tf.nn.relu,
                        kernel_initializer=tf.initializers.he_uniform(),
                        name='fc')

        if hparams['include_global']:
            with tf.variable_scope('concat'):
                self.concat_tensor = tf.concat(
                    [self.g_net, self.l_net], axis=1, name='concat_tensor')
        else:
            self.concat_tensor = self.l_net
        with tf.variable_scope('fc'):
            self.net = self.concat_tensor
            for l in range(hparams['num_fc_layers']):
                self.net = tf.layers.dense(
                    self.net,
                    self.net.shape[1],
                    activation=tf.nn.relu,
                    kernel_initializer=tf.initializers.he_uniform(),
                    name='fc-' + str(l))
        with tf.variable_scope('logits'):
            self.logits = tf.layers.dense(
                self.net,
                hparams['num_classes'],
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
                hparams['learning_rate']).minimize(self.loss, name='optimizer')
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

    def train(self, train_init_op, val_init_op=None, print_stuff=True):
        print('\nTraining')
        self.sess.run(train_init_op)
        self.sess.run(tf.local_variables_initializer())

        # for i in range(len(self.l_layers)):
        #     print("i: ", i, " ", self.sess.run(self.l_layers[i]))

        for epoch in range(hparams['num_epochs']):
            if print_stuff:
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
            if print_stuff:
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
            if print_stuff:
                print('Model Saved')
        if print_stuff:
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

        # print("Predictions: ", predictions)
        if labels:
            self.test_accuracy = acc
            print("Testing Accuracy: ", self.test_accuracy)
        print(self.test_accuracy)

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


if __name__ == '__main__':
    exit
