import tensorflow as tf

TFRECORD_DIR = 'tfrecord/'


def parser(serialized_example, label_to_id, input_config):
    data_fields = {
        feature_name: tf.FixedLenFeature([feature['length']], tf.float32)
        for feature_name, feature in input_config['features'].items()
    }
    data_fields[input_config['label_feature']] = tf.FixedLenFeature(
                                                    [], tf.string)

    parsed_features = tf.parse_single_example(
        serialized_example, features=data_fields)

    output = {}
    for feature_name, value in parsed_features.items():
        if feature_name == input_config['label_feature']:
            if 'labels' not in output:
                output['labels'] = {}
            label_id = label_to_id.lookup(value)
            output['labels']['class'] = label_id
            output['labels']['one_hot'] = tf.one_hot(label_id, depth=2)
        elif input_config['features'][feature_name]['is_time_series']:
            if "time_series_features" not in output:
                output["time_series_features"] = {}
            output["time_series_features"][feature_name] = value
        else:
            if "aux_features" not in output:
                output["aux_features"] = {}
            output["aux_features"][feature_name] = value

    return output


def input_fn(filepath, input_config, batch_size):
    table_initializer = tf.contrib.lookup.KeyValueTensorInitializer(
        keys=list(input_config['label_map'].keys()),
        values=list(input_config['label_map'].values()),
        key_dtype=tf.string,
        value_dtype=tf.int64)
    label_to_id = tf.contrib.lookup.HashTable(
        table_initializer, default_value=-1)

    filepath = TFRECORD_DIR + filepath + '*'
    filenames = tf.gfile.Glob(filepath)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.shuffle(len(filenames))
    # dataset = dataset.repeat()
    dataset = dataset.map(
                lambda x: parser(x, label_to_id, input_config),
                num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    return dataset


if __name__ == '__main__':
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
    batch_size = 5
    with tf.device('/cpu:0'):
        dataset = input_fn('train', input_config, batch_size)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

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

    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        sess.run(iterator.initializer)
        x = sess.run(next_element)
        print(x["labels"])
        # epoch_size = 0
        # while True:
        #     try:
        #         x = sess.run(next_element)
        #         # print("Batch size: ", len(x['labels']))
        #         epoch_size += len(x["labels"])
        #     except tf.errors.OutOfRangeError:
        #         print("Epoch size: ", epoch_size)
        #         break
