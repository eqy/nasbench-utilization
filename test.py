import os
import shutil
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import pandas as pd
import seaborn as sns
import tensorflow as tf

from nasbench import api
from nasbench.lib import model_spec, model_builder, config, cifar

nb = api.NASBench('nasbench_only108.tfrecord')
data = {'params':[], 'time':[], 'acc': [], 'flops': []}

cfg = config.build_config()
cfg['train_data_files'] = ['nasbench/scripts/train_1.tfrecords',
                           'nasbench/scripts/train_2.tfrecords',
                           'nasbench/scripts/train_3.tfrecords',
                           'nasbench/scripts/train_4.tfrecords']
cfg['use_tpu'] = False

keys = list()
count = 0
for key in nb.hash_iterator():
    keys.append(key)
    count += 1

def keytotuple(key):
    cur_network_data = nb.get_metrics_from_hash(key)
    #print(cur_network_data[0])
    #print(cur_network_data[0].keys())
    model = model_spec.ModelSpec(cur_network_data[0]['module_adjacency'],
                                 cur_network_data[0]['module_operations'])
    model_fn = model_builder.build_model_fn(model, cfg, 60000)
    if os.path.exists('empty'):
        shutil.rmtree('empty')
    run_cfg = tf.contrib.tpu.RunConfig(
      model_dir='empty',
      keep_checkpoint_max=3,    # Keeps ckpt at start, halfway, and end
      save_checkpoints_secs=2**30)
      #tpu_config=tf.contrib.tpu.TPUConfig(
      #    iterations_per_loop=cfg['tpu_iterations_per_loop'],
      #    num_shards=cfg['tpu_num_shards']))
    #estimator = tf.contrib.tpu.TPUEstimator(model_fn, config=run_cfg,
    #                                       train_batch_size=cfg['batch_size'],
    #                                       eval_batch_size=cfg['batch_size'],
    #                                       predict_batch_size=cfg['batch_size'],
    #                                       use_tpu=False)#, params=cfg)
    estimator = tf.estimator.Estimator(model_fn, config=run_cfg, params=cfg)
    print(estimator)
    #dummy_input = np.zeros((1, 224, 224, 3))
    #dummy_label = np.zeros((1, 100))
    #dummy_label[0] = 1
    input_train = cifar.CIFARInput('train', cfg)
    print(cfg['batch_size'])

    #input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": dummy_input}, y=dummy_label, shuffle=True)
    #estimator.train(input_fn)
    #estimator.train(input_fn=input_train.input_fn,
    #                max_steps=1)
    #print(tf.get_default_graph().as_graph_def())

    with tf.Graph().as_default() as g:
        features = tf.placeholder(tf.float32, [cfg['batch_size'], 32, 32, 3])
        labels = tf.placeholder(tf.int32, [cfg['batch_size']])
        _ = model_fn(features, labels, mode=tf.estimator.ModeKeys.TRAIN, params=cfg)
        with tf.Session() as sess:
            run_meta = tf.RunMetadata()
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
            n_flops = flops.total_float_ops
            print(n_flops)
            #print(sess.graph.as_graph_def())

    training_time_sum = 0.0
    acc_sum = 0.0
    params = cur_network_data[0]['trainable_parameters']
    count = 0
    for item in cur_network_data[1][108]:
        count += 1
        training_time_sum += item['final_training_time']
        acc_sum += item['final_test_accuracy']
    training_time = training_time_sum/count
    acc = acc_sum/count

    return (params, training_time, acc, n_flops)

    #data['params'].append(params)
    #data['time'].append(training_time)
    #data['acc'].append(acc)
    #data['flops'].append(n_flops)

p = Pool(64)
results = p.map(keytotuple, keys)

df = pd.DataFrame(data)
df.to_pickle('nasbench_data2.df')
plt.figure()
sns.scatterplot(x='params', y='time', data=df, alpha=0.1)
plt.savefig('timevsparams.pdf')
