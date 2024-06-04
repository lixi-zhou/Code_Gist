from deduplicator import analyse_model_weights
from deduplicator import analyse_models, analyse_models_v2, analyse_models_v2_and_dedup
from deduplicator import analyse_weights

import tensorflow as tf
import numpy as np
import model_quantization
import itertools
from tqdm import tqdm
import pandas as pd
import os


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


from tensorflow.keras.datasets import mnist
# Generate the train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
 
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
 
# Generate some noisy data
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
 
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

model_names = [
    '30k_normal_added_10k_mix-45',
    '30k_normal-45',
    'based_model-45',
    'mnist_and_30k-45',
]
model_dedup_comb = list(itertools.combinations(model_names, 2))

model_dedup_comb = [
    ('30k_normal_added_10k_mix-45','based_model-45'),
    ('30k_normal-45','mnist_and_30k-45')
]

with tf.device('/cpu:0'):

    DATA_DRIFT_MODELS_PATH = "models_modified/"

    list_model_a = []
    list_model_b = []
    list_fp = []
    list_sim = []
    list_block_size = []
    list_total_blocks = []
    list_unique_blocks = []
    list_reduced_blocks = []
    list_removed_params = []
    list_padded_duplicate_blocks = []
    list_padded_unique_blocks = []
    list_m1_acc = []
    list_m2_acc = []
    list_m1_dedup_acc = []
    list_m2_dedup_acc = []
    block_size = 20
    weight_lower_bound = 0.1
    fps = [20]
    sims = [.7,.8,.9]

    for m1_name, m2_name in tqdm(model_dedup_comb):
        m1 = tf.keras.models.load_model(DATA_DRIFT_MODELS_PATH + m1_name)
        m2 = tf.keras.models.load_model(DATA_DRIFT_MODELS_PATH + m2_name)

        m1, m1_quant_data = model_quantization.quant_model(m1)
        m2, m2_quant_data = model_quantization.quant_model(m2)

        m1._name = m1_name
        m2._name = m2_name
        result = analyse_models_v2_and_dedup(
            m1, m2,
            {
                'pairwise': {
                    'fp': fps,  # for different floating point thresholds
                    'sim': sims,  # for naive diff similarity percentage
                },
            },
            block_size,
            block_size,
            weight_lower_bound,
            'model_modifi_dedup/' + m1_name + m2_name
        )
        
        m1 = model_quantization.dequant_model(m1, m1_quant_data)
        m2 = model_quantization.dequant_model(m2, m2_quant_data)

        _, m1_acc = m1.evaluate(x_test, y_test)
        _, m2_acc = m2.evaluate(x_test, y_test)
        
        for fp in fps:
            for sim in sims:
                list_model_a.append(m1_name)
                list_model_b.append(m2_name)
                list_m1_acc.append(m1_acc)
                list_m2_acc.append(m2_acc)
                list_fp.append(fp)
                list_sim.append(sim)
                list_block_size.append(block_size)
                list_total_blocks.append(result['data'][fp][sim]['total_blocks'])
                list_unique_blocks.append(result['data'][fp][sim]['num_unique'])
                list_reduced_blocks.append(result['data'][fp][sim]['num_reduced'])
                list_removed_params.append(result['data'][fp][sim]['removed_params'])
                list_padded_duplicate_blocks.append(result['data'][fp][sim]['num_padded_duplicate_blocks'])
                list_padded_unique_blocks.append(result['data'][fp][sim]['num_padded_unique_blocks'])
                model_deduplicate_path = os.path.join('model_modifi_dedup', m1_name + m2_name, 'models')
                model_1_deduplicate_path = os.path.join(
                    model_deduplicate_path, 
                    m1_name+'_'+str(fp)+'_'+str(sim)+'_'+str(weight_lower_bound)+'_'+str(block_size)+'_'+str(block_size))
                model_2_deduplicate_path = os.path.join(
                    model_deduplicate_path, 
                    m2_name+'_'+str(fp)+'_'+str(sim)+'_'+str(weight_lower_bound)+'_'+str(block_size)+'_'+str(block_size))
                
                m1_dedup = tf.keras.models.load_model(model_1_deduplicate_path)
                m2_dedup = tf.keras.models.load_model(model_2_deduplicate_path)
                m1_dedup = model_quantization.dequant_model(m1_dedup, m1_quant_data)
                m2_dedup = model_quantization.dequant_model(m2_dedup, m2_quant_data)
                _, m1_dedup_acc = m1_dedup.evaluate(x_test, y_test)
                _, m2_dedup_acc = m2_dedup.evaluate(x_test, y_test)
                list_m1_dedup_acc.append(m1_dedup_acc)
                list_m2_dedup_acc.append(m2_dedup_acc)
    
    ori_dedup_result_df = pd.DataFrame({'Model A': list_model_a, 'Model B': list_model_b, 
              'Model A Acc': list_m1_acc, 'Model B Acc': list_m2_acc,
              'FP': list_fp, 'Sim': list_sim, 'Block Size': list_block_size,
              'Total Blocks': list_total_blocks, 'Unique Blocks': list_unique_blocks,
              'Reduced Blocks': list_reduced_blocks, 'Removed Params': list_removed_params,
              'Padded Dup Blocks': list_padded_duplicate_blocks, 'Padded Unique Blocks': list_padded_unique_blocks,
             'Model A Dedup Acc': list_m1_dedup_acc, 'Model B Dedup Acc': list_m2_dedup_acc})
    ori_dedup_result_df.to_csv('quant_new_model_modify_dedup_' + str(fp) + '_' + str(block_size) + '_' +  str(weight_lower_bound) + '_mb.csv', index=False)

# with tf.device('/cpu:0'):

#     DATA_DRIFT_MODELS_PATH = "vgg_model_pruned/"

#     list_model_a = []
#     list_model_b = []
#     list_fp = []
#     list_sim = []
#     list_block_size = []
#     list_dedup_blocks = []
#     list_dedup_bytes = []
#     list_m1_acc = []
#     list_m2_acc = []
#     list_m1_dedup_acc = []
#     list_m2_dedup_acc = []
#     block_size = 200
#     weight_lower_bound = 0.1

#     for m1_name, m2_name in tqdm([('vgg16', 'vgg19')]):
#         m1 = tf.keras.models.load_model('vgg_model_pruned/vgg16_cifar10_0.5-0.8.h5')
#         m2 = tf.keras.models.load_model('vgg_model_pruned/vgg19_cifar10_0.5_0.8.h5')

#         m1._name = m1_name
#         m2._name = m2_name
#         result = analyse_models_v2_and_dedup(
#             m1, m2,
#             {
#                 'pairwise': {
#                     'fp': [0.01],  # for different floating point thresholds
#                     'sim': [.7, .8, .9],  # for naive diff similarity percentage
#                 },
#             },
#             block_size,
#             block_size,
#             weight_lower_bound,
#             'vgg_pruned_deduplicate/' + m1_name + m2_name
#         )
        
#         _, m1_acc = m1.evaluate(x_test, y_test)
#         _, m2_acc = m2.evaluate(x_test, y_test)
        
#         for fp in result.keys():
#             for sim in result[fp].keys():
#                 list_model_a.append(m1_name)
#                 list_model_b.append(m2_name)
#                 list_m1_acc.append(m1_acc)
#                 list_m2_acc.append(m2_acc)
#                 list_fp.append(fp)
#                 list_sim.append(sim)
#                 list_block_size.append(block_size)
#                 list_dedup_blocks.append(str(result[fp][sim]['num_reduced']) + '/' + str(result[fp][sim]['total_blocks']))
#                 list_dedup_bytes.append(str(result[fp][sim]['bytes_reduced']) + '/' + str(result[fp][sim]['total_bytes']))
#                 model_deduplicate_path = os.path.join('vgg_pruned_deduplicate', m1_name + m2_name, 'models')
#                 model_1_deduplicate_path = os.path.join(
#                     model_deduplicate_path, 
#                     m1_name+'_'+str(fp)+'_'+str(sim)+'_'+str(weight_lower_bound)+'_'+str(block_size)+'_'+str(block_size))
#                 model_2_deduplicate_path = os.path.join(
#                     model_deduplicate_path, 
#                     m2_name+'_'+str(fp)+'_'+str(sim)+'_'+str(weight_lower_bound)+'_'+str(block_size)+'_'+str(block_size))
                
#                 m1_dedup = tf.keras.models.load_model(model_1_deduplicate_path)
#                 m2_dedup = tf.keras.models.load_model(model_2_deduplicate_path)
#                 _, m1_dedup_acc = m1_dedup.evaluate(x_test, y_test)
#                 _, m2_dedup_acc = m2_dedup.evaluate(x_test, y_test)
#                 list_m1_dedup_acc.append(m1_dedup_acc)
#                 list_m2_dedup_acc.append(m2_dedup_acc)
    
#     ori_dedup_result_df = pd.DataFrame({'Model A': list_model_a, 'Model B': list_model_b, 
#               'Model A Acc': list_m1_acc, 'Model B Acc': list_m2_acc,
#               'FP': list_fp, 'Sim': list_sim, 'Block Size': list_block_size,
#               'Deduplicate Blocks': list_dedup_blocks, 'Deduplicate Bytes': list_dedup_bytes,
#              'Model A Dedup Acc': list_m1_dedup_acc, 'Model B Dedup Acc': list_m2_dedup_acc})
#     ori_dedup_result_df.to_csv('vgg_model_pruned_dedup_20_0.1_mb.csv', index=False)