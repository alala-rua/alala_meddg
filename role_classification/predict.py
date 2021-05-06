import json
import datetime,time
import os
import shutil
import tensorflow as tf
os.environ["RECOMPUTE"] = '1'

from bert4keras.backend import keras, K, batch_gather
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from bert4keras.layers import Loss

from keras.layers import Lambda, Dense, Input, Permute, Activation
from keras.models import Model

import numpy as np
from tqdm import tqdm

data_path = "./data/"

projectName = 'ICLR_2021_Workshop_MLPCP_Track_1_医病分类'

config_path = './pretrain_weights/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './pretrain_weights/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './pretrain_weights/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'


token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

maxlen = 256
batch_size = 32
learning_rate = 1e-5
epochs = 20


i2c = [
    'Patients',
    'Doctor'
]

c2i = {item:idx for idx, item in enumerate(i2c)}

data_path = "./data/"


import pickle

with open(os.path.join(data_path, "train_data.pk"), "rb") as f:
    train_data = pickle.load(f)
    
with open(os.path.join(data_path, "dev_data.pk"), "rb") as f:
    dev_data = pickle.load(f)


# 模型

from bert4keras.layers import Layer

class MaskMean(Layer):
    
    def __init__(self, **kwargs):
        super(MaskMean, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(MaskMean, self).build(input_shape)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def compute_mask(self, inputs, mask):
        return None

    def call(self, inputs, mask):

        tokens = inputs
        mask = K.expand_dims(K.cast(mask, dtype="float"), axis=-1)
        
        return K.sum(tokens*mask, axis=1) / K.sum(mask, axis=1)


bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    return_keras_model=False,
)

mean_output = MaskMean(name="Mask-Mean")(bert.model.output)

dense_output = Dense(units=len(i2c), kernel_initializer=bert.initializer, name="Type-Dense-Softmax")(mean_output)

model = Model(bert.model.inputs, dense_output)

train_model = Model(bert.model.inputs, dense_output)

optimizer = Adam(learning_rate=learning_rate)
# train_model.compile(loss=sparse_categorical_crossentropy_with_prior_maker(label_prior), optimizer=optimizer)
train_model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)

# 加载参数
model.load_weights("./param/outputModelWeights/ICLR_2021_Workshop_MLPCP_Track_1_医病分类/best_weights")

# 统计转移矩阵
trans = np.zeros((len(c2i), len(c2i)))

for item in train_data+dev_data:
    for idx in range(1, len(item)):
        trans[c2i[item[idx-1]['id']]][c2i[item[idx]['id']]] += 1

trans /= trans.sum(axis=-1, keepdims=True)

def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def get_best_path(launch, trans):
    n, m = launch.shape
    
    best_path = np.zeros((n-1, m), dtype=np.int32)
    best_score = np.array(launch[0])
    
    for i in range(1, n): 
        
        best_score = best_score[...,None] + trans + launch[i:i+1,:]
        best_path[i-1] = best_score.argmax(axis=0)
        best_score = best_score.max(axis=0)
    
    re_best_path = [best_score.argmax(axis=0)]
    
    for i in range(n-2, -1, -1):
        re_best_path = [best_path[i][re_best_path[0]]] + re_best_path
        
    return re_best_path, best_score.max()
    


def predict_test(data):

    batch_token_ids, batch_segment_ids = [], []
    data_map = []

    for data_item in data:

        data_map_item = []

        for s in data_item['history']:

            content = s

            token_ids, _ = tokenizer.encode(content)
            token_ids = token_ids[1:-1]

            if len(token_ids) > maxlen - 2:
                token_ids = token_ids[-maxlen+2:]

            token_ids = [tokenizer.token_to_id("[CLS]")] + token_ids + [tokenizer.token_to_id("[SEP]")]
            token_ids += [0] * (maxlen - len(token_ids))

            data_map_item.append(len(batch_token_ids))
            batch_token_ids.append(token_ids)
            batch_segment_ids.append([0] * maxlen)

        data_map.append(data_map_item)


    y_pred = model.predict([batch_token_ids, batch_segment_ids], batch_size=128, verbose=1)

    result = []

    for data_map_item in data_map:
        launch_score = np.zeros((len(data_map_item) + 1, 2))

        for idx, d_idx in enumerate(data_map_item):
            launch_score[idx] = y_pred[d_idx]
            
        # 最后一句不可能是病人的
        
        launch_score = np.log(softmax(launch_score, axis=-1))
        
        launch_score[0][c2i['Doctor']] = -1e12
        launch_score[0][c2i['Patients']] = 0
        
        launch_score[-1][c2i['Doctor']] = 0
        launch_score[-1][c2i['Patients']] = -1e9

        best_path, _ = get_best_path(launch_score, np.log(trans))
        
        best_path = best_path[:-1]

        result.append([i2c[item] for item in best_path])

    return result


with open(os.path.join(data_path, "PhasesBTestData.pk"), "rb") as f:
    test_data = pickle.load(f)


test_pre = predict_test(test_data)

test_add_info = []

for data_item, s_labels in zip(test_data, test_pre):
    
    test_add_info_item = []
    
    for s, s_label in zip(data_item['history'], s_labels):
        
        test_add_info_item.append(
            {
                'id': s_label,
                'Sentence': s,
                'Symptom': [],
                'Medicine': [],
                'Test': [],
                'Attribute': [],
                'Disease': []
            }
        )
    
    test_add_info.append(test_add_info_item)

with open(os.path.join(data_path, "test_add_info.pk"), "wb") as f:
    pickle.dump(test_add_info, f)


