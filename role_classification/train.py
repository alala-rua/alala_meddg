import json
import datetime,time
import os
import shutil
import tensorflow as tf

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

# label_prior = np.array([2.92228585e-04, 5.40078840e-01, 4.04145915e-05, 4.59588517e-01])


import pickle

with open(os.path.join(data_path, "train_data.pk"), "rb") as f:
    train_data = pickle.load(f)
    
with open(os.path.join(data_path, "dev_data.pk"), "rb") as f:
    dev_data = pickle.load(f)


def create_classification_data(data_item):
    
    data_labs = []

    for sentence in data_item:
        data_labs.append([sentence['Sentence'], sentence['id']])

    return data_labs


train_label_data = []

for data_item in train_data:
    train_label_data.extend(create_classification_data(data_item))
    
dev_label_data = []

for data_item in dev_data:
    dev_label_data.extend(create_classification_data(data_item))


print(len(train_label_data))
print(len(dev_label_data))


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        
        batch_token_ids, batch_segment_ids, batch_label = [], [], []
        
        for is_end, data_item in self.sample(random):
            
            content, label = data_item
                
            token_ids, _ = tokenizer.encode(content)
            token_ids = token_ids[1:-1]

            if len(token_ids) > maxlen - 2:
                token_ids = token_ids[-maxlen+2:]

            token_ids = [tokenizer.token_to_id("[CLS]")] + token_ids + [tokenizer.token_to_id("[SEP]")]
            token_ids += [0] * (maxlen - len(token_ids))

            batch_token_ids.append(token_ids)
            batch_segment_ids.append([0] * maxlen)
            batch_label.append(c2i[label])

            if len(batch_token_ids) == self.batch_size or is_end:

                yield {
                    'Input-Token': np.array(batch_token_ids),
                    'Input-Segment': np.array(batch_segment_ids),
                },{
                    'Type-Dense-Softmax': np.array(batch_label)[:,np.newaxis]
                }

                batch_token_ids, batch_segment_ids, batch_label = [], [], []


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


train_model.summary()

# 模型验证
def predict(data):
    
    batch_token_ids, batch_segment_ids, batch_label = [], [], []

    for content in data:

        token_ids, _ = tokenizer.encode(content)
        token_ids = token_ids[1:-1]

        if len(token_ids) > maxlen - 2:
            token_ids = token_ids[-maxlen+2:]

        token_ids = [tokenizer.token_to_id("[CLS]")] + token_ids + [tokenizer.token_to_id("[SEP]")]
        token_ids += [0] * (maxlen - len(token_ids))

        batch_token_ids.append(token_ids)
        batch_segment_ids.append([0] * maxlen)

    batch_token_ids = np.array(batch_token_ids)
    batch_segment_ids = np.array(batch_segment_ids)
    
    y_pred = model.predict([batch_token_ids, batch_segment_ids], batch_size=128, verbose=1)
    
    return [i2c[item.argmax()] for item in y_pred]


from sklearn.metrics import classification_report, accuracy_score

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self, model_saved_path):
        self.model_saved_path = model_saved_path
        self.best_metrics = 0

    def on_epoch_end(self, epoch, logs=None):
        metrics = self.evaluate(dev_label_data)  # 评测模型
        
        if metrics > self.best_metrics:
            self.best_metrics = metrics
            model.save_weights(os.path.join(self.model_saved_path, "best_weights"), overwrite=True)  # 保存模型


    def evaluate(self, data):
        
        texts = []
        label_true = []
        
        for s,label in data:
            texts.append(s)
            label_true.append(label)

        label_pred = predict(texts)

        print(classification_report(label_true, label_pred, digits=4))
        acc = accuracy_score(label_true, label_pred)
        return acc
        
        

print(projectName + ' Train...')
now = time.strftime("%Y-%m-%d_%H-%M-%S")
resultPath = './param/outputModelWeights/{}/'.format(projectName)
if not os.path.exists(resultPath):
    os.makedirs(resultPath)

train_generator = data_generator(train_label_data, batch_size)

train_model.fit(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    callbacks=[Evaluator(resultPath)]
)