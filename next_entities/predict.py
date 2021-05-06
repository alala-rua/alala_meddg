import json
import datetime,time
import os
import shutil
import tensorflow as tf
os.environ["RECOMPUTE"] = '1'

import pickle
from bert4keras.backend import keras, K, batch_gather
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, SpTokenizer
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from bert4keras.layers import Loss

from keras.layers import Lambda, Dense, Input, Permute, Activation
from keras.models import Model

import numpy as np
from tqdm.notebook import tqdm
# from rouge import Rouge  # pip install rouge
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

projectName = 'ICLR_2021_Workshop_MLPCP_Track_1_模板生成填充_文本分类_模型简化2_增加转移概率'

config_path = './pretrain_weights/PCL-MedBERT-wwm/bert_config.json'
checkpoint_path = './pretrain_weights/PCL-MedBERT-wwm/bert_model.ckpt'
dict_path = './pretrain_weights/PCL-MedBERT-wwm/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

maxlen = 512
epochs = 100
steps_per_epoch = 1024
batch_size = 16

pp = 2
skip_epochs = 50

append_entities_len = 40

learning_rate = 1e-5

i2c = [
    ('None', 'None'),
    ('胃痛', 'Symptom'),
    ('肌肉酸痛', 'Symptom'),
    ('咽部痛', 'Symptom'),
    ('胃肠不适', 'Symptom'),
    ('咽部灼烧感', 'Symptom'),
    ('腹胀', 'Symptom'),
    ('稀便', 'Symptom'),
    ('肠梗阻', 'Symptom'),
    ('胸痛', 'Symptom'),
    ('饥饿感', 'Symptom'),
    ('烧心', 'Symptom'),
    ('寒战', 'Symptom'),
    ('气促', 'Symptom'),
    ('嗜睡', 'Symptom'),
    ('粘便', 'Symptom'),
    ('四肢麻木', 'Symptom'),
    ('腹痛', 'Symptom'),
    ('恶心', 'Symptom'),
    ('胃肠功能紊乱', 'Symptom'),
    ('反流', 'Symptom'),
    ('里急后重', 'Symptom'),
    ('鼻塞', 'Symptom'),
    ('体重下降', 'Symptom'),
    ('贫血', 'Symptom'),
    ('发热', 'Symptom'),
    ('过敏', 'Symptom'),
    ('痉挛', 'Symptom'),
    ('黑便', 'Symptom'),
    ('头晕', 'Symptom'),
    ('乏力', 'Symptom'),
    ('心悸', 'Symptom'),
    ('肠鸣', 'Symptom'),
    ('尿急', 'Symptom'),
    ('细菌感染', 'Symptom'),
    ('喷嚏', 'Symptom'),
    ('腹泻', 'Symptom'),
    ('焦躁', 'Symptom'),
    ('痔疮', 'Symptom'),
    ('精神不振', 'Symptom'),
    ('咳嗽', 'Symptom'),
    ('脱水', 'Symptom'),
    ('消化不良', 'Symptom'),
    ('食欲不振', 'Symptom'),
    ('月经紊乱', 'Symptom'),
    ('背痛', 'Symptom'),
    ('呼吸困难', 'Symptom'),
    ('吞咽困难', 'Symptom'),
    ('水肿', 'Symptom'),
    ('肛周疼痛', 'Symptom'),
    ('呕血', 'Symptom'),
    ('菌群失调', 'Symptom'),
    ('便血', 'Symptom'),
    ('口苦', 'Symptom'),
    ('淋巴结肿大', 'Symptom'),
    ('头痛', 'Symptom'),
    ('尿频', 'Symptom'),
    ('排气', 'Symptom'),
    ('黄疸', 'Symptom'),
    ('呕吐', 'Symptom'),
    ('有痰', 'Symptom'),
    ('打嗝', 'Symptom'),
    ('螺旋杆菌感染', 'Symptom'),
    ('胃复安', 'Medicine'),
    ('泮托拉唑', 'Medicine'),
    ('马来酸曲美布丁', 'Medicine'),
    ('磷酸铝', 'Medicine'),
    ('诺氟沙星', 'Medicine'),
    ('金双歧', 'Medicine'),
    ('人参健脾丸', 'Medicine'),
    ('三九胃泰', 'Medicine'),
    ('泌特', 'Medicine'),
    ('康复新液', 'Medicine'),
    ('克拉霉素', 'Medicine'),
    ('乳果糖', 'Medicine'),
    ('奥美', 'Medicine'),
    ('果胶铋', 'Medicine'),
    ('嗜酸乳杆菌', 'Medicine'),
    ('谷氨酰胺肠溶胶囊', 'Medicine'),
    ('四磨汤', 'Medicine'),
    ('思连康', 'Medicine'),
    ('多潘立酮', 'Medicine'),
    ('得舒特', 'Medicine'),
    ('肠溶胶囊', 'Medicine'),
    ('胃苏', 'Medicine'),
    ('蒙脱石散', 'Medicine'),
    ('益生菌', 'Medicine'),
    ('藿香正气丸', 'Medicine'),
    ('诺氟沙星胶囊', 'Medicine'),
    ('复方消化酶', 'Medicine'),
    ('布洛芬', 'Medicine'),
    ('硫糖铝', 'Medicine'),
    ('乳酸菌素', 'Medicine'),
    ('雷呗', 'Medicine'),
    ('莫沙必利', 'Medicine'),
    ('补脾益肠丸', 'Medicine'),
    ('香砂养胃丸', 'Medicine'),
    ('铝碳酸镁', 'Medicine'),
    ('马来酸曲美布汀', 'Medicine'),
    ('消炎利胆片', 'Medicine'),
    ('多酶片', 'Medicine'),
    ('思密达', 'Medicine'),
    ('阿莫西林', 'Medicine'),
    ('颠茄片', 'Medicine'),
    ('耐信', 'Medicine'),
    ('瑞巴派特', 'Medicine'),
    ('培菲康', 'Medicine'),
    ('吗叮咛', 'Medicine'),
    ('曲美布汀', 'Medicine'),
    ('甲硝唑', 'Medicine'),
    ('胶体果胶铋', 'Medicine'),
    ('吗丁啉', 'Medicine'),
    ('健胃消食片', 'Medicine'),
    ('兰索拉唑', 'Medicine'),
    ('马来酸曲美布汀片', 'Medicine'),
    ('莫沙比利', 'Medicine'),
    ('左氧氟沙星', 'Medicine'),
    ('斯达舒', 'Medicine'),
    ('抗生素', 'Medicine'),
    ('达喜', 'Medicine'),
    ('山莨菪碱', 'Medicine'),
    ('健脾丸', 'Medicine'),
    ('肠胃康', 'Medicine'),
    ('整肠生', 'Medicine'),
    ('开塞露', 'Medicine'),
    ('腹腔镜', 'Test'),
    ('小肠镜', 'Test'),
    ('糖尿病', 'Test'),
    ('CT', 'Test'),
    ('B超', 'Test'),
    ('呼气实验', 'Test'),
    ('肛门镜', 'Test'),
    ('便常规', 'Test'),
    ('尿检', 'Test'),
    ('钡餐', 'Test'),
    ('转氨酶', 'Test'),
    ('尿常规', 'Test'),
    ('胶囊内镜', 'Test'),
    ('肝胆胰脾超声', 'Test'),
    ('胃镜', 'Test'),
    ('结肠镜', 'Test'),
    ('腹部彩超', 'Test'),
    ('胃蛋白酶', 'Test'),
    ('血常规', 'Test'),
    ('肠镜', 'Test'),
    ('性质', 'Attribute'),
    ('诱因', 'Attribute'),
    ('时长', 'Attribute'),
    ('位置', 'Attribute'),
    ('胰腺炎', 'Disease'),
    ('肠炎', 'Disease'),
    ('肝硬化', 'Disease'),
    ('阑尾炎', 'Disease'),
    ('肺炎', 'Disease'),
    ('食管炎', 'Disease'),
    ('便秘', 'Disease'),
    ('胃炎', 'Disease'),
    ('感冒', 'Disease'),
    ('胆囊炎', 'Disease'),
    ('胃溃疡', 'Disease'),
    ('肠易激综合征', 'Disease')
]

c2i = { v:idx  for idx, v in enumerate(i2c) }


data_path = "./data/"


from bert4keras.layers import Layer, Embedding, Add

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


# 手动定义loss, 以及acc

# y_true 0 mask 1 正例 2 负例
def mult_circle_loss(inputs, mask=None):
    
    y_true, y_pred = inputs
    zeros = K.zeros_like(y_pred[..., :1])
    
    y_true_p = K.cast(K.equal(y_true, 1), K.floatx())
    y_true_n = K.cast(K.equal(y_true, 2), K.floatx())
    
    y_pred_p = -y_pred + (1 - y_true_p) * -1e12
    y_pred_n = y_pred + (1 - y_true_n) * -1e12

    y_pred_p = K.concatenate([y_pred_p, zeros], axis=-1)
    y_pred_n = K.concatenate([y_pred_n, zeros], axis=-1)

    p_loss = tf.reduce_logsumexp(y_pred_p, axis=-1)
    n_loss = tf.reduce_logsumexp(y_pred_n, axis=-1)
    
    return pp * p_loss + n_loss

bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
)

mean_output = MaskMean(name="Mask-Mean")(bert.model.output)
    
final_output = Dense(
    units=int(mean_output.shape[-1]), 
    kernel_initializer=bert.initializer, 
    activation="tanh",
    name="Label-Tanh"
)(mean_output)

final_output = Dense(
    units=len(i2c), 
    kernel_initializer=bert.initializer, 
    name="Label-Id"
)(final_output)

# 增加转移概率

input_append_entity_ids = Input(shape=(append_entities_len, ), name='Input-Append-Entities')

append_entity_represent = Embedding(
    input_dim=len(i2c),
    output_dim=len(i2c),
    embeddings_initializer=bert.initializer,
    mask_zero=True,
    name='Entities-Trans'
)(input_append_entity_ids)


append_entity_represent = MaskMean(name="Entities-Mean")(append_entity_represent)

# 加和
final_output = Add(name="Final-Add")([final_output, append_entity_represent])


final_input = Input(shape=(len(i2c), ), name='Output-Label-Id')

type_loss = Lambda(mult_circle_loss, name='Circle-Loss')([final_input, final_output])

train_loss = {
    'Circle-Loss': lambda y_true, y_pred: y_pred
}
    

model = Model(bert.model.inputs + [input_append_entity_ids], final_output)
train_model = Model(bert.model.inputs + [input_append_entity_ids, final_input], type_loss)
    
optimizer = Adam(learning_rate=learning_rate)
train_model.compile(loss=train_loss, optimizer=optimizer)

model.load_weights("./param/outputModelWeights/ICLR_2021_Workshop_MLPCP_Track_1_模板生成填充_文本分类_模型简化2_增加转移概率/best_weights")


with open(os.path.join(data_path, "test_add_info_entities.pk"), "rb") as f:
    test_data = pickle.load(f)


def create_content(data_item):

    content = ""
    append_entities = set()

    for s_idx in range(len(data_item)-1, -1, -1):

        sentence = data_item[s_idx]

        for k, vs in sentence.items():
            if type(vs) == list:
                for v in vs:
                    if (v, k) in c2i:
                        append_entities.add((v, k))

        content = sentence['Sentence'] + content

    return content, append_entities

def predict(data):

    batch_token_ids, batch_segment_ids, batch_append_entity_ids  = [], [], []

    for data_item in data:
            
        content, append_entities = create_content(data_item)

        token_ids, _ = tokenizer.encode(content)
        token_ids = token_ids[1:-1]

        if len(token_ids) > maxlen - 2:
            token_ids = token_ids[-maxlen+2:]

        token_ids = [tokenizer.token_to_id("[CLS]")] + token_ids + [tokenizer.token_to_id("[SEP]")]
        token_ids += [0] * (maxlen - len(token_ids))

        batch_token_ids.append(token_ids)
        batch_segment_ids.append([0] * maxlen)

        # 增加出现实体
        append_entity_ids = [c2i[item] for item in append_entities]
        append_entity_ids += [0] * (append_entities_len - len(append_entity_ids))

        batch_append_entity_ids.append(append_entity_ids)
        

    y = model.predict([batch_token_ids, batch_segment_ids, batch_append_entity_ids], batch_size=64, verbose=1)

    predict_data = []

    for idx in range(len(batch_token_ids)):
        
        predict_data_item = {k:set() for e, k in i2c[1:]}
        
        for item_idx, item in enumerate(y[idx]):
            if item > 0 and item_idx > 0:
                e, t = i2c[item_idx]
                predict_data_item[t].add(e)
                
        predict_data.append(predict_data_item)
        
    return predict_data

print("预测测试集")
test_predict_data = predict(test_data)

for data_item, test_predict_data_item in zip(test_data, test_predict_data):
    
    add_item = {
        'id': 'Doctor',
        'Sentence': '',
        'Symptom': list(test_predict_data_item['Symptom']),
        'Medicine': list(test_predict_data_item['Medicine']),
        'Test': list(test_predict_data_item['Test']),
        'Attribute': list(test_predict_data_item['Attribute']),
        'Disease': list(test_predict_data_item['Disease']),
    }
    
    data_item.append(add_item)

with open(os.path.join(data_path, "test_add_info_entities_with_predict_entities.pk"), "wb") as f:
    pickle.dump(test_data, f)


# 额外预测训练集与开发集

def create_content_label(data_item):
    
    for idx in range(len(data_item)-1, 0, -1):
        
        if data_item[idx]['id'] == 'Doctor':
            
            label = data_item[idx].copy()

            del label['id']
            del label['Sentence']
            
            label = {k:set(v) for k,v in label.items()}

            content = ""
            append_entities = set()

            for s_idx in range(idx-1, -1, -1):
                
                sentence = data_item[s_idx]
                
                for k, vs in sentence.items():
                    if type(vs) == list:
                        for v in vs:
                            append_entities.add((v, k))
                            
                content = sentence['Sentence'] + content

            yield content, append_entities, label
    

import copy

# 训练集与开发集 专用
def predict_all(data):

    batch_token_ids, batch_segment_ids, batch_append_entity_ids  = [], [], []
    
    data_map = []

    for data_item in data:
        
        data_map_item = []
        
        for content, append_entities, label in create_content_label(data_item):

            token_ids, _ = tokenizer.encode(content)
            token_ids = token_ids[1:-1]

            if len(token_ids) > maxlen - 2:
                token_ids = token_ids[-maxlen+2:]

            token_ids = [tokenizer.token_to_id("[CLS]")] + token_ids + [tokenizer.token_to_id("[SEP]")]
            token_ids += [0] * (maxlen - len(token_ids))
            
            data_map_item.append(len(batch_token_ids))

            batch_token_ids.append(token_ids)
            batch_segment_ids.append([0] * maxlen)
            
            # 增加出现实体
            append_entity_ids = [c2i[item] for item in append_entities if item in c2i]
            append_entity_ids += [0] * (append_entities_len - len(append_entity_ids))

            batch_append_entity_ids.append(append_entity_ids)
        
        data_map.append(data_map_item)

    y = model.predict([batch_token_ids, batch_segment_ids, batch_append_entity_ids], batch_size=128, verbose=1)

    predict_data = []

    for data_item, data_map_item in zip(data, data_map):
        
        data_item = copy.deepcopy(data_item)
        
        data_map_item_idx = 0
        
        for idx in range(len(data_item)-1, 0, -1):
            
            if data_item[idx]['id'] == 'Doctor':
                
                y_idx = data_map_item[data_map_item_idx]
                
                # 预测对应的值
                predict_data_item = {k:set() for e, k in i2c[1:]}

                for item_idx, item in enumerate(y[y_idx]):
                    if item > 0 and item_idx > 0:
                        e, t = i2c[item_idx]
                        predict_data_item[t].add(e)
                
                data_item[idx]['bert_word'] = (sorted(predict_data_item['Symptom']) 
                                               + sorted(predict_data_item['Attribute'])
                                               + sorted(predict_data_item['Test'])
                                               + sorted(predict_data_item['Disease'])
                                               + sorted(predict_data_item['Medicine']))

                data_map_item_idx += 1
        
        predict_data.append(data_item)

    return predict_data

print("预测开发集")
with open(os.path.join(data_path, "dev_data.pk"), "rb") as f:
    dev_data = pickle.load(f)

dev_data_with_bert_entites = predict_all(dev_data)

with open(os.path.join(data_path, "dev_data_with_bert_entites.pk"), 'wb') as f:
    pickle.dump(dev_data_with_bert_entites, f)

print("预测训练集")
with open(os.path.join(data_path, "train_data.pk"), "rb") as f:
    train_data = pickle.load(f)

train_data_with_bert_entites = predict_all(train_data)

with open(os.path.join(data_path, "train_data_with_bert_entites.pk"), 'wb') as f:
    pickle.dump(train_data_with_bert_entites, f)