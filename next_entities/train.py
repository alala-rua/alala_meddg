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

import pickle

with open(os.path.join(data_path, "train_data.pk"), "rb") as f:
    train_data = pickle.load(f)
    
with open(os.path.join(data_path, "dev_data.pk"), "rb") as f:
    dev_data = pickle.load(f)

print(len(train_data))
print(len(dev_data))

def create_content_label(data_item):
    
    for idx in range(len(data_item)-1, 0, -1):
        
        if data_item[idx]['id'] == 'Doctor':
            
            label = data_item[idx].copy()

            del label['id']
            del label['Sentence']
            
            label = {k:set(v) for k,v in label.items()}

            content = ""

            for s_idx in range(idx-1, -1, -1):
                
                sentence = data_item[s_idx]

                content = sentence['Sentence'] + content

            yield content, label

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        
        batch_token_ids, batch_segment_ids = [], []
        batch_labels = []
        
        prob = 1.0

        for is_end, data_item in self.sample(random):
            
            for content, label in create_content_label(data_item):
                
                if sum([len(v) for v in label.values()]) == 0:
                    if np.random.uniform() < prob:
                        continue
                        
#                 print(content)
                        
#                 print(label)
                
                # change prob
                prob -= (prob / epochs / skip_epochs)

                token_ids, _ = tokenizer.encode(content)
                token_ids = token_ids[1:-1]

                if len(token_ids) > maxlen - 2:
                    token_ids = token_ids[-maxlen+2:]

                token_ids = [tokenizer.token_to_id("[CLS]")] + token_ids + [tokenizer.token_to_id("[SEP]")]
                token_ids += [0] * (maxlen - len(token_ids))

                batch_token_ids.append(token_ids)
                batch_segment_ids.append([0] * maxlen)
                
                # 生成答案
                label_item = [2 for _ in i2c]

                for k, v in label.items():
                    
                    for v_item in v:
                        label_item[c2i[(v_item, k)]] = 1
                
                batch_labels.append(label_item)
                
                if len(batch_token_ids) == self.batch_size or is_end:

                    yield {
                        'Input-Token': np.array(batch_token_ids),
                        'Input-Segment': np.array(batch_segment_ids),
                        'Output-Label-Id': np.array(batch_labels),
                    },{
                        'Circle-Loss': np.zeros((len(batch_token_ids),)),
                    }

                    batch_token_ids, batch_segment_ids = [], []
                    batch_labels = []

# 模型开始

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

final_input = Input(shape=(len(i2c), ), name='Output-Label-Id')

type_loss = Lambda(mult_circle_loss, name='Circle-Loss')([final_input, final_output])

train_loss = {
    'Circle-Loss': lambda y_true, y_pred: y_pred
}
    

model = Model(bert.model.inputs, final_output)
train_model = Model(bert.model.inputs + [final_input], type_loss)
    
optimizer = Adam(learning_rate=learning_rate)
train_model.compile(loss=train_loss, optimizer=optimizer)

train_model.summary()


def predict(data):
    
    batch_token_ids, batch_segment_ids = [], []

    for data_item in data:

        content = "".join(data_item['history'])

        token_ids, _ = tokenizer.encode(content)
        token_ids = token_ids[1:-1]

        if len(token_ids) > maxlen - 2:
            token_ids = token_ids[-maxlen+2:]

        token_ids = [tokenizer.token_to_id("[CLS]")] + token_ids + [tokenizer.token_to_id("[SEP]")]
        token_ids += [0] * (maxlen - len(token_ids))

        batch_token_ids.append(token_ids)
        batch_segment_ids.append([0] * maxlen)
        
    
    y = model.predict([batch_token_ids, batch_segment_ids], batch_size=128, verbose=1)
    
    predict_data = []

    for idx in range(len(batch_token_ids)):
        
        predict_data_item = {k:set() for e, k in i2c}
        
        for item_idx, item in enumerate(y[idx]):
            if item > 0:
                e, t = i2c[item_idx]
                predict_data_item[t].add(e)
                
        predict_data.append(predict_data_item)
    
    return predict_data
    

def evaluate(data):

    batch_token_ids, batch_segment_ids = [], []
    
    true_labels = []

    for data_item in data:

        for content, label in create_content_label(data_item):

            token_ids, _ = tokenizer.encode(content)
            token_ids = token_ids[1:-1]

            if len(token_ids) > maxlen - 2:
                token_ids = token_ids[-maxlen+2:]

            token_ids = [tokenizer.token_to_id("[CLS]")] + token_ids + [tokenizer.token_to_id("[SEP]")]
            token_ids += [0] * (maxlen - len(token_ids))

            batch_token_ids.append(token_ids)
            batch_segment_ids.append([0] * maxlen)
            
            true_labels.append(label)


    y = model.predict([batch_token_ids, batch_segment_ids], batch_size=128, verbose=1)

    predict_data = []

    for idx in range(len(batch_token_ids)):
        
        predict_data_item = {k:set() for e, k in i2c}
        
        for item_idx, item in enumerate(y[idx]):
            if item > 0:
                e, t = i2c[item_idx]
                predict_data_item[t].add(e)
                
        predict_data.append(predict_data_item)
        
    score = {k:{'p': 0, 's': 0, 'i':0} for e, k in i2c}

    for predict_data_item, true_data_item in zip(predict_data, true_labels):

        for label, pred_entity in  predict_data_item.items():

            score[label]['p'] += len(pred_entity)
            score[label]['s'] += len(true_data_item[label])
            score[label]['i'] += len(true_data_item[label] & pred_entity)

    all_p, all_s, all_i = 0, 0, 0

    for k,v in score.items():
        all_p += v['p']
        all_s += v['s']
        all_i += v['i']

        P = v['i'] / v['p'] if v['p'] != 0 else 0
        R = v['i'] / v['s'] if v['s'] != 0 else 0 
        f1 = 2 * P * R / (P + R) if P + R != 0 else 0 

        print(k + " P: %.2f R: %.2f F1: %.2f" % (P*100, R*100, f1*100))

    P = all_i / all_p if all_p != 0 else 0
    R = all_i / all_s if all_s != 0 else 0 
    f1 = 2 * P * R / (P + R) if P + R != 0 else 0 

    print("ALL P: %.2f R: %.2f F1: %.2f" % (P*100, R*100, f1*100))

    return f1

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self, model_saved_path):
        
        self.best_bleu = 0.
        self.model_saved_path = model_saved_path

    def on_epoch_end(self, epoch, logs=None):
        metrics = evaluate(dev_data)

        if metrics > self.best_bleu:
            self.best_bleu = metrics
            model.save_weights(os.path.join(self.model_saved_path, "best_weights"), overwrite=True)  # 保存模型

print(projectName + ' Train...')
now = time.strftime("%Y-%m-%d_%H-%M-%S")
resultPath = './param/outputModelWeights/{}/'.format(projectName)
if not os.path.exists(resultPath):
    os.makedirs(resultPath)

train_generator = data_generator(train_data, batch_size)

train_model.fit(
    train_generator.forfit(),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[Evaluator(resultPath)]
)