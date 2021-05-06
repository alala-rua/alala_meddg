# meddg by alala

### 0. 环境安装并激活
```
conda env create -f environment.yml
conda activate alala_meddg
```

### 1. 数据准备
在项目根目录下新建data目录
```
mkdir data
```
将官方提供的数据集 new_train.pk 放入 data 目录

将官方提供的测试集 PhasesBTestData.pk 放入 data 目录

下载预训练参数

百度网盘：[链接](https://pan.baidu.com/s/13--Yw20KCA2-D62R1F2Ydw) 提取码：bpdl

解压
```
tar zxvf pretrain_weights.tar.gz
```
会生成pretrain_weights文件夹

### 2. 载入训练完的参数，可选
下载训练完的参数，分5卷，放入项目根目录下

百度网盘：[链接](https://pan.baidu.com/s/1sH4GmQh7gwBKJ0rfturmQw) 提取码：7rem

解压
```
cat param.tar.gz* | tar zx
```
会生成param文件夹，即为训练完成的参数，可跳过训练步骤直接预测

### 3. 数据预处理
```
python data_prepare.py
```

### 4. 角色识别
```
python role_classification/train.py ## 训练过程，可选
python role_classification/predict.py
```

### 5. 实体识别
```
python entity_recognition/data_convert.py
python entity_recognition/train.py ## 训练过程，可选
python entity_recognition/predict.py
```

### 6. 实体预测
```
python next_entities/train.py ## 训练过程，可选
python next_entities/predict.py
```

### 7. 回复生成
```
python generation/data_convert.py
## 以下四行为训练过程，可选
python generation/train_base.py --project_name ICLR_2021_Workshop_MLPCP_Track_1_生成_base_bert_entity \ 
                                --corpus_path ./data/t5_base/corpus.train.bert_entity.tfrecord \
                                --dev_path ./data/dig_dev_data_with_bert_entity.pk
python generation/train_base.py --project_name ICLR_2021_Workshop_MLPCP_Track_1_生成_base_self_entity \ 
                                --corpus_path ./data/t5_base/corpus.train.tfrecord \
                                --dev_path ./data/dig_dev_data_with_bert_entity.pk
python generation/train_difficult.py --project_name ICLR_2021_Workshop_MLPCP_Track_1_生成_difficult_bert_entity \ 
                                --corpus_path ./data/t5_difficult/corpus.train.bert_entity.tfrecord \
                                --dev_path ./data/dig_dev_data_with_bert_entity.pk
python generation/train_difficult.py --project_name ICLR_2021_Workshop_MLPCP_Track_1_生成_difficult_self_entity \ 
                                --corpus_path ./data/t5_difficult/corpus.train.tfrecord \
                                --dev_path ./data/dig_dev_data_with_bert_entity.pk

python generation/predict.py
```