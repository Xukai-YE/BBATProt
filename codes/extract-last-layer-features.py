# -*- coding: utf-8 -*-
import os
import numpy as np
np.random.seed(13)
import jsonlines
import warnings
warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # #### generate input file for training data
# f1=open('./Dipeptidyl_peptidase‑IV_inhibitory_peptides/test.txt','r', encoding='UTF-8')
# f1=open('./Golgi protein localization/D3_train.txt','r', encoding='UTF-8')
# f1=open('../data/train_Mem.txt','r', encoding='UTF-8')
# f1=open('./AMP_train/new_test.txt','r', encoding='UTF-8')
# f1=open('./AMP_train/YADAMP.txt','r', encoding='UTF-8')

#
# data1=f1.readlines()
# f1.close()
# Frag=[]
# for i in range(len(data1)):
#     if i%2==0:
#         label=data1[i].strip().split(':')[1]
#         seq=data1[i+1].strip()
#         Frag.append(seq+' '+label)
# seq_length=[]
# np.random.shuffle(Frag)
# fout1=open('./AMP_train/YADAMP.tsv','w')
# # fout1=open('./Golgi protein localization/train.tsv','w')
# # fout2=open('./Golgi protein localization/train_nolabel.tsv','w')
# fout2=open('./AMP_train/YADAMP_nolabel.tsv','w')
#
# for line in Frag:
#     label=line.split(' ')[1]
#     seq=line.split(' ')[0]
#     fout1.write(label+'\t')
#
#     for i in range(len(seq)):
#         if i!=(len(seq)-1):
#             fout1.write(seq[i]+' ')
#             fout2.write(seq[i]+' ')
#         else:
#             fout1.write(seq[i]+'\n')
#             fout2.write(seq[i]+'\n')
#     seq_length.append(len(seq))
# fout1.close()
# fout2.close()
# print('max seq length:',max(seq_length))


with open('./AMP_train/diff-amp.txt', 'r') as f1:
    data1 = f1.readlines()

# 打开输出的带标签的tsv文件和不带标签的tsv文件
with open('./AMP_train/diff_amp.tsv', 'w') as fout1:
    # 遍历数据并写入文件
    for seq in data1:
        fout1.write( seq )


print('Conversion completed successfully.')

# # # ## extract features for training data
# # # os.system('python ./extract_features.py --do_lower_case=True --input_file=./Kcr_train/test_nolabel.tsv --output_file=./Kcr_train/test_Mini.jsonl --vocab_file=../BERT-Kcr-models/BERT-Mini/vocab.txt --bert_config_file=../BERT-Kcr-models/BERT-Mini/bert_config.json --init_checkpoint=../BERT-Kcr-models/BERT-Mini/bert_model.ckpt --layers=-1 --max_seq_length=128 --batch_size=16')
# # # os.system('python ./extract_features.py --do_lower_case=True --input_file=./train_and_test/train_nolabel.tsv --output_file=./train_and_test/train.jsonl --vocab_file=../BERT-Kcr-models/BERT-Mini/vocab.txt --bert_config_file=../BERT-Kcr-models/BERT-Mini/bert_config.json --init_checkpoint=../BERT-Kcr-models/BERT-Mini/bert_model.ckpt --layers=-1 --max_seq_length=128 --batch_size=16')
# # # os.system('python ./extract_features.py --do_lower_case=True --input_file=./train_and_test/train_n_nolabel.tsv --output_file=./train_and_test/train_n_Base.jsonl --vocab_file=../BERT-Kcr-models/BERT-Base/vocab.txt --bert_config_file=../BERT-Kcr-models/BERT-Base/bert_config.json --init_checkpoint=../BERT-Kcr-models/BERT-Base/bert_model.ckpt --layers=-1 --max_seq_length=500 --batch_size=16')
# # # os.system('python ./extract_features.py --do_lower_case=True --input_file=./train_and_test/train_n_nolabel.tsv --output_file=./train_and_test/train_n_Medium.jsonl --vocab_file=../BERT-Kcr-models/BERT-Small/vocab.txt --bert_config_file=../BERT-Kcr-models/BERT-Medium/bert_config.json --init_checkpoint=../BERT-Kcr-models/BERT-Medium/bert_model.ckpt --layers=-1 --max_seq_length=500 --batch_size=16')
# # # os.system('python ./extract_features.py --do_lower_case=True --input_file=./train_and_test/train_Kgly_nolabel.tsv --output_file=./Kgly_train/train_Kgly_prot.jsonl --vocab_file=../BERT-Kcr-models/BERT-Protein/vocab.txt --bert_config_file=../BERT-Kcr-models/BERT-Protein/bert_config.json --init_checkpoint=../BERT-Kcr-models/BERT-Protein/model.ckpt --layers=-1 --max_seq_length=128 --batch_size=16')
# # os.system('python ./extract_features.py --do_lower_case=True --input_file=./Dipeptidyl_peptidase‑IV_inhibitory_peptides/test_nolabel.tsv --output_file=./Dipeptidyl_peptidase‑IV_inhibitory_peptides/test_Small.jsonl --vocab_file=../BERT-Kcr-models/BERT-Small/vocab.txt --bert_config_file=../BERT-Kcr-models/BERT-Small/bert_config.json --init_checkpoint=../BERT-Kcr-models/BERT-Small/bert_model.ckpt --layers=-1 --max_seq_length=128 --batch_size=16')
# os.system('python ./extract_features.py --do_lower_case=True --input_file=./Golgi_protein_localization/train_nolabel.tsv --output_file=./Golgi_protein_localization/train_Small.jsonl --vocab_file=../BERT-Kcr-models/BERT-Small/vocab.txt --bert_config_file=../BERT-Kcr-models/BERT-Small/bert_config.json --init_checkpoint=../BERT-Kcr-models/BERT-Small/bert_model.ckpt --layers=-1 --max_seq_length=512 --batch_size=16')
os.system('python ./extract_features.py --do_lower_case=True --input_file=./AMP_train/diff_amp.tsv --output_file=./AMP_train/diff_amp.jsonl --vocab_file=../BERT-Kcr-models/BERT-Small/vocab.txt --bert_config_file=../BERT-Kcr-models/BERT-Small/bert_config.json --init_checkpoint=../BERT-Kcr-models/BERT-Small/bert_model.ckpt --layers=-1 --max_seq_length=100 --batch_size=16')

# # #
# # # #
# # # layer = []
# # # count = 0
# #
# # # with jsonlines.open('./AMP_train/test_Small.jsonl') as reader:
# # #     for obj in reader:
# # #         sample = []
# # #
# # #         for j in range(31):
# # #             aa=obj['features'][j]['layers'][0]['values']
# # #             sample.extend(aa)
# # #         layer.append(sample)
# # #
# # # print('layer.shape:', np.array(layer).shape)
# # # print(count)
# # # np.savetxt('./AMP_train/train_Small_features.txt', layer, fmt='%s')
# # # #
# #
layer=[]

# with jsonlines.open('./Golgi_protein_localization/train_Small.jsonl') as reader:
with jsonlines.open('./AMP_train/diff_amp.jsonl') as reader:

    for obj in reader:
        sample = []
        if 'features' in obj: # Check if 'features' key exists in the current object
            for feature in obj['features']:
                if 'layers' in feature: # Check if 'layers' key exists in the current feature
                    for layer_obj in feature['layers']:
                        if 'values' in layer_obj: # Check if 'values' key exists in the current layer_obj
                            values = layer_obj['values']
                            sample.extend(values)
        layer.append(sample)

with open('./AMP_train/diff_amp_features.txt', 'w') as f:
    for sample in layer:
        f.write(' '.join(map(str, sample)) + '\n')

print('layer.shape:', np.array(layer).shape)
# #
#
#
# #
# #
# #### generate input file for test data
# # f2=open('../data/independent_test_data.txt','r')
# # data2=f2.readlines()
# # f2.close()
# #
# # Frag=[]
# # for i in range(len(data2)):
# #     if i%2==0:
# #         label=data2[i].strip().split('|')[0].split(':')[1]
# #         seq=data2[i+1].strip()
# #         Frag.append(seq+' '+label)
# #
# # np.random.shuffle(Frag)
# # fout1=open('./train_and_test/test.tsv','w')
# # fout2=open('./train_and_test/test_nolabel.tsv','w')
# # for line in Frag:
# #     label=line.split(' ')[1]
# #     seq=line.split(' ')[0]
# #     fout1.write(label+'\t')
# #     for i in range(len(seq)):
# #         if i!=(len(seq)-1):
# #             fout1.write(seq[i]+' ')
# #             fout2.write(seq[i]+' ')
# #         else:
# #             fout1.write(seq[i]+'\n')
# #             fout2.write(seq[i]+'\n')
# #
# # fout1.close()
# # fout2.close()
# #
# # #
# # # #### extract features for test data
# # # os.system('python ./extract_features.py --do_lower_case=True --input_file=./train_and_test/test_1_nolabel.tsv --output_file=./train_and_test/test_1.jsonl --vocab_file=../BERT-Kcr-models/BERT-Mini/vocab.txt --bert_config_file=../BERT-Kcr-models/BERT-Mini/bert_config.json --init_checkpoint=../BERT-Kcr-models/BERT-Mini/bert_model.ckpt --layers=-1 --max_seq_length=400 --batch_size=16')
# # os.system('python ./extract_features.py --do_lower_case=True --input_file=./train_and_test/test_nolabel.tsv --output_file=./train_and_test/test.jsonl --vocab_file=../BERT-Kcr-models/BERT-Small/vocab.txt --bert_config_file=../BERT-Kcr-models/BERT-Small/bert_config.json --init_checkpoint=../BERT-Kcr-models/BERT-Small/bert_model.ckpt --layers=-1 --max_seq_length=400 --batch_size=16')
# #
# # layer=[]
# # with jsonlines.open('AMP_train/train_Small.jsonl') as reader:
# #     for obj in reader:
# #         sample=[]
# #         for j in range(31):
# #             aa=obj['features'][j]['layers'][0]['values']
# #             sample.extend(aa)
# #         layer.append(sample)
# # print('layer.shape:',np.array(layer).shape)
# # np.savetxt('./train_and_test/test_features.txt',layer,fmt='%s')
#
#
