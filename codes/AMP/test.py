# -*- coding: utf-8 -*-
import numpy as np
import random
random.seed(42)
np.random.seed(137)
import os
import pandas as pd
import math
from sklearn.metrics import roc_auc_score
import tensorflow as tf
tf.random.set_seed(141)
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Activation, Flatten,Conv1D, MaxPooling1D, Flatten, Dense, Dropout, MultiHeadAttention
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import *
from sklearn.metrics import *
from tcn import TCN
from sklearn.metrics import roc_curve, auc
import sys
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import warnings
from tensorflow.keras import Model

sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})

warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(tf.test.is_gpu_available())

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set to the GPU device you want to use

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

class Attention3D(Layer):
    def __init__(self, **kwargs):
        super(Attention3D, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape should be (batch_size, time_steps, input_dim)
        input_dim = input_shape[-1]
        self.W_a = self.add_weight(name='W_a', shape=(input_dim, input_dim), initializer='normal', trainable=True)
        super(Attention3D, self).build(input_shape)

    def call(self, inputs, **kwargs):
        a = inputs
        e = K.dot(a, self.W_a)
        e = K.tanh(e)
        a_probs = K.softmax(e, axis=-1)
        output_attention_mul = Multiply()([inputs, a_probs])
        return output_attention_mul

    def compute_output_shape(self, input_shape):
        return input_shape



max_time_steps = 100
INPUT_SIZE = 512
max_columns=max_time_steps*INPUT_SIZE


########################BBATProt###########################

# model = Sequential([
#     # Conv1D(filters=768, kernel_size=3, activation='relu', input_shape=(200, 768)),
#     # Conv1D(filters=512, kernel_size=3, activation='relu', input_shape=(200, 512)),
#     Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(max_time_steps, INPUT_SIZE)),
#     # # # Conv1D(filters=256, kernel_size=3, activation='relu'),
#     BatchNormalization(),
#     # # Dropout(0.1),
#     # # MaxPooling1D(pool_size=2),
#     # GRU(units=256, return_sequences=True, input_shape=(200, 256)),
#     Bidirectional(LSTM(units=128, return_sequences=True)),
#     Attention3D(),
#     TCN(),
#     # # SeqSelfAttention(attention_activation='relu'),  # Attention mechanism
#     # # MultiHeadAttention(num_heads=12, activation='relu', key_dim=64),
#     Flatten(),
#     Dense(256, activation='relu'),
#     # Dropout(0.1),
#     Dense(128, activation='relu'),
#     Dropout(0.1),
#     Dense(64, activation='relu'),
#     # Dropout(0.1),
#     # # # CRF(2, sparse_target=True)
#     Dense(2, activation='softmax')
# ])
#
# # model=attention_model(max_time_steps, INPUT_SIZE)
# adam = tf.keras.optimizers.Adam(2e-4)
# model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
################################iAMP###########################
# SINGLE_ATTENTION_VECTOR = False

# def attention_3d_block(inputs):
#     # inputs.shape = (batch_size, time_steps, input_dim)
#     input_dim = int(inputs.shape[2])
#     a = inputs
#     #a = Permute((2, 1))(inputs)
#     #a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
#     a = Dense(input_dim, activation='softmax')(a)
#     if SINGLE_ATTENTION_VECTOR:
#         a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
#         a = RepeatVector(input_dim)(a)
#     a_probs = Permute((1, 2), name='attention_vec')(a)
#
#     output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
#     return output_attention_mul
#
# def attention_model(max_time_steps, INPUT_SIZE):
#     inputs = Input(shape=(max_time_steps, INPUT_SIZE))
#     # CNN layer
#     x = Conv1D(filters=64,kernel_size=1, activation='relu')(inputs)
#     x = Dropout(0.3)(x)
#
#     x = Bidirectional(LSTM(units=64, return_sequences=True))(x)
#     x = Dropout(0.3)(x)
#     x = attention_3d_block(x)
#     x = Flatten()(x)
#     x = Dense(256, activation='relu')(x)
#     x = Dropout(0.1)(x)
#     x = Dense(128, activation='relu')(x)
#     x = Dropout(0.1)(x)
#     x = Dense(64, activation='relu')(x)
#     x = Dropout(0.1)(x)
#     x = Dense(2)(x)
#     outputs = Activation('sigmoid')(x)
#
#     model = Model(inputs=inputs, outputs=outputs)
#     return model
#
# model=attention_model(max_time_steps, INPUT_SIZE)
# model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8),loss='binary_crossentropy', metrics=['accuracy'] )

# ----------------------------------------- load labels and features
print('load labels……')
sys.stdout.flush()
f2=open('APD3.tsv','r')
data=f2.readlines()
f2.close()
test_label=[]
for ff in range(len(data)):
    test_label.append(int(data[ff].strip().split('\t')[0]))

print('load features……')
sys.stdout.flush()
# test = np.loadtxt('test_features.txt',dtype='str').astype(np.float)
features = []
with open('APD3_features.txt', 'r') as file:
    for line in file:
        line_data = line.strip().split(' ')

        # Fill with 0 or trim the row data
        if len(line_data) < max_columns:
            line_data += [0.0] * (max_columns - len(line_data))
        elif len(line_data) > max_columns:
            line_data = line_data[:max_columns]
        features.append(line_data)

# Convert the list to a NumPy array
features = np.array(features, dtype=float)
print('features:',features.shape)
test = features.reshape(-1,max_time_steps,INPUT_SIZE)
print('test load feature done!',test.shape)
sys.stdout.flush()
# ------------------------------------------- load model and test
print('load model……')
model = tf.keras.models.load_model('BERT_model-BBAT2.h5', custom_objects={'Attention3D':Attention3D, 'TCN': TCN})


print('start predict……')
yy_pred = model.predict(test, batch_size=16, verbose=1)
true_values = np.array(test_label)

y_pred = yy_pred[:,1]
y_scores = np.array(y_pred)
fpr, tpr, _ = roc_curve(true_values, y_scores)
AUC=roc_auc_score(true_values, y_scores)
print('Test AUC:',AUC)
print('start predict……')
input_layer = model.input
output_layer = model.output

bilstm_layer = [layer for layer in model.layers if 'bidirectional' in layer.name][0]
bilstm_output = bilstm_layer.output

attention_layer = [layer for layer in model.layers if 'attention' in layer.name][0]
attention_output = attention_layer.output

model_bilstm_output = Model(inputs=input_layer, outputs=bilstm_output)
model_attention_output = Model(inputs=input_layer, outputs=attention_output)

bilstm_output_test = model_bilstm_output.predict(test)
attention_output_test = model_attention_output.predict(test)
#
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(test[0].T, cmap='viridis', aspect='auto')
axes[0].set_title('Initial Input Sequence', fontsize=16)
axes[0].set_xlabel('Time Steps', fontsize=14)
axes[0].set_ylabel('Feature Index', fontsize=14)

axes[1].imshow(bilstm_output_test[0].T, cmap='viridis', aspect='auto')
axes[1].set_title('Output after BiLSTM', fontsize=16)
axes[1].set_xlabel('Time Steps', fontsize=14)
axes[1].set_ylabel('Feature Index', fontsize=14)

axes[2].imshow(attention_output_test[0].T, cmap='viridis', aspect='auto')
axes[2].set_title('Output after Attention', fontsize=16)
axes[2].set_xlabel('Time Steps', fontsize=14)
axes[2].set_ylabel('Feature Index', fontsize=14)
#
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(test[0].T, cmap='viridis', aspect='auto')
plt.title('Initial Input Sequence', fontsize=16)
plt.xlabel('Time Steps', fontsize=14)
plt.ylabel('Feature Index', fontsize=14)
plt.colorbar(label='Feature Value')
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(bilstm_output_test[0].T, cmap='viridis', aspect='auto')
plt.title('Output after BiLSTM', fontsize=16)
plt.xlabel('Time Steps', fontsize=14)
plt.ylabel('Feature Index', fontsize=14)
plt.colorbar(label='Feature Value')
plt.show()

plt.figure(figsize=(6, 6))
plt.imshow(attention_output_test[0].T, cmap='viridis', aspect='auto')
plt.title('Output after Attention', fontsize=16)
plt.xlabel('Time Steps', fontsize=14)
plt.ylabel('Feature Index', fontsize=14)
plt.colorbar(label='Attention Weight')
plt.show()




binary_predictions = np.round(y_pred)


# print('start predict……')
# y_pred = yy_pred[:,1]
# y_pred = np.argmax(yy_pred, axis=1)
# true_values = np.argmax(true_values, axis=1)
# y_scores = np.array(y_pred)
# AUC=roc_auc_score(true_values, y_scores)
#
#
#
accuracy = accuracy_score(true_values, binary_predictions)
mcc = matthews_corrcoef(true_values, binary_predictions)
tn, fp, fn, tp = confusion_matrix(true_values, binary_predictions).ravel()
# accuracy = accuracy_score(true_values, y_pred)
# mcc = matthews_corrcoef(true_values, y_pred)
# tn, fp, fn, tp = confusion_matrix(true_values, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = precision_score(true_values,binary_predictions)
fscore = f1_score(true_values, binary_predictions)
# 创建 DataFrame
df = pd.DataFrame({'True Label': true_values, 'Predicted Probability': y_scores})

# 将 DataFrame 保存到 Excel 文件中
df.to_csv('predictions.csv', index=False)

#

confusion_mat = confusion_matrix(true_values, binary_predictions)
print('Confusion Matrix:\n', confusion_mat)

# with open('metrics.txt', 'w') as metrics_file:
#     metrics_file.write("Accuracy: {}\n".format(accuracy))
#     metrics_file.write("MCC: {}\n".format(mcc))
#     metrics_file.write("Sensitivity: {}\n ".format(sensitivity))
#     metrics_file.write("Specificity: {}\n ".format(specificity))
#     metrics_file.write("Precision: {}\n".format(precision))
#     metrics_file.write("F1-Score: {}\n".format(fscore))
#     metrics_file.write("ROC_Auc: {}\n".format(AUC))

#

print("Accuracy:", accuracy)
print("MCC:", mcc)
print("Precision:", precision)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("F1-Score:", fscore)
print('Test AUC:',AUC)
#
#
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix')
# plt.show()
#
#
# sys.stdout.flush()

# # Plot ROC curve
# plt.figure(figsize=(8, 8))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.3f})'.format(auc(fpr, tpr)))
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlabel('False Positive Rate (FPR)')
# plt.ylabel('True Positive Rate (TPR)')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()
