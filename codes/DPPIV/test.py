# -*- coding: utf-8 -*-
import numpy as np
import random
random.seed(42)
np.random.seed(137)
import os
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
warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(tf.test.is_gpu_available())

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set to the GPU device you want to use

# Check if GPU is available
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



max_time_steps = 128
INPUT_SIZE = 512
max_columns=max_time_steps*INPUT_SIZE


########################BBATProt###########################

model = Sequential([
    # Conv1D(filters=768, kernel_size=3, activation='relu', input_shape=(200, 768)),
    # Conv1D(filters=512, kernel_size=3, activation='relu', input_shape=(200, 512)),
    Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(max_time_steps, INPUT_SIZE)),
    # # # Conv1D(filters=256, kernel_size=3, activation='relu'),
    BatchNormalization(),
    # # Dropout(0.1),
    # # MaxPooling1D(pool_size=2),
    # GRU(units=256, return_sequences=True, input_shape=(200, 256)),
    Bidirectional(LSTM(units=256, return_sequences=True)),
    Attention3D(),
    TCN(),
    # # SeqSelfAttention(attention_activation='relu'),  # Attention mechanism
    # # MultiHeadAttention(num_heads=12, activation='relu', key_dim=64),
    Flatten(),
    Dense(256, activation='relu'),
    # Dropout(0.1),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(64, activation='relu'),
    # Dropout(0.1),
    # # # CRF(2, sparse_target=True)
    Dense(2, activation='softmax')
])

# model=attention_model(max_time_steps, INPUT_SIZE)
adam = tf.keras.optimizers.Adam(2e-4)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
################################BERT-Kcr###########################
# model = Sequential()
# model.add(Bidirectional(LSTM(units=32,
#                              batch_input_shape=(None, max_time_steps, INPUT_SIZE),
#                              # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
#                              return_sequences=True,  # True: output at all steps. False: output as last step.
#                              ), merge_mode='concat'))
# model.add(Dropout(0.2))
# # add output layer
# model.add(Flatten())
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(2))
# model.add(Activation('softmax'))
# adam = tf.keras.optimizers.Adam(2e-4)
# model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
#
# # model=attention_model(max_time_steps, INPUT_SIZE)
# model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8),loss='binary_crossentropy', metrics=['accuracy'] )

# ----------------------------------------- load labels and features
print('load labels……')
sys.stdout.flush()
f2=open('test.tsv','r')
data=f2.readlines()
f2.close()
test_label=[]
for ff in range(len(data)):
    test_label.append(int(data[ff].strip().split('\t')[0]))

print('load features……')
sys.stdout.flush()
# test = np.loadtxt('test_features.txt',dtype='str').astype(np.float)
features = []
with open('test_features.txt', 'r') as file:
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
model = tf.keras.models.load_model('BERT_model-T.h5', custom_objects={'Attention3D':Attention3D, 'TCN': TCN})
# model = tf.keras.models.load_model('BERT_Kcr_model.h5')
# model = tf.keras.models.load_model('./train_and_test/BERT_BiLSTM_model.h5')


print('start predict……')
yy_pred = model.predict(test, batch_size=16, verbose=1)
np.savetxt('test_realuts_BBATProt.txt',yy_pred,fmt='%s')
true_values = np.array(test_label)
np.savetxt('true_labels_BBATProt.npy', true_values)

y_pred = yy_pred[:,1]
y_scores = np.array(y_pred)
fpr, tpr, _ = roc_curve(true_values, y_scores)
AUC=roc_auc_score(true_values, y_scores)
print('Test AUC:',AUC)

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


#

confusion_mat = confusion_matrix(true_values, binary_predictions)
print('Confusion Matrix:\n', confusion_mat)

with open('metrics.txt', 'w') as metrics_file:
    metrics_file.write("Accuracy: {}\n".format(accuracy))
    metrics_file.write("MCC: {}\n".format(mcc))
    metrics_file.write("Sensitivity: {}\n ".format(sensitivity))
    metrics_file.write("Specificity: {}\n ".format(specificity))
    metrics_file.write("Precision: {}\n".format(precision))
    metrics_file.write("F1-Score: {}\n".format(fscore))
    metrics_file.write("ROC_Auc: {}\n".format(AUC))

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
