# -*- coding: utf-8 -*-
import numpy as np
import random
# random.seed(42)
# np.random.seed(137) 
import os
import math
from sklearn.metrics import roc_auc_score
import tensorflow as tf
# tf.set_random_seed(141)
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
import sys
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns


TIME_STEPS = 200
INPUT_SIZE = 256
model = Sequential()
model.add(Bidirectional(LSTM(units=32,
    batch_input_shape=(None,TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    return_sequences=True,      # True: output at all steps. False: output as last step.
),merge_mode='concat'))
model.add(Dropout(0.2))
# add output layer
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))
adam =tf.keras.optimizers.Adam(2e-4)
model.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])


# ----------------------------------------- load labels and features
print('load labels……')
sys.stdout.flush()
f2=open('Kcr_train/test.tsv', 'r')
data=f2.readlines()
f2.close()
test_label=[]
for ff in range(len(data)):
    test_label.append(int(data[ff].strip().split('\t')[0]))

print('load features……')
sys.stdout.flush()
test = np.loadtxt('./train_and_test/test_features.txt',dtype='str').astype(np.float)
test = test.reshape(-1,200,256)
print('test load feature done!',test.shape)
sys.stdout.flush()

# ------------------------------------------- load model and test
print('load model……')
# model = tf.keras.models.load_model('../BERT-Kcr-models/BERT_BiLSTM.h5')
model = tf.keras.models.load_model('./train_and_test/BERT_BiLSTM_model.h5')

print('start predict……')
yy_pred = model.predict_proba(test, batch_size=16, verbose=1)
np.savetxt('./Kcr_result/test_realuts-.txt',yy_pred,fmt='%s')
true_values = np.array(test_label)
y_pred = yy_pred[:,1]
y_scores = np.array(y_pred)
AUC=roc_auc_score(true_values, y_scores)
print('Test AUC:',AUC)
binary_predictions = np.round(y_pred)

# Calculate and print classification report
classification_rep = classification_report(true_values, binary_predictions)
print('Classification Report:\n', classification_rep)

# Calculate and print confusion matrix
confusion_mat = confusion_matrix(true_values, binary_predictions)
print('Confusion Matrix:\n', confusion_mat)

fpr, tpr, _ = roc_curve(true_values, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# 绘制混淆矩阵的热图
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

sys.stdout.flush()


