# %% [markdown]
# 

# %%
%load_ext autoreload
%autoreload 2

# %%
import os
os.environ['KERAS_BACKEND'] = "tensorflow"
import tensorflow as tf
import keras
import keras.backend as K
from keras import layers, Model
from keras.optimizers import Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# import tensorflow as tf

from modules.dataset import EquilibratedDataset
from modules.encoding import ProteinOneHotEncoder

# from modules.data.loader import SequenceEncodingDataGenerator

# %%
plt.style.use('seaborn-v0_8')

# %% [markdown]
# ## Encoding
# 
# ### NLF
# 
# We want to encode the sequence first, and retrieve the encoded for each sequence in the join later.

# %%
encoder = ProteinOneHotEncoder()

# %%
vector_size = 200
X, y, vector_size, alphabet_size = EquilibratedDataset(encoder).getdata(vector_size)

x_ab, x_ag = X


# %%
x_train_ab, x_test_ab, x_train_ag, x_test_ag, y_train, y_test = train_test_split(x_ab, x_ag, y, test_size=0.2, random_state=42, stratify=y)


# %%
input_dimensions = (vector_size, alphabet_size)
input_dimensions

# %% [markdown]
# ## Siamese network

# %%
seq_input1 = layers.Input(shape=input_dimensions, name='seq_ag')
seq_input2 = layers.Input(shape=input_dimensions, name='seq_ab')

# %%
# Convolutional modules
filters = 96


conv01 = layers.Conv1D(filters, 11, padding='same', activation="relu")
mp1 = layers.MaxPooling1D(3)
conv02 = layers.Conv1D(filters*2, 7, padding='same', activation="relu")
mp2 = layers.MaxPooling1D(3)
conv03 = layers.Conv1D(filters*4, 3, padding='same', activation="relu")
mp3 = layers.MaxPooling1D(3)
conv04 = layers.Conv1D(filters*2, 3, padding='same', activation="relu")
mp4 = layers.MaxPooling1D(3)

gru = layers.Bidirectional(layers.GRU(filters, return_sequences=False))

def siamese_propagation(x):
    x = conv01(x)
    x = mp1(x)

    x = conv02(x)
    x = mp2(x)

    x = conv03(x)
    x = mp3(x)

    x = conv04(x)
    x = mp4(x)

    x_gru = gru(x)
    return x_gru



# %%
def forward(left, right):
    left = siamese_propagation(left)
    right = siamese_propagation(right)

    merge = layers.multiply([left, right])

    # merge = layers.Dense(filters*2, activation='relu')(merge)
    merge = layers.Dropout(0.2)(merge)
    return layers.Dense(1, activation='sigmoid')(merge)

# %%
def f1(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    pred_pos = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = tp / (pred_pos + K.epsilon())
    recall = tp / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def mcc(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / (denominator + K.epsilon())

def accuracy(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    return (tp + tn) / (tp + tn + fp + fn)

# %%
def binary_crossentropy(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = - K.mean(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
    return loss

# %%
model = Model(inputs=[seq_input1, seq_input2],
            outputs=[forward(seq_input1, seq_input2)])

adam = Adam(learning_rate=1e-4, amsgrad=True, epsilon=1e-6)

checkpoint_callback = keras.callbacks.ModelCheckpoint(f'run/models/siamese/one-hot-{vector_size}-model.h5', monitor='val_mcc', mode='max')
earlystop_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

model.compile(optimizer=adam, loss=binary_crossentropy, metrics=[accuracy, f1, mcc])

with tf.device('/GPU:0'):
    history = model.fit([x_train_ab, x_train_ag], y_train, epochs=100, callbacks=[],
        batch_size=64, verbose=1, validation_split=0.15)


# %%
model.evaluate([x_test_ab, x_test_ag], y_test)

# %%
plt.figure(figsize = (12,8))
plt.subplot(121)
plt.plot(history.history['accuracy'], label = "Accuracy train")
plt.plot(history.history['val_accuracy'], label = "Accuracy validation")
plt.ylabel('Accuracy', fontsize = 14)
plt.xlabel('Epoch', fontsize = 14)
plt.legend(fontsize = 14)

plt.subplot(122)
plt.plot(history.history['loss'], label = "Loss train")
plt.plot(history.history['val_loss'], label = "Loss validation")
# plt.title('model accuracy')
plt.ylabel('Loss', fontsize = 14)
plt.xlabel('Epoch', fontsize = 14)
plt.legend(fontsize = 14)
plt.suptitle('Siamese training', fontsize = 17)
plt.savefig('../plots/Siamese_training_curve.png')
plt.show()


# %%
plt.figure(figsize = (12,8))
plt.subplot(121)
plt.plot(history.history['f1'], label = "F1-score train")
plt.plot(history.history['val_f1'], label = "F1-score validation")
plt.ylabel('F1-score', fontsize = 14)
plt.xlabel('Epoch', fontsize = 14)
plt.legend(fontsize = 14)

plt.subplot(122)
plt.plot(history.history['mcc'], label = "MCC train")
plt.plot(history.history['val_mcc'], label = "MCC Validation")
# plt.title('model accuracy')
plt.ylabel('Mcc', fontsize = 14)
plt.xlabel('Epoch', fontsize = 14)
plt.legend(fontsize = 14)
plt.suptitle('Siamese training', fontsize = 17)
plt.savefig('../plots/Siamese_training_metrics.png')
plt.show()

# %%


# %%


# %%


# %%



