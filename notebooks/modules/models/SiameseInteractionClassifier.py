from keras import layers, Model
from keras.optimizers import Adam
import keras
import keras.backend as K


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

def forward(self, left, right):
        left = self.siamese_propagation(left)
        right = self.siamese_propagation(right)

        merge = layers.multiply([left, right])

        # merge = layers.Dense(filters*2, activation='relu')(merge)
        merge = layers.Dropout(0.2)(merge)
        return layers.Dense(1, activation='sigmoid')(merge)

def binary_crossentropy(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = - K.mean(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        return loss


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

        

class SiameseInteractionClassifier:

        def __init__(self, filters, seq_input1, seq_input2):

                self.conv01 = layers.Conv1D(filters, 11, padding='same', activation="relu")
                self.mp1 = layers.MaxPooling1D(3)
                self.conv02 = layers.Conv1D(filters*2, 7, padding='same', activation="relu")
                self.mp2 = layers.MaxPooling1D(3)
                self.conv03 = layers.Conv1D(filters*4, 3, padding='same', activation="relu")
                self.mp3 = layers.MaxPooling1D(3)
                self.conv04 = layers.Conv1D(filters*2, 3, padding='same', activation="relu")
                self.mp4 = layers.MaxPooling1D(3)

                self.gru = layers.Bidirectional(layers.GRU(filters, return_sequences=False))
                
                self.model = Model(inputs=[seq_input1, seq_input2],
                outputs=[self.forward(seq_input1, seq_input2)])
                adam = Adam(learning_rate=1e-4, amsgrad=True, epsilon=1e-6)
                self.model.compile(optimizer=adam, loss=binary_crossentropy, metrics=[accuracy, f1, mcc])

        def siamese_propagation(self, x):
                x = self.conv01(x)
                x = self.mp1(x)

                x = self.conv02(x)
                x = self.mp2(x)

                x = self.conv03(x)
                x = self.mp3(x)

                x = self.conv04(x)
                x = self.mp4(x)

                x_gru = self.gru(x)
                return x_gru    
        
