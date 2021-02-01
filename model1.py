import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Flatten, Conv2D, LeakyReLU, MaxPool2D, Dropout, Reshape
from tensorflow.keras.layers import LSTM, BatchNormalization, Bidirectional, Layer, Softmax
from tensorflow.keras import Model, initializers
import tensorflow.keras.backend as K
import os


class attention(Layer):
    def __init__(self, attention_dim=1):
        super().__init__()
        self.attention_dim = attention_dim

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], self.attention_dim),
                                 initializer="random_normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], self.attention_dim),
                                 initializer="random_normal")
        self.u = self.add_weight(name="att_u", shape=(self.attention_dim, self.attention_dim),
                                 initializer="random_normal")

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        e = K.dot(e, self.u)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)


class ACRNN(Model):
    def __init__(self, num_classes=6, L1=128, L2=256,
                 cell_units=128, num_linear=768, p=10, time_step=150,
                 F1=64, dropout_keep_prob=0.8):
        super().__init__()
        # strides = [1,1,1,1]
        self.conv1 = Conv2D(filters=L1, kernel_size=(5, 3), padding="same", use_bias=True,
                            bias_initializer=initializers.constant(0.1))
        self.dropout_keep_prob = dropout_keep_prob
        self.max_pool = MaxPool2D(pool_size=2, strides=2, padding='valid')
        self.conv2 = Conv2D(filters=L2, kernel_size=(5, 3), padding="same", use_bias=True,
                            bias_initializer=initializers.constant(0.1))
        self.conv3 = Conv2D(filters=L2, kernel_size=(5, 3), padding="same", use_bias=True,
                            bias_initializer=initializers.constant(0.1))
        self.conv4 = Conv2D(filters=L2, kernel_size=(5, 3), padding="same", use_bias=True,
                            bias_initializer=initializers.constant(0.1))
        self.conv5 = Conv2D(filters=L2, kernel_size=(5, 3), padding="same", use_bias=True,
                            bias_initializer=initializers.constant(0.1))
        self.conv6 = Conv2D(filters=L2, kernel_size=(5, 3), padding="same", use_bias=True,
                            bias_initializer=initializers.constant(0.1))
        self.dropout = Dropout(dropout_keep_prob)
        self.reshape1 = Reshape((-1, time_step, L2 * p))
        self.reshape2 = Reshape((-1, L2 * p))
        # self.reshape3 = Reshape((-1, time_step, num_linear))
        self.flatten = Flatten()
        self.d1 = Dense(num_linear, use_bias=True, bias_initializer=initializers.constant(0.1),
                        kernel_initializer=initializers.TruncatedNormal(mean=0., stddev=0.1))
        self.bn = BatchNormalization()
        self.d2 = Dense(F1, use_bias=True, bias_initializer=initializers.constant(0.1),
                        kernel_initializer=initializers.TruncatedNormal(mean=0., stddev=0.1))
        self.d3 = Dense(num_classes, use_bias=True, bias_initializer=initializers.constant(0.1),
                        kernel_initializer=initializers.TruncatedNormal(mean=0., stddev=0.1), activation='softmax')
        self.bilstm = Bidirectional(LSTM(cell_units, return_sequences=True), merge_mode='concat')
        self.attention = attention()

    def call(self, x):
        # 1
        x = self.conv1(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = self.max_pool(x)
        x = Dropout(self.dropout_keep_prob)(x)
        # 2
        x = self.conv2(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = Dropout(self.dropout_keep_prob)(x)
        # 3
        x = self.conv3(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = Dropout(self.dropout_keep_prob)(x)
        # # 4
        # x = self.conv4(x)
        # x = LeakyReLU(alpha=0.01)(x)
        # x = Dropout(self.dropout_keep_prob)(x)
        # # 5
        # x = self.conv5(x)
        # x = LeakyReLU(alpha=0.01)(x)
        # x = Dropout(self.dropout_keep_prob)(x)
        # # 6
        # x = self.conv6(x)
        # x = LeakyReLU(alpha=0.01)(x)
        # x = Dropout(self.dropout_keep_prob)(x)

        x = self.reshape1(x)
        x = self.reshape2(x)

        x = self.d1(x)
        x = self.bn(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = self.bilstm(x)
        
        x = self.attention(x)
        x = self.d2(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = Dropout(self.dropout_keep_prob)(x)
        x = self.d3(x)
        return x

    def model(self):
        x = Input(shape=(300, 40, 3))
        return Model(inputs=x, outputs=self.call(x))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("tf version =========================", tf.__version__)
    ACRNN().model().summary()
