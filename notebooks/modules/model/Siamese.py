from keras import layers, Model
import keras

def get_base_model(inputs):
    
    conv2d_1 = layers.Conv2D(name='seq_1', filters=64, 
            kernel_size=20, 
            activation='relu')(inputs)
    maxpool_1 = layers.MaxPooling2D(pool_size=(2, 2))(conv2d_1)

    conv2d_2 = layers.Conv2D(filters=128, 
            kernel_size=20, 
            activation='relu')(maxpool_1)
    maxpool_2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2d_2)

    conv2d_3 = layers.Conv2D(filters=128, 
            kernel_size=20, 
            activation='relu')(maxpool_2)
    maxpool_3 = layers.MaxPooling2D(pool_size=(2, 2))(conv2d_3)

    conv2d_4 = layers.Conv2D(filters=256, 
            kernel_size=10, 
            activation='relu')(maxpool_3)

    flatten_1 = layers.Flatten()(conv2d_4)
    outputs = layers.Dense(units=4096)(flatten_1)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model