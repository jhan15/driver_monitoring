import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import RandomRotation, RandomZoom, Dense, Dropout,\
    BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam


class MobileNet(Sequential):
    """
    Customized mobilenet architecture.
    """
    
    def __init__(self, input_shape=(224,224,3), num_classes=2, dropout=0.25, lr=1e-3,
                 augmentation=False, train_base=False, add_layer=False):
        super().__init__()

        self.base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            input_shape=input_shape,
            include_top=False)
        self.base_model.trainable = train_base

        if augmentation:
            self.add(RandomRotation(0.15, input_shape=input_shape))
            self.add(RandomZoom(0.1))
            self.add(Rescaling(1./127.5, offset=-1))
        else:
            self.add(Rescaling(1./127.5, offset=-1, input_shape=input_shape))
        
        self.add(self.base_model)
        self.add(GlobalAveragePooling2D())
        self.add(Dropout(dropout))
        
        if add_layer:
            self.add(Dense(256, activation='relu'))
            self.add(Dropout(dropout))
        
        self.add(Dense(num_classes, activation='softmax'))
        
        self.compile(optimizer=Adam(learning_rate=lr),
                     loss='sparse_categorical_crossentropy',
                     metrics=["accuracy"])
