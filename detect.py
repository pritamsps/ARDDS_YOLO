import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Flatten, Dense
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
import numpy as np

def YOLO(input_shape, num_classes):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(2048, activation='relu')(x)
    predictions = Dense(num_classes + 5, activation='sigmoid')(x) 

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# Define loss function
def custom_loss(y_true, y_pred):
    mask_shape = tf.shape(y_true)[:4]
    mask_shape = tf.concat([mask_shape[:-1], tf.constant([5])], axis=0)

    y_true_masked = tf.concat([y_true, tf.zeros(mask_shape)], axis=-1)
    y_pred_masked = tf.concat([y_pred, tf.zeros(mask_shape)], axis=-1)

    loss_object = binary_crossentropy(y_true_masked, y_pred_masked)
    return tf.reduce_mean(loss_object)

# Define IOU metric
def mean_iou(y_true, y_pred):
    iou = MeanIoU(num_classes=2)
    iou.update_state(tf.round(y_true), tf.round(y_pred))
    return iou.result()


input_shape = (224, 224, 3)
num_classes = 1 
# Create YOLO model
model = YOLO(input_shape, num_classes)

# Compile the model
model.compile(optimizer=Adam(), loss=custom_loss, metrics=[mean_iou])

model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))
