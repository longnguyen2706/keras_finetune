from __future__ import absolute_import
import os
import tensorflow as tf
from slim_utils import *
from data_generator import DataGenerator
from datetime import datetime

from keras import backend as K

from keras import Model, optimizers, Sequential
from keras.applications import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout, Reshape, Conv2D
from keras.preprocessing import image
import numpy as np
from keras.callbacks import TensorBoard, EarlyStopping
from keras.initializers import TruncatedNormal
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import categorical_accuracy as accuracy
from keras.metrics import mae as mean_loss


ARCHITECTURE = 'inception_v3'

GENERAL_SETTING = {
    'early_stopping_n_steps': 5,
    'batch_size': 8,
    'eval_step_interval': 100,
    'flip_left_right': False,
    'output_labels': '/tmp/output_labels.txt',
    'random_brightness': 0,
    'random_crop': 0,
    'random_scale': 0,
    'test_batch_size': -1,
    'testing_percentage': 20,
    'validation_percentage': 10,
    'validation_batch_size': -1,
    # 'csvlogfile': csv_log_directory,
    'how_many_training_steps': 10000,
    'image_dir': '/mnt/6B7855B538947C4E/Dataset/JPEG_data/Hela_JPEG/',
    # 'summaries_dir': summaries_directory
}
log_dir = '/home/long/logdir/keras/'


def create_model_info(architecture):
    model_info = {}
    if architecture == 'inception-v3':
        model_info['bottleneck_tensor_size'] = 2048
        model_info['input_width'] = 299
        model_info['input_height'] = 299
        model_info['input_depth'] = 3
        model_info['input_mean'] = 128
        model_info['input_std'] = 128

    elif architecture == 'resnet_v2':
        model_info['bottleneck_tensor_size'] = 2048
        model_info['input_width'] = 224
        model_info['input_height'] = 224
        model_info['input_depth'] = 3
        model_info['input_mean'] = 128
        model_info['input_std'] = 128

    elif architecture == 'inception_resnet_v2':
        model_info['bottleneck_tensor_size'] = 1536
        model_info['input_width'] = 299
        model_info['input_height'] = 299
        model_info['input_depth'] = 3
        model_info['input_mean'] = 128
        model_info['input_std'] = 128

    return model_info

def get_generators(image_list, image_dir):
    # Needed to make sure the logging output is visible.
    # See https://github.com/tensorflow/tensorflow/issues/3047
    tf.logging.set_verbosity(tf.logging.INFO)

    # Look at the folder structure, and create lists of all the images.
    image_lists = create_image_lists(GENERAL_SETTING['image_dir'], GENERAL_SETTING['testing_percentage'],
                                     GENERAL_SETTING['validation_percentage'])
    class_count = len(image_lists.keys())
    if class_count == 0:
        tf.logging.error('No valid folders of images found at ' + image_dir)
        return -1
    if class_count == 1:
        tf.logging.error('Only one valid folder of images found at ' +
                         image_dir +
                         ' - multiple classes are needed for classification.')
        return -1

    sess = tf.Session()
    with sess.as_default():
        # Set up the image decoding sub-graph.
        jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
            model_info['input_width'], model_info['input_height'],
            model_info['input_depth'], model_info['input_mean'],
            model_info['input_std'])


        train_generator = DataGenerator().generate(sess, image_lists, GENERAL_SETTING['batch_size'], 'training',
                                                   GENERAL_SETTING['image_dir'], jpeg_data_tensor, decoded_image_tensor)
        validation_generator = DataGenerator().generate(sess, image_lists, GENERAL_SETTING['batch_size'], 'validation',
                                                        GENERAL_SETTING['image_dir'], jpeg_data_tensor,
                                                        decoded_image_tensor)

        test_generator = DataGenerator().generate(sess, image_lists, GENERAL_SETTING['batch_size'], 'testing',
                                                  GENERAL_SETTING['image_dir'], jpeg_data_tensor, decoded_image_tensor)
    return train_generator, validation_generator, test_generator


def get_model(num_classes, net, weights='imagenet'):
    if net == 'inception-v3':
        base_model = InceptionV3(weights = weights, include_top=False)
    elif net == 'inception-resnet-v2':
        base_model = InceptionResNetV2(weights=weights, include_top=False)
    elif net == 'resnetv2':
        base_model =

    init = TruncatedNormal(mean=0.0, stddev=0.001, seed=None)

    input = base_model.input
    # input = Conv2D(10, kernel_size = (1,1), padding = 'same', activation = 'relu')(input)
    # input = Conv2D(3, kernel_size = (1,1), padding = 'same', activation = 'relu') (input)
    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    # x = Dense(64, input_shape=(2048,), activation='relu',  kernel_initializer=init)(x)
    # x = Dropout(0.2)(x)

    predictions = Dense(10, input_shape=(2048,), activation='softmax', kernel_initializer=init)(x)

    model = Model(input=input, outputs=predictions)

    print(model.summary())


def main(_):
    model_info = create_model_info(ARCHITECTURE)
    train_len = int(842 * 0.7)
    validation_len = int(842 * 0.1)
    test_len = int(842 * 0.2)

    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

    for layer in base_model.layers:
        layer.trainable = True


    # optimizer = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.0, nesterov=False)  # Inception
    optimizer = optimizers.SGD(lr=0.1, decay=0, momentum=0.0, nesterov=False) # Inception-Resnet
    # optimizer = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.99)
    # optimizer = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)

    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    # model.compile(loss="categorical_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    # input = base_model.input
    # labels = tf.placeholder(tf.float32, shape=(None, 10))
    #
    # from keras.objectives import categorical_crossentropy
    # loss = tf.reduce_mean(categorical_crossentropy(labels, predictions))
    #
    # train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    # acc_ops = accuracy(labels, predictions)
    # loss_ops = mean_loss(labels, predictions)
    #
    # init_op = tf.global_variables_initializer()
    # sess.run(init_op)
    #


    # jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding()



    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=GENERAL_SETTING['batch_size'],
                              write_graph=True, write_grads=False)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0,
                                   mode='auto')

    model.fit_generator(
        train_generator,
        epochs=100,
        steps_per_epoch=train_len // GENERAL_SETTING['batch_size'],
        validation_data=validation_generator,
        validation_steps=validation_len //  GENERAL_SETTING['batch_size'],
        callbacks=[tensorboard,early_stopping],
    )

    # for i, layer in enumerate(base_model.layers):
    #     print(i, layer.name)
    #
    # # we chose to train the top 2 inception blocks, i.e. we will freeze
    # # the first 249 layers and unfreeze the rest:
    # # for layer in model.layers[:249]:
    # #     layer.trainable = False
    # # for layer in model.layers[249:]:
    # #     layer.trainable = True
    #
    # # we need to recompile the model for these modifications to take effect
    # # we use SGD with a low learning rate
    # from keras.optimizers import SGD
    # optimizer = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.99)
    # # model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])
    #
    # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.fit_generator(
    #     train_generator,
    #     epochs=100,
    #     steps_per_epoch=train_len // GENERAL_SETTING['batch_size'],
    #     validation_data=validation_generator,
    #     validation_steps=validation_len // GENERAL_SETTING['batch_size'],
    #     callbacks=[tensorboard]
    # )
    #
    score = model.evaluate_generator(test_generator)
    print("score", score)

if __name__ == '__main__':
      tf.app.run(main=main)
