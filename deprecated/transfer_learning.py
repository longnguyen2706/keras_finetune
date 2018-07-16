from __future__ import absolute_import

from keras import Model, optimizers
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import GlobalAveragePooling2D, Dense

from deprecated.prepare_data import get_generators, create_image_lists

# model = InceptionResNetV2(weights='imagenet')
#
# img_path = '/mnt/6B7855B538947C4E/Dataset/JPEG_data/Hela_JPEG/ActinFilaments/r20oct98.phal.01--1---2.dat.jpg'
#
# img = image.load_img(img_path, target_size=(382, 512))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis =0)
# x = preprocess_input(x)
# preds = model.predict(x)
#
# print ("predicted: ", decode_predictions(preds, top=3)[0])

train_data_dir = '/mnt/6B7855B538947C4E/Dataset/JPEG_data/Hela_JPEG/'


img_width, img_height = 512, 382
# train_data_dir = "data/train"
# validation_data_dir = "data/val"
# nb_train_samples = 4125
# nb_validation_samples = 466
batch_size = 8
epochs = 10

log_dir = '/mnt/6B7855B538947C4E/logdir/keras/'

base_model = InceptionResNetV2(weights='imagenet', include_top=False)

base_model_out = base_model.output
base_model_out = GlobalAveragePooling2D()(base_model_out)

hidden1 = Dense(64, activation='relu')(base_model_out)
predictions = Dense(10, activation='softmax')(hidden1)

for layer in base_model.layers:
    layer.trainable = True

model = Model(input=base_model.input, outputs=predictions)

adam = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.99)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=['accuracy'])

# train_datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     # # horizontal_flip=True,
#     # # fill_mode="nearest",
#     # # zoom_range=0.3,
#     # # width_shift_range=0.3,
#     # # height_shift_range=0.3,
#     # # rotation_range=30
# )
# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size = (img_height, img_width),
#     batch_size=batch_size,
#     class_mode="categorical"
# )
# # print (train_generator.samples)
# # print(train_generator.classes)
# # print (train_generator.filenames)

image_lists = create_image_lists(train_data_dir, 10)
train_generator, validation_generator = get_generators(image_lists, train_data_dir)

#
# tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, batch_size=batch_size,
#                           write_graph=True,write_grads=True)
# early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0,
#                                mode='auto')


model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=100,
    # callbacks=[early_stopping]
)

