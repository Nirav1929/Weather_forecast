import os, os.path
import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
import tensorflow as tf
import pickle
import pandas as pd
import argparse
import model
import data_generator
from keras import backend as K

sys.path.append('..')


# class Trainer:
#     def __init__(self, path):
#     # Parameters
#     params = {'dim': (512,512),
#               'batch_size': 120,
#               'n_channels': 3,
#               'shuffle': True}
#
#     # Datasets
#     partition = # IDs
#     labels = # Labels
#
#     # Generators
#     training_generator = data_generator.DataGenerator(partition['train'], **params)
#     validation_generator = data_generator.DataGenerator(partition['validation'], **params)
#
#     # Design model
#     ae = model
#
#     # Train model on dataset
#     ae.fit_generator(generator=training_generator,
#                      validation_data=validation_generator,
#                      use_multiprocessing=True,
#                      workers=6)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))



def train_init(num_days, img_path, img_shape, model, optimizer, num_epochs, batch_size, save_interval):
    data_generator_obj = data_generator.DataGenerator(img_path, (img_shape, img_shape, 3),
                                                      batch_size, 3, num_days, num_days)
    # physical_devices = tf.config.list_physical_devices('GPU')
    #
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # from core.yolov4 import YOLO, decode, compute_loss, decode_train
    # config = tf.compat.v1.ConfigProto
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # K.set_session(sess)
    # df_train, df_val = data_generator_obj.create_train_val_split(val_split=0.2)
    # print(df_train.shape, df_val.shape)
    # train_loader = data_generator_obj.generate_data_loader(df_train, df_val, True, batch_size, True, transform, True)
    # val_loader = data_loader_obj.generate_data_loader(df_train, df_val, True, batch_size, True, transform, False)
    # model = model.CnnAutoEncoder()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train_val_obj = train_val(df_train, df_val, train_loader, val_loader, model, optimizer, num_epochs, transform,
    # save_interval, learning_rate, batch_size)
    model.compile(optimizer=optimizer, loss=root_mean_squared_error)
    for epoch in range(num_epochs):
        model.fit(data_generator_obj, epochs=1, use_multiprocessing=True, workers=16)
        model.save_weights("./model" + "model_after" + str(epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', help='Number of epochs for which the model will run', type=int, default=5)
    parser.add_argument('--learning_rate', help='Set learning rate for the model', type=float, default=1e-4)
    parser.add_argument('--image_path',
                        help='Describe the image df path which store image patha and corresponding label',
                        type=str, default='/gpfs_common/share02/rhe/nkpatel8/SST_imgs/')
    parser.add_argument('--batch_size', help="Set the batch size", type=int, default=1)
    parser.add_argument('--save_interval', help='Number of epcohs interval after which we save result and model',
                        type=int, default=2
                        )
    parser.add_argument('--num_days', help='Number of days interval for which we do the forecasting',
                        type=int, default=32
                        )
    parser.add_argument('--image_shape', help='Resized image shape',
                        type=int, default=512
                        )
    args = parser.parse_args()

    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    save_interval = args.save_interval
    img_path = args.image_path
    num_days = args.num_days
    img_shape = args.image_shape

    model = model.CnnAutoEncoder((num_days, img_shape, img_shape, 3), batch_size).forward()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    train_init(num_days, img_path, img_shape, model, optimizer, num_epochs, batch_size, save_interval)
