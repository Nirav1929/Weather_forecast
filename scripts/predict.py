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
from tensorflow.keras.models import Model
from PIL import Image

sys.path.append('..')

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def test_init(df, num_days, img_path, img_shape, test_model, batch_size):
    data_generator_obj = data_generator.DataGenerator(df, img_path, (img_shape, img_shape, 3),
                                                      batch_size, 3, num_days, num_days)
    output_imgs = test_model.predict(data_generator_obj, use_multiprocessing=True, workers=16)
    for batch in range(output_imgs.shape[0]):
        for idx in range(output_imgs.shape[1]):
            # print("sdfdsfdffsd", output_imgs[batch, idx, :, :, :].shape)
            curr_img = Image.fromarray(output_imgs[batch, idx, :, :, :], 'RGB')
            curr_img.save("./day_" + str(10000 + idx + 1) + ".jpeg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        help='Describe the model path to use',
                        type=str, default='/gpfs_share/rhe/nkpatel8/Ocean_SST/model/modelmodel_after4')
    parser.add_argument('--image_path',
                        help='Describe the image df path which store image patha and corresponding label',
                        type=str, default='/gpfs_common/share02/rhe/nkpatel8/SST_imgs/')
    parser.add_argument('--test_image_path',
                        help='Describe the test image df path which store image patha and corresponding label',
                        type=str, default='/gpfs_common/share02/rhe/nkpatel8/SST_temp/')
    parser.add_argument('--batch_size', help="Set the batch size", type=int, default=1)
    parser.add_argument('--num_days', help='Number of days interval for which we do the forecasting',
                        type=int, default=32
                        )
    parser.add_argument('--image_shape', help='Resized image shape',
                        type=int, default=512
                        )
    args = parser.parse_args()
    batch_size = args.batch_size
    img_path = args.image_path
    test_img_path = args.test_image_path
    num_days = args.num_days
    img_shape = args.image_shape
    model = model.CnnAutoEncoder((num_days, img_shape, img_shape, 3), batch_size).forward()
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss=root_mean_squared_error)
#    model.load_weights(filepath=args.model_path)
    num_imgs = len([name for name in os.listdir(test_img_path)
                    if os.path.isfile(os.path.join(test_img_path, name))])
    print("num_of_images in input ", num_imgs)
    df = pd.DataFrame(columns=['img_name'])
    for i, img in enumerate(os.listdir(test_img_path)):
        df.loc[i] = img
    test_init(df, num_days, img_path, img_shape, model, batch_size)
