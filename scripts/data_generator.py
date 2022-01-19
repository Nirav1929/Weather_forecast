import numpy as np
import pandas as pd
import keras
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
import re


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, df, image_path, input_shape=(512, 512, 3), batch_size=120, channels=3, steps_in=30,
                 steps_out=30):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.img_path = Path(image_path)
        self.num_imgs = len([name for name in os.listdir(self.img_path)
                             if os.path.isfile(os.path.join(self.img_path, name))])
        self.channels = channels
        self.steps_in = steps_in
        self.steps_out = steps_out
        self.df = df
        self.n = len(self.df)

    def create_train_val_split(self, val_split=0.2):
        df_train, df_val = train_test_split(self.df, test_size=val_split)
        print(df_train.shape, df_val.shape)
        return df_train, df_val

    def add_padding(self, img):
        ww, hh, cc = self.input_shape
        color = (255, 255, 255)
        ht, wd, cc = img.shape
        xx = 0
        yy = 0
        resized = np.full((hh, ww, cc), color, dtype=np.float64)
        resized[yy:yy + ht, xx:xx + wd] = img
        return resized

    def generate_batch(self, df, index):
        # print("curr_index is", index)
        # find the end of this pattern
        in_end_ix = index - self.steps_in
        out_end_ix = index + self.steps_out
        # gather input and output parts of the pattern
        seq_x, seq_y = list(), list()
        iob = False
        oob = False
        # check if we are beyond the sequence
        if in_end_ix < 0:
            img = np.array(Image.open(self.img_path / f"day_{index}.jpeg"), dtype=np.float64)
            img = self.add_padding(img)
            for idx in range(self.steps_in):
                seq_x.append(img)
            iob = True
        if out_end_ix > self.num_imgs - 1:
            for idx in range(self.steps_out):
                seq_y.append(img)
            oob = True
        # if iob:
        #     return np.asarray(seq_x).astype(float), np.asarray(seq_y).astype(float)

        if not iob :
            for idx in range(in_end_ix, index):
                img = np.array(Image.open(self.img_path / f"day_{idx}.jpeg"), dtype=np.float64)
                img = self.add_padding(img)
                seq_x.append(img)
        if not oob:
            for idx in range(index, out_end_ix):
                img = np.array(Image.open(self.img_path / f"day_{idx}.jpeg"), dtype=np.float64)
                img = self.add_padding(img)
                seq_y.append(img)
        return np.asarray(seq_x).astype(float), np.asarray(seq_y).astype(float)

    def __get_data(self, batch_indices):
        # Generates data containing batch_size samples X_batches, Y_batches = np.empty(shape=(self.batch_size,
        # self.steps_in, self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype=np.float64),
        # \ np.empty(shape=(self.batch_size, self.steps_out, self.input_shape[0], self.input_shape[1],
        # self.input_shape[2]), dtype=np.float64)
        X_batches, Y_batches = list(), list()
        for idx in batch_indices:
            xx, yy = self.generate_batch(self.df, idx)
            if xx is not None:
                X_batches.append(xx)
                Y_batches.append(yy)
                # np.append(X_batches, xx, axis=0)
                # np.append(Y_batches, yy, axis=0)
            # batch = [np.stack(samples, axis=0) for samples in zip(X_batches, Y_batches)]
            else:
                return None, None
        return np.asarray(X_batches), np.asarray(Y_batches)
        # return batch

    def __getitem__(self, index):
        day = re.findall(r'\d+', self.df['img_name'].iloc[index])
        batch_indices = [idx for idx in range(int(day[0]) * self.batch_size, (int(day[0]) + 1) * self.batch_size)]
        X, Y = self.__get_data(batch_indices)
        return X, Y

    def __len__(self):
        return len(self.df) // self.batch_size
