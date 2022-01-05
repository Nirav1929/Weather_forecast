from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Conv3D, BatchNormalization, Dropout, MaxPool3D, Flatten, Dense, \
    Reshape, Conv3DTranspose, GlobalAveragePooling3D, UpSampling3D


class CnnAutoEncoder:

    def __init__(self, s, batch_size):
        self.input_shape = s

        self.inp = Input(self.input_shape, batch_size=batch_size)
        self.encoder = Conv3D(8, (3, 3, 3), activation='leaky_relu', padding="same")(self.inp)
        self.encoder = BatchNormalization(momentum=0.0)(self.encoder)
        self.encoder = MaxPool3D((2, 2, 2))(self.encoder)
        self.encoder = Dropout(.2)(self.encoder)

        self.encoder = Conv3D(16, (3, 3, 3), activation='leaky_relu', padding="same")(self.encoder)
        self.encoder = BatchNormalization(momentum=0.0)(self.encoder)
        self.encoder = MaxPool3D((2, 2, 2))(self.encoder)
        self.encoder = Dropout(.2)(self.encoder)

        self.encoder = Conv3D(32, (3, 3, 3), activation='leaky_relu', padding="same")(self.encoder)
        self.encoder = BatchNormalization(momentum=0.0)(self.encoder)
        self.encoder = MaxPool3D((2, 2, 2))(self.encoder)
        self.encoder = Dropout(.2)(self.encoder)

        self.encoder = Conv3D(64, (3, 3, 3), activation='leaky_relu', padding="same")(self.encoder)
        self.encoder = BatchNormalization(momentum=0.0)(self.encoder)
        self.encoder = MaxPool3D((2, 2, 2))(self.encoder)
        self.encoder = Dropout(.2)(self.encoder)

        self.encoder = Conv3D(128, (3, 3, 3), activation='leaky_relu', padding="same")(self.encoder)
        self.encoder = BatchNormalization(momentum=0.0)(self.encoder)
        self.encoder = MaxPool3D((2, 2, 2))(self.encoder)
        self.encoder = Dropout(.2)(self.encoder)

        self.layer = GlobalAveragePooling3D('channels_last')(self.encoder)
        self.layer = Dense(256, activation='linear')(self.layer)

        self.decoder = Reshape((1, 16, 16, 1))(self.layer)
        self.decoder = Conv3DTranspose(128, (3, 3, 3), strides=2, activation='relu', padding='same')(self.decoder)
        self.decoder = BatchNormalization()(self.decoder)
        self.decoder = UpSampling3D(size=(2, 2, 2))(self.decoder)

        self.decoder = Conv3DTranspose(64, (3, 3, 3), strides=2, activation='relu', padding='same')(self.decoder)
        self.decoder = BatchNormalization()(self.decoder)
        self.decoder = UpSampling3D(size=(2, 2, 2))(self.decoder)

        self.decoder = Conv3DTranspose(32, (3, 3, 3), activation='relu', padding='same')(self.decoder)
        self.decoder = BatchNormalization()(self.decoder)
        self.decoder = UpSampling3D(size=(2, 2, 2))(self.decoder)

        self.decoder = Conv3DTranspose(16, (3, 3, 3), activation='relu', padding='same')(self.decoder)
        self.decoder = BatchNormalization()(self.decoder)
        # self.decoder = UpSampling2D(size=(2, 2))(self.decoder)

        self.decoder = Conv3DTranspose(8, (3, 3, 3), activation='relu', padding='same')(self.decoder)
        self.decoder = BatchNormalization()(self.decoder)
        # self.decoder = UpSampling2D(size=(1, 2))(self.decoder)
        self.decoder = Conv3D(3, (3, 3, 3), activation='linear', padding='same')(self.decoder)

    def forward(self):
        self.auto_encoder = Model(self.inp, self.decoder)
        return self.auto_encoder
