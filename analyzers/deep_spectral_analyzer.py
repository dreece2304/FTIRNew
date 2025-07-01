# analyzers/deep_spectral_analyzer.py

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

class DeepSpectralDenoiser:
    def __init__(self, input_dim):
        self.model = self._build_autoencoder(input_dim)

    def _build_autoencoder(self, input_dim):
        inp = Input(shape=(input_dim,))
        encoded = Dense(128, activation='relu')(inp)
        encoded = Dense(64, activation='relu')(encoded)
        bottleneck = Dense(32, activation='relu')(encoded)

        decoded = Dense(64, activation='relu')(bottleneck)
        decoded = Dense(128, activation='relu')(decoded)
        out = Dense(input_dim, activation='linear')(decoded)

        model = Model(inp, out)
        model.compile(optimizer=Adam(1e-3), loss='mse')
        return model

    def train(self, X, epochs=100, batch_size=16):
        self.model.fit(X, X, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)

    def denoise(self, spectrum):
        return self.model.predict(spectrum[np.newaxis, :])[0]
