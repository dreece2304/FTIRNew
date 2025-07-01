# scripts/train_autoencoder.py

#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from analyzers.deep_spectral_analyzer import DeepSpectralDenoiser
from ftir_analyzer_publication import FTIRAnalyzer

def load_spectra_from_folder(folder_path, file_ext=".csv", max_files=None):
    folder = Path(folder_path)
    files = list(folder.glob(f"*{file_ext}"))
    if max_files:
        files = files[:max_files]

    spectra = []
    for file in files:
        try:
            analyzer = FTIRAnalyzer()
            analyzer.load_file(file)
            analyzer.remove_co2_artifacts()
            analyzer.baseline_als()
            analyzer.normalize()
            spectra.append(analyzer.intensity)
        except Exception as e:
            print(f"Failed to load {file.name}: {e}")
    return np.array(spectra)

def main():
    print("Training FTIR Autoencoder...")

    folder_path = "data/ToProcess"  # adjust as needed
    file_ext = ".csv"
    max_files = 100

    spectra = load_spectra_from_folder(folder_path, file_ext=file_ext, max_files=max_files)
    print(f"Loaded {spectra.shape[0]} spectra with length {spectra.shape[1]}")

    X_train, X_val = train_test_split(spectra, test_size=0.1, random_state=42)

    autoencoder = DeepSpectralDenoiser(input_dim=spectra.shape[1])

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint("models/autoencoder.h5", save_best_only=True)
    ]

    autoencoder.model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=100,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )

    print("Training complete. Model saved to models/autoencoder.h5")

if __name__ == "__main__":
    main()
