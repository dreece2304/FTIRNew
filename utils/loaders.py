# utils/loaders.py

import pandas as pd
import numpy as np

try:
    import rampy
except ImportError:
    rampy = None

try:
    import skspec
    from skspec import Spectrum
except ImportError:
    skspec = None

def load_rampy(filepath):
    if rampy is None:
        raise ImportError("RaMPy is not installed.")
    data = rampy.load_ir_file(filepath)
    return np.array(data['x']), np.array(data['y'])

def load_skspectra(filepath):
    if skspec is None:
        raise ImportError("scikit-spectra is not installed.")
    spec = Spectrum.read_csv(filepath)
    return np.array(spec.wavenumbers), np.array(spec.intensity)

def load_csv_fallback(filepath):
    df = pd.read_csv(filepath)
    wn = df.iloc[:, 0].values
    intensity = df.iloc[:, 1].values
    return wn, intensity
