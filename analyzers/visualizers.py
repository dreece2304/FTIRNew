# analyzers/visualizers.py

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import cv2

def plot_peak_heatmap(peak_data, labels=None):
    plt.figure(figsize=(10, 6))
    sns.heatmap(peak_data, xticklabels=labels, cmap="coolwarm", annot=True)
    plt.title("Peak Area Heatmap")
    plt.xlabel("Sample")
    plt.ylabel("Peak Index")
    plt.tight_layout()
    plt.show()

def interactive_spectrum(wavenumber, intensity, peaks=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wavenumber, y=intensity, mode='lines', name='Spectrum'))

    if peaks is not None:
        fig.add_trace(go.Scatter(
            x=peaks['position'],
            y=peaks['intensity'],
            mode='markers+text',
            name='Peaks',
            text=peaks['classification'],
            textposition='top center'
        ))

    fig.update_layout(
        title="FTIR Spectrum (Interactive)",
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Absorbance",
        xaxis=dict(autorange='reversed'),
        template='plotly_white'
    )
    fig.show()

def image_peak_detection(image_path):
    img = cv2.imread(image_path, 0)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours
