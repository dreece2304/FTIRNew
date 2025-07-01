#!/usr/bin/env python3
"""
scripts/FTIRComplex.py

Main entry point for the FTIR analysis pipeline.
Prompts the user to select a file, choose a processing mode, and then runs:
  • Preprocessing (CO₂ removal, baseline, normalize)
  • ML peak detection & classification
  • Autoencoder denoising
  • Publication & interactive plotting
  • Data export
"""

import sys
from pathlib import Path

# === Ensure project root (parent of scripts/) is on sys.path ===
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# ============================================================

# Module imports
import ftir_analyzer_publication
import ftir_ml_enhanced
from analyzers import deep_spectral_analyzer
import analyzers.visualizers
import utils.loaders
import scripts.select_file_dialog

# Aliases for convenience
FTIRAnalyzer = ftir_analyzer_publication.FTIRAnalyzer
MLEnhancedFTIRAnalyzer = ftir_ml_enhanced.MLEnhancedFTIRAnalyzer
create_ml_publication_plot = ftir_ml_enhanced.create_ml_publication_plot
DeepSpectralDenoiser = deep_spectral_analyzer.DeepSpectralDenoiser
interactive_spectrum = analyzers.visualizers.interactive_spectrum
load_rampy = utils.loaders.load_rampy
load_skspectra = utils.loaders.load_skspectra
load_csv_fallback = utils.loaders.load_csv_fallback
get_filepath_from_dialog = scripts.select_file_dialog.get_filepath_from_dialog


def analyze_single_file(filepath):
    print(f"\nLoading {filepath}...")
    analyzer = FTIRAnalyzer()

    try:
        analyzer.load_file(filepath)
    except Exception as e:
        print(f"Default loader failed: {e}")
        ext = Path(filepath).suffix.lower()
        if ext == '.csv':
            wn, intensity = load_csv_fallback(filepath)
        elif ext == '.jdx':
            wn, intensity = load_rampy(filepath)
        else:
            wn, intensity = load_skspectra(filepath)
        analyzer.wavenumber, analyzer.intensity = wn, intensity

    print("Preprocessing...")
    analyzer.remove_co2_artifacts()
    analyzer.baseline_als()
    analyzer.normalize()

    print("Running ML analysis...")
    ml_analyzer = MLEnhancedFTIRAnalyzer(analyzer)
    ml_analyzer.adaptive_denoise(method='combined')
    ml_analyzer.ml_peak_detection()
    ml_analyzer.classify_peaks()

    print("Applying deep learning denoising...")
    autoencoder = DeepSpectralDenoiser(input_dim=len(analyzer.intensity))
    analyzer.intensity = autoencoder.denoise(analyzer.intensity)

    print("Creating figures and exporting data...")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    analyzer.plot_publication_spectrum(
        save_path=output_dir / f"{Path(filepath).stem}_standard.pdf",
        title=f"FTIR: {Path(filepath).stem}"
    )

    create_ml_publication_plot(
        ml_analyzer,
        save_path=output_dir / f"{Path(filepath).stem}_ml_enhanced.pdf"
    )

    ml_analyzer.create_interactive_plot(
        filename=str(output_dir / f"{Path(filepath).stem}_interactive.html")
    )

    interactive_spectrum(analyzer.wavenumber, analyzer.intensity, ml_analyzer.peaks_ml)

    analyzer.export_data(output_dir / f"{Path(filepath).stem}_processed.csv")


def main():
    # Prompt user to select a file
    filepath = get_filepath_from_dialog()
    if not filepath:
        print("No file selected. Exiting.")
        return

    # Choose processing mode
    print("Choose processing mode:")
    print("1 - Full pipeline (preprocess, ML, autoencoder, plots)")
    print("2 - Preprocess only (baseline, normalize, export)")
    choice = input("Enter choice [1/2]: ").strip()

    if choice == '2':
        print(f"\nLoading {filepath} for preprocess only...")
        analyzer = FTIRAnalyzer()
        analyzer.load_file(filepath)
        analyzer.remove_co2_artifacts()
        analyzer.baseline_als()
        analyzer.normalize()

        print("Exporting preprocessed data...")
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        analyzer.export_data(output_dir / f"{Path(filepath).stem}_preprocessed.csv")
    else:
        analyze_single_file(filepath)


if __name__ == "__main__":
    main()
