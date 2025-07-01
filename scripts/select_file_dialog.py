# scripts/select_file_dialog.py

def get_filepath_from_dialog():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="Select a FTIR spectrum file",
        filetypes=[
            ("JDX files", "*.jdx"),
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]
    )
    return filepath if filepath else None
