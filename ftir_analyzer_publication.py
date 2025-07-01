#!/usr/bin/env python3
"""
Publication-Quality FTIR Analyzer for Alucone MLD Films
Enhanced with spectrochempy for robust file handling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, sparse
from scipy.interpolate import CubicSpline
from pathlib import Path

try:
    import spectrochempy as scp
    HAS_SCP = True
except ImportError:
    HAS_SCP = False
    print("Install spectrochempy for better file format support: pip install spectrochempy")


# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'Arial',
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.width': 1,
    'ytick.minor.width': 1,
    'xtick.minor.size': 4,
    'ytick.minor.size': 4,
    'lines.linewidth': 2,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'legend.frameon': False,
    'legend.fontsize': 11
})


class FTIRAnalyzer:
    def __init__(self):
        self.wavenumber = None
        self.intensity = None
        self.filename = None
        self.baseline = None
        self.corrected_intensity = None
        self.metadata = {}

    def load_file(self, filepath):
        """Universal file loader with spectrochempy fallback"""
        filepath = Path(filepath)
        self.filename = filepath.name

        if HAS_SCP:
            try:
                dataset = scp.read(filepath)
                if hasattr(dataset, 'x') and hasattr(dataset, 'data'):
                    self.wavenumber = np.array(dataset.x.data)
                    self.intensity = np.array(dataset.data.squeeze())
                    if hasattr(dataset, 'meta'):
                        self.metadata = dataset.meta
                    # Validate and clean
                    self._validate_data()
                    print(f"Loaded {len(self.wavenumber)} points from {self.filename}")
                    return
                else:
                    raise ValueError("Invalid dataset structure")
            except Exception as e:
                print(f"Spectrochempy failed: {e}. Trying manual parsing...")

        # Fallback manual loading
        ext = filepath.suffix.lower()
        if ext in ['.jdx', '.dx']:
            self._load_jdx_manual(filepath)
        elif ext == '.csv':
            self._load_csv_manual(filepath)
        elif ext == '.txt':
            self._load_txt_manual(filepath)
        else:
            raise ValueError(f"Unsupported format: {ext}")

        self._validate_data()
        print(f"Loaded {len(self.wavenumber)} points from {self.filename}")

    def _load_jdx_manual(self, filepath):
        """Improved JDX parser"""
        content = filepath.read_text(encoding='utf-8', errors='ignore')
        # Metadata
        for line in content.splitlines():
            if line.startswith('##TITLE='):
                self.metadata['title'] = line.split('=', 1)[1].strip()
            elif line.startswith('##DATE='):
                self.metadata['date'] = line.split('=', 1)[1].strip()

        # Find data section
        data_markers = ['##XYDATA=', '##XYPOINTS=', '##PEAK TABLE=', '##DATA TABLE=']
        idx = -1
        for m in data_markers:
            idx = content.find(m)
            if idx != -1:
                break
        if idx == -1:
            raise ValueError("No data section found in JDX")
        lines = content[idx:].splitlines()[1:]

        x_data, y_data = [], []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('##'):
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) >= 2:
                try:
                    x_data.append(float(parts[0]))
                    y_data.append(float(parts[1]))
                except ValueError:
                    if len(parts) > 2 and x_data:
                        try:
                            start = float(parts[0])
                            for i, val in enumerate(parts[1:]):
                                x_data.append(start - i)
                                y_data.append(float(val))
                        except:
                            continue
        if not x_data:
            raise ValueError("No valid data in JDX")
        self.wavenumber = np.array(x_data)
        self.intensity = np.array(y_data)

    def _load_csv_manual(self, filepath):
        """Load CSV files"""
        for delim in [',', '\t', ';', ' ']:
            try:
                data = np.loadtxt(filepath, delimiter=delim, skiprows=1)
                if data.shape[1] >= 2:
                    self.wavenumber = data[:, 0]
                    self.intensity = data[:, 1]
                    return
            except:
                continue
        raise ValueError("CSV parsing failed")

    def _load_txt_manual(self, filepath):
        """Load text as CSV"""
        self._load_csv_manual(filepath)

    def _validate_data(self):
        """Clean NaNs, ensure ordering, remove duplicates"""
        mask = np.isfinite(self.wavenumber) & np.isfinite(self.intensity)
        self.wavenumber = self.wavenumber[mask]
        self.intensity = self.intensity[mask]

        if len(self.wavenumber) > 1 and self.wavenumber[0] < self.wavenumber[-1]:
            self.wavenumber = self.wavenumber[::-1]
            self.intensity = self.intensity[::-1]

        wn_u, idx = np.unique(self.wavenumber, return_index=True)
        if len(wn_u) < len(self.wavenumber):
            self.wavenumber = wn_u
            self.intensity = self.intensity[idx]

    def smooth_spectrum(self, method='savgol', **kwargs):
        if method == 'savgol':
            w = kwargs.get('window_length', 11)
            p = kwargs.get('polyorder', 3)
            self.intensity = signal.savgol_filter(self.intensity, w, p)
        elif method == 'gaussian':
            sigma = kwargs.get('sigma', 1.0)
            from scipy.ndimage import gaussian_filter1d
            self.intensity = gaussian_filter1d(self.intensity, sigma)
        elif method == 'median':
            k = kwargs.get('kernel_size', 5)
            from scipy.signal import medfilt
            self.intensity = medfilt(self.intensity, k)

    def remove_co2_artifacts(self, co2_regions=None, method='spline'):
        if co2_regions is None:
            co2_regions = [(2300, 2400), (650, 690)]
        for low, high in co2_regions:
            mask = (self.wavenumber >= low) & (self.wavenumber <= high)
            if not mask.any(): continue
            idxs = np.where(mask)[0]
            if method == 'spline':
                buf = 10
                start = max(0, idxs[0]-buf)
                end = min(len(self.wavenumber)-1, idxs[-1]+buf)
                mask_interp = ~mask
                cs = CubicSpline(self.wavenumber[mask_interp], self.intensity[mask_interp])
                self.intensity[mask] = cs(self.wavenumber[mask])
            else:
                s = self.intensity[idxs[0]-1] if idxs[0]>0 else self.intensity[idxs[0]]
                e = self.intensity[idxs[-1]+1] if idxs[-1]<len(self.intensity)-1 else self.intensity[idxs[-1]]
                self.intensity[mask] = np.linspace(s, e, mask.sum())

    def baseline_als(self, lam=1e6, p=0.001, niter=10):
        L = len(self.intensity)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        D = lam * D.dot(D.transpose())
        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)
        for _ in range(niter):
            W.setdiag(w)
            Z = W + D
            z = sparse.linalg.spsolve(Z, w * self.intensity)
            w = p*(self.intensity>z) + (1-p)*(self.intensity<z)
        self.baseline = z
        self.corrected_intensity = self.intensity - z
        return z

    def baseline_polynomial(self, degree=5, regions=None):
        from scipy.signal import find_peaks
        if regions is None:
            peaks, _ = find_peaks(self.intensity, prominence=0.05)
            mask = np.ones(len(self.intensity), bool)
            for pk in peaks:
                mask[max(0,pk-20):min(len(mask),pk+20)] = False
        else:
            mask = np.zeros(len(self.intensity), bool)
            for start, end in regions:
                mask |= (self.wavenumber>=start)&(self.wavenumber<=end)
        if mask.sum()>degree+1:
            coeffs = np.polyfit(self.wavenumber[mask], self.intensity[mask], degree)
            self.baseline = np.polyval(coeffs, self.wavenumber)
            self.corrected_intensity = self.intensity - self.baseline

    def normalize(self, method='max', region=None):
        data = self.corrected_intensity if self.corrected_intensity is not None else self.intensity
        if region:
            mask = (self.wavenumber>=region[0])&(self.wavenumber<=region[1])
            ref = data[mask]
        else:
            ref = data
        if method=='max':
            f = np.max(np.abs(ref))
        elif method=='area':
            f = np.trapz(np.abs(ref))
        elif method=='vector':
            f = np.linalg.norm(ref)
        else:
            mn, mx = np.min(ref), np.max(ref)
            self.intensity = (self.intensity-mn)/(mx-mn)
            if self.corrected_intensity is not None:
                self.corrected_intensity = (self.corrected_intensity-mn)/(mx-mn)
            return
        if f!=0:
            self.intensity /= f
            if self.corrected_intensity is not None:
                self.corrected_intensity /= f

    def find_peaks_advanced(self, prominence=0.02, width=None):
        data = self.corrected_intensity if self.corrected_intensity is not None else self.intensity
        peaks, props = signal.find_peaks(data, prominence=prominence, width=width,
                                         height=np.max(data)*0.01, distance=10)
        results=[]
        for i, pk in enumerate(peaks):
            res={'position':self.wavenumber[pk],
                 'intensity':data[pk],
                 'prominence':props['prominences'][i],
                 'width':props.get('widths',[None])[i],
                 'height':props.get('peak_heights',[None])[i],
                 'classification':self._classify_peak(self.wavenumber[pk])}
            results.append(res)
        return results

    def _classify_peak(self, pos):
        classes={
            (400,800):'Al-O stretch',
            (800,1000):'Si-O stretch',
            (1000,1300):'C-O stretch',
            (1300,1500):'CH2/CH3 deformation',
            (1500,1700):'C=C/C=O stretch',
            (2800,3000):'C-H stretch',
            (3000,3200):'N-H/=C-H stretch',
            (3200,3700):'O-H stretch'
        }
        for (l,h),name in classes.items():
            if l<=pos<=h:
                return name
        return 'Unassigned'

    def plot_publication_spectrum(self, save_path=None, **kwargs):
        figsize=kwargs.get('figsize',(8,6))
        dpi=kwargs.get('dpi',300)
        show_baseline=kwargs.get('show_baseline',True)
        fig,ax=plt.subplots(figsize=figsize,dpi=dpi)
        ax.plot(self.wavenumber,self.intensity,'k-',linewidth=2,label='Spectrum')
        if show_baseline and self.baseline is not None:
            ax.plot(self.wavenumber,self.baseline,'r--',linewidth=1.5,alpha=0.7,label='Baseline')
            offset=np.min(self.intensity)-np.max(self.corrected_intensity)-0.1
            ax.plot(self.wavenumber,self.corrected_intensity+offset,'b-',linewidth=2,label='Corrected')
        regions=[(400,800,'Al-O','#E8F4FD'),(1000,1300,'C-O','#E8F5E9'),
                 (2800,3000,'C-H','#FFF3E0'),(3200,3700,'O-H','#F3E5F5')]
        ymin,ymax=ax.get_ylim()
        for low,high,label,color in regions:
            ax.axvspan(low,high,alpha=0.2,color=color,zorder=0)
            ax.text((low+high)/2,ymax*0.95,label,ha='center',va='top',fontsize=10,alpha=0.7)
        ax.set_xlabel('Wavenumber (cm$^{-1}$)',fontsize=14)
        ax.set_ylabel(kwargs.get('ylabel','Absorbance (a.u.)'),fontsize=14)
        ax.set_xlim(4000,400)
        ax.grid(True,alpha=0.3,linestyle='--')
        ax.minorticks_on()
        if kwargs.get('title'):
            ax.set_title(kwargs['title'],fontsize=16,pad=10)
        if show_baseline:
            ax.legend(loc='upper right',fontsize=11)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path,dpi=dpi,bbox_inches='tight')
            print(f"Saved figure to {save_path}")
        plt.show()
        return fig,ax

    def export_data(self, filepath, include_all=True):
        data={'Wavenumber':self.wavenumber,'Intensity':self.intensity}
        if include_all and self.baseline is not None:
            data['Baseline']=self.baseline
            data['Corrected']=self.corrected_intensity
        df=pd.DataFrame(data)
        outfile=Path(filepath)
        if outfile.suffix=='.csv':
            df.to_csv(outfile,index=False)
        elif outfile.suffix in ('.xlsx','.xls'):
            df.to_excel(outfile,index=False)
        else:
            np.savetxt(outfile,df.values,header=','.join(df.columns),delimiter=',',comments='')
        print(f"Exported data to {filepath}")

def batch_process(folder_path, output_dir=None, pattern='*'):
    from tqdm import tqdm
    folder=Path(folder_path)
    files=list(folder.glob(pattern))
    if output_dir:
        Path(output_dir).mkdir(exist_ok=True)
    results=[]
    for fp in tqdm(files,desc="Processing"):
        try:
            analyzer=FTIRAnalyzer()
            analyzer.load_file(fp)
            analyzer.remove_co2_artifacts()
            analyzer.baseline_als()
            analyzer.normalize()
            peaks=analyzer.find_peaks_advanced()
            if output_dir:
                save_path=Path(output_dir)/f"{fp.stem}_proc.pdf"
                analyzer.plot_publication_spectrum(save_path=save_path,title=fp.stem)
            results.append({'file':fp.name,'n_peaks':len(peaks)})
        except Exception as e:
            print(f"Error {fp.name}: {e}")
    return results

if __name__=="__main__":
    print("Publication-Quality FTIR Analyzer")
    print("="*50)
    deps={'numpy':np,'scipy':signal,'matplotlib':plt,'spectrochempy':'scp' if HAS_SCP else None}
    for name,mod in deps.items():
        print(f"  {name}: {'✓' if mod else '✗'}")
    if not HAS_SCP:
        print("Install spectrochempy for better file support: pip install spectrochempy")
