#!/usr/bin/env python3
"""
ML-Enhanced FTIR Analyzer with Advanced Processing
Includes deep learning options and automated optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import pywt


class MLEnhancedFTIRAnalyzer:
    def __init__(self, base_analyzer):
        self.analyzer = base_analyzer
        self.denoised_intensity = None
        self.peaks_ml = None
        self.feature_dict = {}
        self.quality_score = None

    def adaptive_denoise(self, method='auto', preserve_peaks=True):
        if method == 'auto':
            methods=['wavelet','adaptive_savgol','combined']
            best_score=-np.inf
            for m in methods:
                d=self._apply_denoising(m,preserve_peaks)
                score=self._evaluate_denoising(self.analyzer.intensity,d)
                if score>best_score:
                    best_score, best_denoised = score,d
                    best_m=m
            self.denoised_intensity=best_denoised
            print(f"Auto-selected denoising: {best_m} (score {best_score:.3f})")
        else:
            self.denoised_intensity=self._apply_denoising(method,preserve_peaks)
        return self.denoised_intensity

    def _apply_denoising(self, method, preserve_peaks):
        data=self.analyzer.intensity.copy()
        if method=='wavelet':
            den= self._wavelet_denoise(data)
        elif method=='adaptive_savgol':
            den= self._adaptive_savgol(data)
        elif method=='combined':
            wv=self._wavelet_denoise(data)
            sg=self._adaptive_savgol(data)
            snr=self._estimate_local_snr(data)
            w=1/(1+np.exp(-snr))
            den = w*wv + (1-w)*sg
        else:
            sigma=len(data)/500
            den=gaussian_filter1d(data,sigma)
        if preserve_peaks:
            pks,_=find_peaks(data,prominence=0.01)
            for pk in pks:
                sl=slice(max(0,pk-10),min(len(data),pk+10))
                den[sl]=0.7*data[sl]+0.3*den[sl]
        return den

    def _wavelet_denoise(self,data):
        wavelet='db6'
        lvl=min(pywt.dwt_max_level(len(data),pywt.Wavelet(wavelet).dec_len),6)
        coeffs=pywt.wavedec(data,wavelet,level=lvl)
        sigma=np.median(np.abs(coeffs[-1]))/0.6745
        thr=sigma*np.sqrt(2*np.log(len(data)))
        coeffs[1:]=[pywt.threshold(c,thr*0.8,mode='soft') for c in coeffs[1:]]
        d=pywt.waverec(coeffs,wavelet)
        return d[:len(data)]

    def _adaptive_savgol(self,data):
        noise=self._estimate_local_noise(data)
        den=np.zeros_like(data)
        for i in range(len(data)):
            ln=noise[max(0,i-50):min(len(data),i+50)].mean()
            w=11 if ln<0.01 else 21 if ln<0.05 else 31
            if w%2==0: w+=1
            st=max(0,i-w//2); en=min(len(data),i+w//2+1)
            if en-st>=w:
                try:
                    den[i]=signal.savgol_filter(data[st:en],w,3)[i-st]
                except:
                    den[i]=data[i]
            else:
                den[i]=data[i]
        return den

    def _estimate_local_noise(self,data,window=50):
        noise=np.zeros_like(data)
        for i in range(len(data)):
            seg=data[max(0,i-window//2):min(len(data),i+window//2)]
            if len(seg)>5:
                d=seg-np.median(seg)
                noise[i]=1.4826*np.median(np.abs(d))
        return noise

    def _estimate_local_snr(self,data,window=100):
        power=gaussian_filter1d(data**2,window/4)
        noise=self._estimate_local_noise(data,window)
        snr=10*np.log10(power/(noise**2+1e-10))
        return np.clip(snr,-10,50)

    def _evaluate_denoising(self,orig,den):
        op,_=find_peaks(orig,prominence=0.01)
        dp,_=find_peaks(den,prominence=0.01)
        peak_score=len(set(op)&set(dp))/max(len(op),1)
        smooth=1-np.std(np.diff(den,2))/np.std(orig)
        snr_imp=1-np.std(np.diff(den))/np.std(np.diff(orig))
        return 0.4*peak_score+0.3*smooth+0.3*snr_imp

    def ml_peak_detection(self,min_prominence=0.005):
        data=self.denoised_intensity if self.denoised_intensity is not None else self.analyzer.intensity
        p1,_=find_peaks(data,prominence=min_prominence,width=3)
        scales=np.arange(5,50,2)
        cwt=signal.cwt(data,signal.ricker,scales)
        p2=signal.find_peaks_cwt(np.abs(cwt.mean(axis=0)),scales,min_snr=2)
        fd=np.gradient(data)
        p3=np.where((fd[:-1]>0)&(fd[1:]<=0))[0]
        allp=np.unique(np.concatenate([p1,p2,p3]))
        peaks=[]
        for pk in allp:
            cnt=sum(np.any(np.abs(arr-pk)<5) for arr in (p1,p2,p3))
            if 5<pk<len(data)-5 and cnt>=2:
                prom=signal.peak_prominences(data,[pk])[0][0]
                if prom<min_prominence: continue
                w=signal.peak_widths(data,[pk],rel_height=0.5)[0][0]
                half=int(w/2)
                st,ed=max(0,pk-half),min(len(data),pk+half)
                area=abs(np.trapz(data[st:ed],self.analyzer.wavenumber[st:ed]))
                lw, rw = pk-st, ed-pk
                asym=(rw-lw)/(rw+lw) if rw+lw>0 else 0
                peaks.append({
                    'position':self.analyzer.wavenumber[pk],
                    'intensity':data[pk],'prominence':prom,
                    'width':w*np.mean(np.abs(np.diff(self.analyzer.wavenumber))),
                    'area':area,'asymmetry':asym,'index':pk
                })
        self.peaks_ml=pd.DataFrame(peaks)
        return self.peaks_ml

    def classify_peaks(self,custom_groups=None):
        if self.peaks_ml is None or self.peaks_ml.empty:
            self.ml_peak_detection()
        if custom_groups is None:
            groups={
                'Al-O stretch':{'range':(400,800),'width':(20,150),'intensity':'high'},
                'Si-O stretch':{'range':(800,950),'width':(30,120),'intensity':'medium'},
                'C-O stretch':{'range':(950,1300),'width':(30,100),'intensity':'medium'},
                'CH2 bend':{'range':(1350,1500),'width':(20,80),'intensity':'low'},
                'CH2 sym':{'range':(2800,2900),'width':(20,60),'intensity':'medium'},
                'CH3 asym':{'range':(2900,3000),'width':(20,60),'intensity':'medium'},
                'O-H free':{'range':(3500,3700),'width':(50,200),'intensity':'variable'},
                'O-H H-bond':{'range':(3200,3500),'width':(100,400),'intensity':'high'}
            }
        else:
            groups=custom_groups
        cls,conf=[],[]
        for _,pk in self.peaks_ml.iterrows():
            best,score='Unassigned',0
            for name,crit in groups.items():
                s=0
                if crit['range'][0]<=pk['position']<=crit['range'][1]: s+=0.5
                if crit['width'][0]<=pk['width']<=crit['width'][1]: s+=0.3
                inten=pk['intensity']
                if crit['intensity']=='variable': s+=0.2
                elif crit['intensity']=='high' and inten>0.5: s+=0.2
                elif crit['intensity']=='medium' and 0.2<inten<0.7: s+=0.2
                elif crit['intensity']=='low' and inten<0.3: s+=0.2
                if s>score: best,score=name,s
            cls.append(best); conf.append(score)
        self.peaks_ml['classification']=cls
        self.peaks_ml['confidence']=conf
        return self.peaks_ml

    def extract_features(self,include_derivatives=True,include_moments=True):
        data=self.denoised_intensity if self.denoised_intensity is not None else self.analyzer.intensity
        feats={}
        feats['mean_intensity']=np.mean(data)
        feats['std_intensity']=np.std(data)
        feats['max_intensity']=np.max(data)
        feats['total_area']=np.trapz(data,self.analyzer.wavenumber)
        if self.peaks_ml is not None and len(self.peaks_ml)>0:
            feats['n_peaks']=len(self.peaks_ml)
            feats['total_peak_area']=self.peaks_ml['area'].sum()
            feats['mean_peak_width']=self.peaks_ml['width'].mean()
        regions={'fingerprint':(400,1500),'double_bond':(1500,1800),
                 'CH_stretch':(2800,3000),'OH_NH':(3000,3700)}
        for name,(lo,hi) in regions.items():
            mask=(self.analyzer.wavenumber>=lo)&(self.analyzer.wavenumber<=hi)
            if mask.any():
                rd=data[mask]
                feats[f'{name}_area']=np.trapz(rd)
        if include_derivatives:
            fd=np.gradient(data); sd=np.gradient(fd)
            feats['first_deriv_std']=np.std(fd)
            feats['second_deriv_std']=np.std(sd)
        if include_moments:
            from scipy import stats
            feats['skew']=stats.skew(data); feats['kurtosis']=stats.kurtosis(data)
        self.feature_dict=feats
        return feats

    def assess_spectrum_quality(self):
        scores={}
        noise=self._estimate_local_noise(self.analyzer.intensity)
        scores['snr']=min(np.mean(self.analyzer.intensity)/(np.mean(noise)+1e-10)/20,1.0)
        if self.analyzer.baseline is not None:
            scores['baseline_q']=1-min(np.var(self.analyzer.baseline)/np.var(self.analyzer.intensity),1.0)
        else:
            scores['baseline_q']=0.5
        if self.peaks_ml is not None and len(self.peaks_ml)>1:
            pos=np.sort(self.peaks_ml['position'].values)
            scores['sep']=min(np.diff(pos).min()/20,1.0)
            scores['sharp']=1-min(self.peaks_ml['width'].mean()/100,1.0)
        else:
            scores['sep']=scores['sharp']=0
        self.quality_score=np.mean(list(scores.values()))
        scores['overall']=self.quality_score
        return scores

    def create_interactive_plot(self,filename=None):
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        fig=make_subplots(rows=3,cols=2,
                          specs=[[{"colspan":2},None],[{"colspan":2},None],[{},{}]],
                          subplot_titles=["FTIR w/ ML","Denoising","Peak Distribution","Quality"])
        wn=self.analyzer.wavenumber; data=self.analyzer.intensity
        fig.add_trace(go.Scatter(x=wn,y=data,name='Original',line=dict(color='lightgray')),row=1,col=1)
        if self.denoised_intensity is not None:
            fig.add_trace(go.Scatter(x=wn,y=self.denoised_intensity,name='Denoised'),row=1,col=1)
        if self.analyzer.baseline is not None:
            fig.add_trace(go.Scatter(x=wn,y=self.analyzer.baseline,name='Baseline',line=dict(dash='dash')),row=1,col=1)
        if self.peaks_ml is not None and not self.peaks_ml.empty:
            for cls in self.peaks_ml['classification'].unique():
                df=self.peaks_ml[self.peaks_ml['classification']==cls]
                fig.add_trace(go.Scatter(x=df['position'],y=df['intensity'],mode='markers',name=cls),row=1,col=1)
            fig.add_trace(go.Histogram(x=self.peaks_ml['position'],nbinsx=20,name='Dist'),row=3,col=1)
        noise = data - (self.denoised_intensity if self.denoised_intensity is not None else data)
        fig.add_trace(go.Scatter(x=wn,y=noise,name='Noise'),row=2,col=1)
        qs=self.assess_spectrum_quality()
        fig.add_trace(go.Bar(x=list(qs.keys()),y=list(qs.values()),name='Quality'),row=3,col=2)
        fig.update_xaxes(autorange='reversed')
        fig.update_layout(height=900,title=f"FTIR ML Analysis: {self.analyzer.filename}")
        if filename:
            fig.write_html(filename)
        fig.show()
        return fig

    def optimize_parameters(self):
        best=None; best_score=-np.inf
        for method,params in [('als',{'lam':1e6,'p':0.001}),('als',{'lam':1e5,'p':0.01}),('polynomial',{'degree':5})]:
            try:
                if method=='als': self.analyzer.baseline_als(**params)
                else: self.analyzer.baseline_polynomial(**params)
                ci=self.analyzer.corrected_intensity
                var=np.var(ci) if ci is not None else 0
                scr=var
                if scr>best_score: best_score,best=(method,params)
            except: continue
        if best:
            m,p=best
            if m=='als': self.analyzer.baseline_als(**p)
            else: self.analyzer.baseline_polynomial(**p)
        print(f"Optimized baseline: {best}")
        return best

    def batch_clustering(self,analyzers,method='auto',n_clusters=None):
        feats=[]; names=[]
        for an in analyzers:
            ml=MLEnhancedFTIRAnalyzer(an); ml.adaptive_denoise()
            ml.ml_peak_detection(); ml.classify_peaks()
            feats.append(list(ml.extract_features().values())); names.append(an.filename)
        X=np.nan_to_num(np.array(feats))
        Xs=StandardScaler().fit_transform(X)
        if method=='hierarchical' or (method=='auto' and len(Xs)<10):
            Z=linkage(Xs,'ward'); clusters=fcluster(Z,n_clusters or 2,'maxclust')
        else:
            km=KMeans(n_clusters or 3,random_state=42).fit(Xs); clusters=km.labels_
        ts=TSNE(n_components=2,random_state=42).fit_transform(Xs)
        plt.figure(figsize=(8,6)); plt.scatter(ts[:,0],ts[:,1],c=clusters)
        for i,nm in enumerate(names): plt.annotate(nm,(ts[i,0],ts[i,1]))
        plt.title("t-SNE Clustering"); plt.show()
        return clusters, feats
