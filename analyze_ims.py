"""
analyze.py - Mia Panlilio 2023

Performs correlational analyses on a directory of images.
"""


import numpy as np
from sklearn.feature_selection import mutual_info_regression
from scipy import stats
from scipy import linalg
from skimage import filters, measure, io
import h5py
import re
import os
import copy
import logging
import time
import csv 
import json
import sys

import matplotlib.pyplot as plt
from matplotlib import patheffects

logging.basicConfig(format="%(asctime)s %(module)s %(funcName)s [%(levelname)s] : %(message)s")
logger = logging.getLogger('coloc')
logger.setLevel(logging.DEBUG)

class Dataset:
    def __init__(self, name=None, images=None, do_bootstrap=False, n_bootstraps=1000, bootstrap_block_size=1.0, 
            nn=3, discrete='auto',CI=0.95, threshold='otsu', threshold_kw={}, area_min=0):
        self.name = name
        self.images = images
        
        #User-defined parameters for statistical calculations
        self._do_bootstrap = do_bootstrap
        self._n_bootstraps = n_bootstraps
        self._bootstrap_block_size = bootstrap_block_size
        self._CI = CI
        self._nn = nn
        self._discrete = discrete
        self._threshold = threshold.lower()
        self._threshold_kw = threshold_kw
        self._area_min = area_min

        #Private variables to coordinate statistical calculations between methods
        self._getkey = ""

        #Attributes to store calculations
        self._mutual_info = None
        self._pearson = None
        self._spearman = None
        self._overlap = None
        self._intensity = None

    def __getitem__(self,key):
        random_sample = np.random.randint(0,len(self.images))
        if self._do_bootstrap and int(self._bootstrap_block_size)!=1:
            corner = []
            ext = []
            for dim in self.images[0][0].shape:
                ext.append(int(dim*self._bootstrap_block_size))
                corner.append(np.random.randint(0,dim-ext[-1]))
            if len(corner)==2: #2D
                im = self.images[random_sample,:,corner[0]:corner[0]+ext[0],corner[1]:corner[1]+ext[1]]
            else: #3D
                im = self.images[random_sample,corner[0]:corner[0]+ext[0],corner[1]:corner[1]+ext[1],corner[2]:corner[2]+ext[2]]
        elif type(key)!=str and key < len(self.images):
            im = self.images[key,...]
        else:
            im = self.images[random_sample,...] 
        
        #Data prep for xcalc
        if str(key).lower()=="flat" or str(key).lower()=="bootstrap":
            im = im.reshape(im.shape[0],np.prod(im.shape[1:]))
            im = im.transpose() #shape = (linear-spatial-index,channel)
        if str(key).lower()=="bootstrap":
            bs_sampler = [np.random.randint(0,len(im)) for _ in range(len(im))]
            im = im[bs_sampler,:] #shape = (linear-spatial-index,channel)
        if str(key).lower()=="overlap":
            im = np.moveaxis(im,0,-1) #shape = (z,y,x,channel) or (y,x,channel)
        return im
    
    def xcalc(self, fun, funname=''):
        Y = np.empty((3,len(self.images[0]),len(self.images[0])))
        Y.fill(np.nan)
        issquare = fun(self.dummy_int_array((16,4)),self.dummy_int_array((16,1)))[0].shape==(4,4)
        if self._do_bootstrap:
            self._getkey = 'bootstrap'
            fun('init') #adjust object according to function being calculated
            X = np.zeros((self._n_bootstraps,len(self.images[0]),len(self.images[0])))
            for idx in range(self._n_bootstraps):
                im = self[self._getkey]
                if issquare: 
                    x,_ = fun(im)
                    X[idx] = x
                else:
                    for ch in range(im.shape[-1]):
                        x,_ = fun(im[...,ch:],im[...,ch])
                        X[idx,ch,ch:] = x
                        X[idx,ch:,ch] = x
                if np.mod(idx+1,100)==0 or (idx+1)==self._n_bootstraps:
                    logger.info(f"{funname} - Bootstrapped {idx+1} of {self._n_bootstraps}")
                Y[0] = np.mean(X,axis=0)
                Y[1] = np.quantile(X,0.5*(1-self._CI),axis=0)
                Y[2] = np.quantile(X,0.5*(1+self._CI),axis=0)
        else:
            self._getkey = 'flat'
            fun('init') #adjust object according to function being calculated
            im = self[self._getkey]
            if issquare:
                x,ci = fun(im)
                Y[0] = x
                Y[1] = ci[0]
                Y[2] = ci[-1]
            else:
                for ch in range(im.shape[-1]):
                    x,ci = fun(im[...,ch:],im[...,ch])
                    Y[0,ch,ch:] = x
                    Y[0,ch:,ch] = x
                    Y[1,ch,ch:] = ci[0]
                    Y[1,ch:,ch] = ci[0]
                    Y[2,ch,ch:] = ci[-1]
                    Y[2,ch:,ch] = ci[-1]
        return Y

    @staticmethod
    def dummy_int_array(sz,maxval=1):
        A = np.array([np.random.randint(0,maxval+1) for _ in range(np.prod(sz))])
        A = np.reshape(A,sz)
        return A

    def mutual_info(self,force=False):
        if ( self._mutual_info is None or force ):
            self._mutual_info = self.xcalc(self.mutual_info_helper,'mutual_info')
        else:
            logger.warning("Mutual information has already been stored in the object. To force a recalculation, use mutual_info(force=True)")

    def _mutual_info_helper(self,X,y=None):
        if str(X).lower()=='init':
            pass
        else:
            mi = mutual_info_regression(X,np.squeeze(y),discrete_features=self._discrete,n_neighbors=self._nn)/np.log(2)
            ci = (np.nan,np.nan)
            return mi,ci

    def mutual_info_helper(self,X,y=None):
        if str(X).lower()=='init':
            pass
        else:
            maxval = np.max(X)
            bins = np.arange(0,maxval+1)
            Px = []
            for ch in range(X.shape[-1]):
                px,_ = np.histogram(X[:,ch],bins=bins,density=True)
                Px.append(px)

            MI = np.empty((X.shape[-1],X.shape[-1]))
            CI = (np.nan,np.nan)
            for ch in range(X.shape[-1]):
                px = Px[ch]
                for ch_ in range(ch,X.shape[-1]):
                    py = Px[ch_]
                    pxy,_,_ = np.histogram2d(X[:,ch],X[:,ch_],bins=[bins,bins],density=True)
                    mi = self.shannon_entropy(px) + self.shannon_entropy(py) - self.shannon_entropy(pxy)
                    MI[ch,ch_] = mi
                    MI[ch_,ch] = mi
            return MI,CI

    @staticmethod
    def shannon_entropy(px,eps=1e-23):
        px = px.flatten()
        h = -((px*np.log(np.maximum(px,eps))).sum())
        h = np.maximum(h,0) #prevents returning -0.0 entropy
        return h/np.log(2)

            

    def pearson(self, force=False):
        if ( self._pearson is None or force ):
            self._pearson = self.xcalc(self.pearson_helper,'pearson')
        else:
            logger.warning("Pearson correlation has already been stored in the object. To force a recalculation, use pearson(force=True)")
    
    def pearson_helper(self,X,y=None):
        if str(X).lower()=='init':
            pass
        else:
            r = np.corrcoef(X,rowvar=False)
            ci = (np.nan,np.nan)
            return r,ci

    def pearson_helper_legacy(self,X,y=None):
        #Deprecated: 20% slower than np.corrcoeff
        if str(X).lower()=='init':
            pass
        else:
            R = np.empty(X.shape[-1])
            CI = np.empty((2,X.shape[-1]))
            for ch in range(X.shape[-1]):
                r = stats.pearsonr(np.squeeze(X[:,ch]),np.squeeze(y))
                R[ch] = r.statistic
                if not(self._do_bootstrap): #Use built-in method to calculate CI if bootstrapping will not be performed later
                    CI[0,ch] = r.confidence_interval(self._CI).low
                    CI[1,ch] = r.confidence_interval(self._CI).high
                else:
                    CI[0,ch] = np.nan
                    CI[1,ch] = np.nan
            return R,CI

    def spearman(self, force=False):
        if ( self._spearman is None or force ):
            self._spearman = self.xcalc(self.spearman_helper,'spearman')
        else:
            logger.warning("Spearman correlation has already been stored in the object. To force a recalculation, use spearman(force=True)")

    def spearman_helper_legacy(self,X,y=None):
        #Deprecated: marginally slower than current method of np.corrcoef with scipy.stats.rankdata
        if str(X).lower()=='init':
            pass
        else:
            r = stats.spearmanr(X).statistic #for our purposes X[0]==y, therefore no need to pass y onto spearmanr
            ci = (np.nan,np.nan) #nb: no built-in method for CI available for scipy.stats.spearmanr (2023-07-06)
            return r,ci
    
    def spearman_helper(self,X,y=None):
        if str(X).lower()=='init':
            pass
        else:
            Xr = stats.rankdata(X,axis=0)
            r = np.corrcoef(Xr,rowvar=False)
            ci = (np.nan,np.nan)
            return r,ci

    def overlap(self, force=False):
        """
        Calculates the percent overlap across channels.

        Given the thresholded intensity values in channels C0 and C1, overlap is defined as: 
            overlap(0,1) = intersection(C0,C1)/C0,
        i.e. the fraction of C0 that overlaps with C1.

        Therefore, the overlap matrix is NOT GENERALLY SYMMETRIC.

        Along the diagonal the total area fraction is given:
            overlap(k,k) = Ck/(total image size)
        """

        if ( self._overlap is None or force ):
            self._overlap = self.xcalc(self.overlap_helper,'overlap')
        else:
            logger.warning("Overlap has already been stored in the object. To force a recalculation, use overlap(force=True)")

    def overlap_helper(self,X,y=None):
        if str(X).lower()=='init':
            self._getkey = 'overlap' #prevents image array from being flattened during xcalc
        else:
            L = np.empty((X.shape[-1],np.prod(X.shape[:-1])))
            for ch in range(X.shape[-1]):
                L[ch] = self.threshold(X[...,ch]).flatten()
            R = np.empty((X.shape[-1],X.shape[-1]))
            CI = (np.nan,np.nan)
            for ch in range(len(L)):
                for ch_ in range(len(L)):
                    if ch==ch_:
                        R[ch,ch_] = L[ch].tolist().count(True)*1.0/len(L[ch])
                    else:
                        R[ch,ch_] = np.logical_and(L[ch],L[ch_]).tolist().count(True)*1.0/L[ch].tolist().count(True)
            return R,CI

    def threshold(self,im):
        try:
            t = getattr(filters,f'threshold_{self._threshold}')(im,**self._threshold_kw)
        except:
            t = filters.threshold_otsu(im)
            logger.warning(f"Error encountered using threshold method: {self._threshold}. Otsu used instead. Check that the threshold method is an attribute of skimage.filters and that the parameter dictionary is correct.")
        L,n = measure.label(im >= t, return_num=True)
        for idx in range(1,n+1):
            if (L==idx).flatten().sum() < self._area_min:
                L[L==idx] = 0
        L = L>0
        return L

    def intensity(self):
        self._intensity = []
        for ch in range(self.images.shape[1]):
            im = self.images[:,ch,...]
            self._intensity.append({
                'mean' : np.mean(im),
                'median' : np.median(im),
                'min' : np.min(im),
                'max' : np.max(im),
                'sd' : np.std(im)
                 })

    def all_stats(self):
        logger.info('Calculating Pearson correlation...')
        self.pearson()
        logger.info('...done.')
        logger.info('Calculating Spearman correlation...')
        self.spearman()
        logger.info('...done.')
        logger.info('Calculating mutual information...')
        self.mutual_info()
        logger.info('...done.')
        logger.info('Calculating thresholded intensity overlap...')
        self.overlap()
        logger.info('...done.')
        logger.info('Calculating image intensity statistics...')
        self.intensity()
        logger.info('...done.')

class Reader:
    def __init__(self, imsname=None):
        self.imsname = imsname
        self.datasets = self.ls_dsets()
        logger.info(f"Highest resolution datasets found : {self.datasets}")

    def ls_dsets(self):
        h5ls = H5ls(highest_res_only=True)   
        with h5py.File(self.imsname,'r') as f:
            f.visititems(h5ls)
        return h5ls.names

    def get_dataset(self,x=0,y=0,z=0,nx=8,ny=8,nz=8,**dataset_kwargs):
        name = f"{os.path.basename(self.imsname)[:-4]}_x{x}_y{y}_z{z}"
        images = []
        with h5py.File(self.imsname,'r') as f:
            for dset in self.datasets:
                im = f[dset][z:z+nz,y:y+ny,x:x+nx]
                images.append(im)
        images = np.array([images])
        logger.info(f"images shape = {images.shape}")
        D = Dataset(name=name, images=images, **dataset_kwargs)
        return D


class H5ls:
    """
    List all datasets in an HDF5 file, by default retrieving only those at the highest resolution. 
    Based on solution from 
    https://stackoverflow.com/questions/31146036/how-do-i-traverse-a-hdf5-file-using-h5py 
    user Bremsstrahlung
    """
    def __init__(self,highest_res_only=True):
        self.names = []
        self.highest_res_only = highest_res_only
        self.max_res = 0 

    def __call__(self,name,h5obj):
        if isinstance(h5obj,h5py.Dataset) and not name in self.names:
            sz = np.prod(h5obj.shape)
            if self.highest_res_only and sz > self.max_res:
                self.names = [name]
                self.max_res = sz
            elif self.highest_res_only and sz==self.max_res:
                self.names += [name]
            elif not self.highest_res_only:
                self.names += [name]
            

class Writer:
    def __init__(self, destdir=None, stats_to_get=('pearson','spearman','mutual_info','overlap','intensity')):
        if destdir is None:
            self.destdir = os.path.join(os.getcwd(),f"results_{hex(round(1e6*time.time()))[:1:-1]}")
        else:
            self.destdir = destdir
        self.stats_to_get = stats_to_get
            
    @staticmethod
    def fmt_statname(statname,x,y,ci=None):
        if ci is None:
            s = f"{statname}_{x}_{y}"
        else:
            s = f"{statname}CI{ci:1.5f}_c{x}_c{y}"
        return s
        
    def stats_to_dict(self,dset,append_to=None):
        S = {}
        S['name'] = dset.name
        for f in self.stats_to_get:
            X = getattr(dset,f"_{f}")
            if X is None:
                continue
            elif type(X[0])==dict:
                for c,x in enumerate(X):
                    for key,val in x.items():
                        S[self.fmt_statname(f,key,c)] = x[key]
            else:
                for s in range(len(X[0])):
                    s1 = s if linalg.issymmetric(X[0]) else 0
                    for s_ in range(s1,len(X[0])):
                        S[self.fmt_statname(f,s,s_)] = X[0,s,s_]
                        S[self.fmt_statname(f,s,s_,ci=0.5*(1-dset._CI))] = X[1,s,s_]
                        S[self.fmt_statname(f,s,s_,ci=0.5*(1+dset._CI))] = X[2,s,s_]
        
        if type(append_to)==dict:
            append_to = self._convert_vals_to_lists(append_to)
            N = []
            
            if len(append_to)>0:
                for k,v in append_to.items():
                    N.append(len(append_to[k]))
                if not(np.all(np.array(N)/N[0]==1.0)):
                    logger.error(f'List sizes do not match!! N = {N}')
                
                for key,val in S.items():
                    if key not in append_to.keys():
                        v = np.empty((N[0]))
                        v.fill(np.nan)
                        append_to[key] = list(v)
                    append_to[key].append(val)
                S = append_to
        else:
            S = self._convert_vals_to_lists(S)

        return S

    
    @staticmethod
    def _convert_vals_to_lists(dictionary):
        if len(dictionary)>0:
            for key,val in dictionary.items():
                if type(val)!=list:
                    dictionary[key] = [val]
        return dictionary
    
    def save_results(self,dictionary,csvname=None):
        self.mkdest()
        if csvname is None:
            csvname = f"results_{hex(round(1e6*time.time()))[:1:-1]}.csv"
        if not os.path.isabs(csvname):
            csvname = os.path.join(self.destdir,csvname)
        S = []
        for idx in range(len(next(iter(dictionary.values())))):
            S.append({ k: v[idx] for k,v in dictionary.items() })
        fields = list(dictionary.keys())
        with open(csvname,'w',newline='') as f:
            w = csv.DictWriter(f,fieldnames=fields)
            w.writeheader()
            w.writerows(S)
        return csvname

    def save_heatmaps(self,dset,dirname=None):
        if dirname is None:
            dirname = self.destdir
        self.mkdest()

        for f in self.stats_to_get:
            X = getattr(dset,f"_{f}")
            if type(X)!=np.ndarray:
                logger.warning(f'Statistic "{f}" is not a numpy array and cannot be plotted at this time.')
            else:
                kw = self._heatmap_kwargs(f)
                self.heatmap(X[0],f,**kw)
                filename = f"{dset.name}_{f}.pdf"
                plt.savefig(os.path.join(dirname,filename))

    def mkdest(self):
        if not os.path.exists(self.destdir):
            os.makedirs(self.destdir)

    @staticmethod
    def heatmap(X,title='',vmin=None,vmax=None,cmap=None,textcolor='w',do_show=False):
        fig,ax = plt.subplots(figsize=(5,4))
        im = ax.pcolormesh(X,vmin=vmin,vmax=vmax,cmap=cmap,edgecolors='w')
        ax.set_xticks(np.arange(X.shape[1])+0.5,labels=np.arange(X.shape[1]))
        ax.set_yticks(np.arange(X.shape[0])+0.5,labels=np.arange(X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                txt = ax.text(j+0.5,i+0.5,f"{X[i,j]:.3f}",ha='center',va='center',color=textcolor)
                txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground='k')])
        ax.set_title(title)
        ax.set_xlabel('channel')
        ax.set_ylabel('channel')
        ax.spines[:].set_visible(False)
        cbar = ax.figure.colorbar(im,ax=ax)
        
        if do_show:
            plt.show()

    @staticmethod
    def _heatmap_kwargs(statname):
        if statname in ('pearson','spearman'):
            vmin = -1
            vmax = 1
            cmap = 'coolwarm'
        elif statname in ('overlap'):
            vmin = 0
            vmax = 1
            cmap = 'coolwarm'
        else:
            vmin = None
            vmax = None
            cmap = None
        kw = {'vmin' : vmin, 'vmax' : vmax, 'cmap' : cmap} 
        return kw


def main(imsname, x=0, y=0, z=0, nx=8, ny=8, nz=8, 
        destdir=None, stats_to_get=('pearson','spearman','mutual_info','overlap','intensity'),
        do_bootstrap=True, n_bootstraps=200, bootstrap_block_size=0.25, 
        nn=3, discrete='auto',CI=0.95, threshold='otsu', threshold_kw={}, area_min=0):
    
    reader = Reader(imsname=imsname)
    writer = Writer(destdir=destdir, stats_to_get=stats_to_get)
    datastore = None
    csv_files = [] 
    
    x,y,z = (np.array([x]),np.array([y]),np.array([z]))
    nx,ny,nz = (np.array([nx]),np.array([ny]),np.array([nz]))

    if not len(x)==len(y)==len(z):
        raise IndexError('x,y,z arguments all must be of the same length.')
    
    if not len(nx)==len(ny)==len(nz)==len(x):
        if len(nx)==len(ny)==len(nz)==1:
            nx = np.array([nx for _ in len(x)])
            ny = np.array([ny for _ in len(x)])
            nz = np.array([nz for _ in len(z)])
        else:
            raise IndexError('nx,ny,nz must all be of the same length L, where L is len(x) or 1.')

    logger.info(f"\n imsname : {imsname}\n destdir : {writer.destdir}\n stats_to_retrieve : {stats_to_get}"+
                f"\n do_bootstrap : {do_bootstrap}\n n_bootstraps : {n_bootstraps}\n bootstrap_block_size : {bootstrap_block_size}"+
                f"\n CI : {CI}\n threshold : {threshold}\n threshold_kw : {threshold_kw}\n area_min : {area_min}")
    counter = 0
    for x_,y_,z_,nx_,ny_,nz_ in zip(x,y,z,nx,ny,nz):
        dset = reader.get_dataset(x=x_, y=y_, z=z_, nx=nx_, ny=ny_, nz=nz_,
                                do_bootstrap=do_bootstrap, n_bootstraps=n_bootstraps, bootstrap_block_size=bootstrap_block_size,
                                nn=nn, discrete=discrete, CI=CI, threshold=threshold, threshold_kw=threshold_kw, area_min=area_min)
        logger.info(f"{counter+1}/{len(x)} ROI x,y,z = ({x_},{y_},{z_}). PROCESSING...")
        for st in stats_to_get:
            getattr(dset,st.lower())()
        datastore = writer.stats_to_dict(dset, append_to=datastore)
        writer.save_heatmaps(dset)
        csvname = writer.save_results(datastore)
        csv_files.append(csvname)
        logger.info(f"{counter+1}/{len(x)} ROI x,y,z = ({x_},{y_},{z_}). DONE.")
        counter += 1
    if len(csv_files)>1:
        for name in csv_files[:-1]:
            os.remove(name)

def test(key=None,show_dummy_data=False):
    #Generate a dummy dataset with known statistics
    black = np.zeros((256,256))
    white = np.ones((256,256))
    bw = np.concatenate((black,white),axis=0)
    wb = np.concatenate((white,black),axis=0)
    images = []
    images.append(np.concatenate((bw,bw),axis=1))
    images.append(np.concatenate((wb,wb),axis=1))
    images.append(np.concatenate((bw.T,wb.T),axis=0))
    images.append(np.concatenate((wb.T,bw.T),axis=0))
    images = np.array(images)

    if show_dummy_data:
        _,ax = plt.subplots(2,2)
        ax[0][0].imshow(images[0])
        ax[0][1].imshow(images[1])
        ax[1][0].imshow(images[2])
        ax[1][1].imshow(images[3])
        plt.show()

    #Load into Dataset class and print statistics
    D = Dataset(name='test',images=np.expand_dims(images,0),nn=16,discrete=[True for ch in range(images.shape[1])])
    D.all_stats()
    print('spearman :\n',D._spearman[0])
    print('pearson :\n',D._pearson[0])
    print('mutual info :\n',D._mutual_info[0])
    print('overlap :\n',D._overlap[0])
    print('intensity :\n',D._intensity)

    if str(key).lower()=='bootstrap':
        D = Dataset(name='test',images=np.expand_dims(images,0),nn=5,discrete=[True for ch in range(images.shape[1])],
                do_bootstrap=True,n_bootstraps=50,bootstrap_block_size=0.55)
        D.all_stats()
        print('spearman :\n',D._spearman[0])
        print('pearson :\n',D._pearson[0])
        print('mutual info :\n',D._mutual_info[0])
        print('overlap :\n',D._overlap[0])
        print('intensity:\n',D._intensity)
    
    elif str(key).lower()=='write':
        writer = Writer()
        S = {}
        for idx in range(3):
            D.name = f"test{idx}"
            S = writer.stats_to_dict(D,append_to=S)
        print('dictionary writer tested :\n',S)
        S = writer.save_results(S)
        print('csv conversion test saved to {os.path.join(self.destdir,csvname)}')
        writer.save_heatmaps(D)
    
    elif str(key).lower()=='savedummy':
        fig,ax = plt.subplots(1,4,figsize=(10,3))
        for i in range(4):
            ax[i].imshow(images[i])
            ax[i].set_title(f"channel{i}")
        fig.tight_layout()
        plt.savefig(os.path.join(os.getcwd(),"dummydata.png"))

if __name__=="__main__":
    arg1 = sys.argv[1]
    if re.search(".json$",arg1):
        with open(sys.argv[1]) as f:
            params = json.load(f)
        main(**params)
    else:
        raise TypeError('First argument must be a JSON configuration file.')
