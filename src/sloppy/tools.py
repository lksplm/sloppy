import numpy as np
import matplotlib.pyplot as plt
from sloppy.raytracing import *
import inspect
from scipy.optimize import minimize, minimize_scalar
from functools import partial

#paraxial eigenmode functions
def waists_vs_param(cavfct, parname, scanrange, N=300, degmodenum=1):
    stab = lambda m: abs(0.5*np.trace(m))<1
    La = inspect.signature(cavfct).parameters[parname].default
    Las = La + np.linspace(-scanrange*La, scanrange*La, N)
    ms = np.zeros((N))
    ws = np.zeros((N,2))
    freqs = np.zeros((N,4))
    for i, l in enumerate(Las):
        pardct = {parname: l}
        sys = RaySystem( cavfct(**pardct) )

        try:
            system = sys.abcd
            w = system.waist_at(0)
        except:
            ws[i,:] = np.nan
            ms[i] = np.nan
            freqs[i,...] = np.nan
        else:
            ms[i] = stab(system.abcd_rt)
            ws[i,:] = np.sort(w)
            freqs[i,...] = np.concatenate(system.get_freqs())

    # find the degeneracy condition: smallest s-fold transverse mode splitting with a stable mode
    degIdx=1+degmodenum
    stable_mode_mask = [all(wpair>0) for wpair in ws]
    subset_idx = np.argmin(np.abs(freqs[:,degIdx][stable_mode_mask]))
    idx = np.arange(np.abs(freqs[:,degIdx]).shape[0])[stable_mode_mask][subset_idx] 
    
    g, ax = plt.subplots(ncols=2, figsize=(8,4))
    ax[0].plot(Las, ws*1e3)
    ax[0].set_ylabel('um')
    ax[1].plot(Las, freqs*1e-6)
    ax[1].set_ylabel('MHz')
    ax[1].axhline(0, color='grey')
    ax[0].axvline(Las[idx], color='grey')
    ax[1].axvline(Las[idx], color='grey')
    plt.show()
    Ldeg = Las[idx]
    return Ldeg

def degeneracy_length(cavfct, parname, scanrange=1e-3, degmodenum=1):
    def get_freq(l):
        elements = cavfct(**{parname: l})
        sys = RaySystem(elements)
        system = sys.abcd
        freqs = np.concatenate(system.get_freqs())
        degIdx=1+degmodenum
        return abs(freqs[degIdx])**2 #which frequency to select!
    
    La = inspect.signature(cavfct).parameters[parname].default
    res = minimize_scalar(get_freq, bounds=((1-scanrange)*La, (1+scanrange)*La), method='bounded')
    return res