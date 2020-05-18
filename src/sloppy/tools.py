import numpy as np
import matplotlib.pyplot as plt
from sloppy.raytracing import *
import inspect
from scipy.optimize import minimize, minimize_scalar
from functools import partial
from ipywidgets import Layout, IntSlider, FloatLogSlider, FloatSlider, interactive, fixed

#paraxial eigenmode functions
def degeneracy_length(cavfct, parname, scanrange=1e-3, s=3):
    """
    Finds the degeneracy of the cavity cavfct as a function of parname for frequency freqs[which] (which=2 for s=3) #TODO refine for arbitrary s and transverse mode
    """
    def get_freq(l):
        elements = cavfct(**{parname: l})
        sys = RaySystem(elements)
        system = sys.abcd
        freqs = np.concatenate(system.get_freqs(s=s))
        return abs(freqs[2])**2 #which frequency to select!
    
    La = inspect.signature(cavfct).parameters[parname].default
    res = minimize_scalar(get_freq, bounds=((1-scanrange)*La, (1+scanrange)*La), method='bounded')
    return res


def cavity_parameter_interaction_factory(cavfct, parname, scanrange, N = 300):
    fig, ax = plt.subplots(ncols=2, figsize=(8,4), sharex=True)
    lws = [ax[0].plot([0, 1], [0, 1])[0] for i in range(2)]
    ax[0].set_ylabel('um')
    lfs = [ax[1].plot([0, 1], [0, 1])[0] for i in range(4)]
    ax[1].set_ylabel('MHz')
    axh1 = ax[1].axhline(0, color='grey')
    axv0 = ax[0].axvline(0, color='grey')
    axv1 = ax[1].axvline(0, color='grey')
    plt.show()
    
    magnitude = lambda x: int(np.floor(np.log10(x)))
    
    def update_waists_vs_params(cavfct, parname, scanrange, N = 300, **kwargs):
        stab = lambda m: abs(0.5*np.trace(m))<1

        La = kwargs[parname]
        Las = La + np.linspace(-scanrange*La, scanrange*La, N)
        ms = np.zeros((N))
        ws = np.zeros((N,2))
        freqs = np.zeros((N,4))
        for i, l in enumerate(Las):
            pardct = kwargs.copy()
            pardct.update({parname: l})
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
                #ft, fs3 = system.get_freqs()
                #fsr = system.fsr
                #freqs[i,...] = np.concatenate((ft, np.mod(3*ft, fsr)))
                idx = np.argmin(np.abs(freqs[:,2]))
        for i in range(2):
            lws[i].set_data(np.stack((Las, ws[:,i]*1e3), axis=0))
        for i in range(4):
            lfs[i].set_data(np.stack((Las, freqs[:,i]*1e-6), axis=0))
        axv0.set_xdata([Las[idx], Las[idx]])
        axv1.set_xdata([Las[idx], Las[idx]])
        Ldeg = Las[idx]
        ax[0].set_xlim(Las[0], Las[-1])
        ax[0].set_ylim(1.1*np.nanmin(ws*1e3), 1.1*np.nanmax(ws*1e3))
        ax[1].set_ylim(1.1*np.nanmin(freqs*1e-6), 1.1*np.nanmax(freqs*1e-6))
        fig.canvas.draw_idle()

    lo = Layout(width='80%', height='30px')
    sliders = {v.name: FloatSlider(value=v.default, min=v.default*0.5, max=v.default*1.5, step=10**(magnitude(v.default)-2), readout_format='.2e', layout=lo) for v in inspect.signature(cavfct).parameters.values()}
    sliders.update({'scanrange': FloatLogSlider(value=scanrange, min=-3, max=1, step=0.5, layout=lo)})
    return interactive(update_waists_vs_params, cavfct=fixed(cavfct), parname = fixed('lens_dist'), N=fixed(N), **sliders)

def waists_vs_param(cavfct, parname, scanrange, N = 300):
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

    idx = np.argmin(np.abs(freqs[:,2]))
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

def rays_interaction_factory(cavfct, parname, scanrange=1e-2, rmax=6.):
    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(11,6.6), sharex='row', sharey='row')
    ms = 8.
    lines = [a.scatter([], [], c=[], cmap='jet', s=ms) for a in ax.flatten()]
    ax[0,0].set_title('Raytracing')
    ax[0,1].set_title('ABCD')
    ax[0,2].set_title('ABCD (Extracted)')
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')

    res = degeneracy_length(cavfct, parname, scanrange)
    Ldeg = res.x

    def makeidx(hit_m):
        return np.arange(hit_m.shape[0])
    
    magnitude = lambda x: int(np.floor(np.log10(x)))
    
    def update(ar=0.0, br=0.0, ap=0., bp=0., Nrt=500, **kwargs):
        #kwargs.update({parname: Ldeg+dl*1e-1})
        elements = cavfct(**kwargs)
        sys = RaySystem(elements)
        system = sys.abcd
        mu1, mu2 = system.q
        waist = system.waist_at(0)[0] #fix for now since waist is not at 0 anymore!
        rmu1 = np.linalg.norm(np.real(mu1[:2]))
        rmu2 = np.linalg.norm(np.real(mu2[:2]))
        mu = np.real(ar*waist/rmu1*mu1*np.exp(1j*ap) + br*waist/rmu2*mu2*np.exp(1j*bp))
        
        ray0 = sys.screen.eigenvectors_to_rays(mu)
 
        traj_hit = sys.propagate(ray0, Nrt=Nrt, at_screen=True)
        hit_scr = sys.screen.r_to_screen_coords(traj_hit[:,0,0,:])
        hit_scrs = sys.screen.s_to_screen_coords(traj_hit[:,1,0,:])

        lines[0].set_offsets(hit_scr[:,:])
        lines[0].set_array(makeidx(hit_scr))
        lines[3].set_offsets(hit_scrs[:,:])
        lines[3].set_array(makeidx(hit_scrs))
        
        hit_m = propagate_ABCD(mu, system.abcd_rt, Nrt=Nrt)
        lines[1].set_offsets(hit_m[:,0:2])
        lines[1].set_array(makeidx(hit_m))
        lines[4].set_offsets(hit_m[:,2:])
        lines[4].set_array(makeidx(hit_m))
        
        #abcd_fd = sys.extract_ABCD(epsr=1e-3, epss=1e-3, Nrt=1) 
        #hit_fd = propagate_ABCD(mu, abcd_fd, Nrt=Nrt)
        #lines[2].set_offsets(hit_fd[:,0:2])
        #lines[2].set_array(makeidx(hit_fd))
        #lines[5].set_offsets(hit_fd[:,2:])
        #lines[5].set_array(makeidx(hit_fd))
        
        for l in lines:
            l.set_clim(vmin=0, vmax=Nrt)
            
        ax[0,0].set_xlim(1.5*np.nanmin(hit_scr[:,0]), 1.5*np.nanmax(hit_scr[:,0]))
        ax[0,0].set_ylim(1.5*np.nanmin(hit_scr[:,1]), 1.5*np.nanmax(hit_scr[:,1]))
        ax[1,0].set_xlim(1.5*np.nanmin(hit_scrs[:,0]), 1.5*np.nanmax(hit_scrs[:,0]))
        ax[1,0].set_ylim(1.5*np.nanmin(hit_scrs[:,1]), 1.5*np.nanmax(hit_scrs[:,1]))
        fig.canvas.draw_idle()

    lo = Layout(width='80%', height='30px')

    sliders = {v.name: FloatSlider(value=v.default if v.name!=parname else Ldeg, min=v.default*0.1, max=v.default*1.5, step=10**(magnitude(v.default)-3), readout_format='.2e', layout=lo) for v in inspect.signature(cavfct).parameters.values()}
    ars = FloatSlider(value=0.1, min=0., max=rmax, step=1e-2, readout_format='.3f', layout=lo)
    brs = FloatSlider(value=0.1, min=0., max=rmax, step=1e-2, readout_format='.3f', layout=lo)
    aps = FloatSlider(value=0., min=0., max=1*np.pi, step=1e-2, readout_format='.3f', layout=lo)
    bps = FloatSlider(value=0., min=0., max=1*np.pi, step=1e-2, readout_format='.3f', layout=lo)
    Nrts = IntSlider(value=100, min=100, max=2000, step=100, layout=lo)
    raysliders = {'ar': ars, 'br': brs, 'ap': aps, 'bp': bps, 'Nrt': Nrts}
    sliders.update(**raysliders)
    return interactive(update, **sliders)
