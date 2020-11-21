import numpy as np
import matplotlib.pyplot as plt
from sloppy.raytracing import *
import inspect
from scipy.optimize import minimize, minimize_scalar
from functools import partial
from ipywidgets import Layout, IntSlider, FloatLogSlider, FloatSlider, interactive, fixed

magnitude = lambda x: 1. if x==0. else int(np.floor(np.log10(x)))

def waist_vs_l(cavfct, cavpars, Npts=500):
    elements = cavfct(**cavpars)
    sys = RaySystem(elements)

    system = sys.abcd
    x = np.linspace(0, system.Ltot, Npts)
    ws = system.compute_waists(x)

    plt.figure()
    plt.plot(x, ws)
    plt.show()

#paraxial eigenmode functions
def degeneracy_length(cavfct, parname, scanrange=1e-3, s=3, degmodenum=1):
    """
    Finds the degeneracy of the cavity cavfct as a function of parname for frequency freqs[degmodenum] and s-fold degeneracy
    """
    def get_freq(l):
        elements = cavfct(**{parname: l})
        sys = RaySystem(elements)
        system = sys.abcd
        freqs = np.concatenate(system.get_freqs(s=s))
        degIdx=1+degmodenum
        return abs(freqs[degIdx])**2 #which frequency to select!
    
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


def waists_vs_param(cavfct, parname, scanrange, N=300, degmodenum=1, s=3):
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
            freqs[i,...] = np.concatenate(system.get_freqs(s=s))

    # find the degeneracy condition: smallest s-fold transverse mode splitting with a stable mode
    degIdx=1+degmodenum
    #stable_mode_mask = [all(wpair>0) for wpair in ws]
    #subset_idx = np.argmin(np.abs(freqs[:,degIdx][stable_mode_mask]))
    #idx = np.arange(np.abs(freqs[:,degIdx]).shape[0])[stable_mode_mask][subset_idx] 
    idx = np.argmin(np.abs(freqs[:,degIdx]))
    
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
    
    magnitude = lambda x: 1. if x==0. else int(np.floor(np.log10(x)))
    
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

"""
Helper functions to find coefficients/parameters for minimum deviation of the ray
"""
coeffc2 = lambda x: 1./(2*x)
coeffc4 = lambda x: 1./(8*x**3)
coeffc6 = lambda x: 1./(16*x**5)
coeffc8 = lambda x: 5./(128*x**7)

def degeneracy_length_ray(cavfct, parname, r=0, scanrange=1e-3, degmodenum=1, La=None, Nrt=30, **kwargs):
    def get_dev(l, r, arg):
        kwargs=arg.copy()
        kwargs.update({parname: l})
        elements = cavfct(**kwargs)
        sys = RaySystem(elements)
        system = sys.abcd
        mu1, mu2 = system.q
        waist = system.waist_at(0)[0]
        mu = mu1 if degmodenum==1 else mu2
        rmu = np.linalg.norm(np.real(mu[:2]))
        mu = np.real(r*waist/rmu*mu)
        ray0 = sys.screen.eigenvectors_to_rays(mu)
        traj_hit = sys.propagate(ray0, Nrt=Nrt, at_screen=True)
        hit_scr = sys.screen.r_to_screen_coords(traj_hit[:,0,0,:])
        hit_1 = hit_scr[::3,:]
        hit_2 = hit_scr[1::3,:]
        hit_3 = hit_scr[2::3,:]
        #approximate path length described by dots through piecwise linear distance
        dev = np.linalg.norm(hit_1[1:,:]-hit_1[:-1,:], axis=1).sum() +\
                np.linalg.norm(hit_2[1:,:]-hit_2[:-1,:], axis=1).sum() +\
                np.linalg.norm(hit_3[1:,:]-hit_3[:-1,:], axis=1).sum()
        return dev
    if La is None:
        La = inspect.signature(cavfct).parameters[parname].default
    extrargs = kwargs.copy()
    res = minimize_scalar(get_dev, bounds=((1-scanrange)*La, (1+scanrange)*La), args=(r, extrargs), method='bounded')
    return res

def coefficients_interaction_factory(cavfct, degmodenum=1, coefpars = {'c4': coeffc4(5.0)}, negative=True, rmax=12., Nplt=100, Nrt=30):
    magnitude = lambda x: int(np.floor(np.log10(x)))
    
    def get_dev(r, **kwargs):
        elements = cavfct(**kwargs)
        sys = RaySystem(elements)
        system = sys.abcd
        mu1, mu2 = system.q
        waist = system.waist_at(0)[0]
        mu = mu1 if degmodenum==1 else mu2
        rmu = np.linalg.norm(np.real(mu[:2]))
        mu = np.real(r*waist/rmu*mu)
        ray0 = sys.screen.eigenvectors_to_rays(mu)
        traj_hit = sys.propagate(ray0, Nrt=Nrt, at_screen=True)
        hit_scr = sys.screen.r_to_screen_coords(traj_hit[:,0,0,:])
        hit_1 = hit_scr[::3,:]
        hit_2 = hit_scr[1::3,:]
        hit_3 = hit_scr[2::3,:]
        #approximate path length described by dots through piecwise linear distance
        dev = np.linalg.norm(hit_1[1:,:]-hit_1[:-1,:], axis=1).sum() +\
                np.linalg.norm(hit_2[1:,:]-hit_2[:-1,:], axis=1).sum() +\
                np.linalg.norm(hit_3[1:,:]-hit_3[:-1,:], axis=1).sum()
        return dev

    fig, ax = plt.subplots()
    line = ax.plot([], [])[0]
    plt.title('Deviation vs radius')
    plt.xlabel(r'$r_{in}$ [waists]')
    plt.ylabel(r'$dev$ [mm]')
    plt.show()
    
    #Because the coefficients for aspheres/waveplates get super small, there is some rescaling magic here!
    def update(scalings, **kwargs):
        rs = np.linspace(np.finfo(np.float32).eps,rmax, Nplt)
        Ls = np.zeros_like(rs)

        scargs = {}
        for k, v in kwargs.items():
            scargs[k] = v*10**scalings[k]

        for i, r in enumerate(rs):
            try:
                d = get_dev(r=r, **scargs)
            except:
                Ls[i] = np.nan
            else:
                Ls[i] = d
        line.set_xdata(rs)
        line.set_ydata(Ls)
        ax.set_xlim(rs[0], rs[-1])
        ax.set_ylim(1.1*np.nanmin(Ls), 1.1*np.nanmax(Ls))
        fig.canvas.draw_idle()

    lo = Layout(width='80%', height='30px')
    sliders = {}
    scalings = {}
    for k, v in coefpars.items():
        vs = v*10**(-magnitude(v))
        scalings[k] = magnitude(v)
        sliders[k] = FloatSlider(value=vs, min=-1.2*vs*negative, max=1.2*vs, step=1e-3, readout_format='.3e', layout=lo)

    print(scalings)
    return interactive(update, scalings=fixed(scalings), **sliders)

def get_deviation(r, degmodenum=1, **kwargs):
    elements = cavfct(**kwargs)
    sys = RaySystem(elements)
    system = sys.abcd
    mu1, mu2 = system.q
    waist = system.waist_at(0)[0]
    mu = mu1 if degmodenum==1 else mu2
    rmu = np.linalg.norm(np.real(mu[:2]))
    mu = np.real(r*waist/rmu*mu)
    ray0 = sys.screen.eigenvectors_to_rays(mu)
    traj_hit = sys.propagate(ray0, Nrt=Nrt, at_screen=True)
    hit_scr = sys.screen.r_to_screen_coords(traj_hit[:,0,0,:])
    hit_1 = hit_scr[::3,:]
    hit_2 = hit_scr[1::3,:]
    hit_3 = hit_scr[2::3,:]
    #approximate path length described by dots through piecwise linear distance
    dev = np.linalg.norm(hit_1[1:,:]-hit_1[:-1,:], axis=1).sum() +\
            np.linalg.norm(hit_2[1:,:]-hit_2[:-1,:], axis=1).sum() +\
            np.linalg.norm(hit_3[1:,:]-hit_3[:-1,:], axis=1).sum()
    return dev