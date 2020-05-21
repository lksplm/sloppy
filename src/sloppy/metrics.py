"""
Metrics and tools to measure/visualize/optimize cavity abberations
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sloppy.raytracing import *
import inspect
from scipy.optimize import minimize, minimize_scalar
from functools import partial
from ipywidgets import Layout, IntSlider, FloatLogSlider, FloatSlider, interactive, fixed
from joblib import Parallel, delayed
from collections import OrderedDict

magnitude = lambda x: 1. if x==0. else int(np.floor(np.log10(x)))

coeffc2 = lambda x: 1./(2*x)
coeffc4 = lambda x: 1./(8*x**3)
coeffc6 = lambda x: 1./(16*x**5)
coeffc8 = lambda x: 5./(128*x**7)

def metric_dev(cavfct, cavpars, r, degmodenum=1, Nrt=30, **kwargs):
    """
    Metric 'deviation' tries to measure length of the path traced out by ray intersections on the screen (piecwise linear distance between hits)
    Args:
        cafct (fct): cavity function
        cavpars (dict): cavity paramters
        r (float): input radius in terms of waists (scales paraxial eigenvectors)
        degmodenum (int): 1 or 2, wich paraxial eigenvector to pick
        Nrt (int): number of roundtrips tp propagate
    """
    elements = cavfct(**cavpars)
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
    #clean this up, make general for s-fold
    dev = np.linalg.norm(hit_1[1:,:]-hit_1[:-1,:], axis=1).sum() +\
            np.linalg.norm(hit_2[1:,:]-hit_2[:-1,:], axis=1).sum() +\
            np.linalg.norm(hit_3[1:,:]-hit_3[:-1,:], axis=1).sum()
    return dev

def metric_var(cavfct, cavpars, r, degmodenum=1, Nrt=30, **kwargs):
    """
    Metric 'var' computes standard deviation of ray intersections on the screen
    Args:
        cafct (fct): cavity function
        cavpars (dict): cavity paramters
        r (float): input radius in terms of waists (scales paraxial eigenvectors)
        degmodenum (int): 1 or 2, wich paraxial eigenvector to pick
        Nrt (int): number of roundtrips tp propagate
    """
    elements = cavfct(**cavpars)
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
    hit_1 = hit_scr[3::3,:]
    hit_2 = hit_scr[1::3,:]
    hit_3 = hit_scr[2::3,:]
    #approximate path length described by dots through piecwise linear distance
    #clean this up, make general for s-fold
    dev = np.std(hit_1, axis=0).sum() + np.std(hit_2, axis=0).sum() + np.std(hit_3, axis=0).sum()
    return dev

def metric_opl(cavfct, cavpars, r, degmodenum=1, **kwargs):
    """
    Metric optical path length computes (optical) distance traversed by rays upon three roundtrips
    Args:
        cafct (fct): cavity function
        cavpars (dict): cavity paramters
        r (float): input radius in terms of waists (scales paraxial eigenvectors)
        degmodenum (int): 1 or 2, wich paraxial eigenvector to pick
    """
    elements = cavfct(**cavpars)
    sys = RaySystem(elements)
    system = sys.abcd
    mu1, mu2 = system.q
    waist = system.waist_at(0)[0]
    mu = mu1 if degmodenum==1 else mu2
    rmu = np.linalg.norm(np.real(mu[:2]))
    mu = np.real(r*waist/rmu*mu)
    ray0 = sys.screen.eigenvectors_to_rays(mu)
    traj_hit = sys.propagate(ray0, Nrt=3)
    ls = np.linalg.norm(traj_hit[1:,0,0,:]-traj_hit[:-1,0,0,:], axis=1)
    ns = np.array(system.nlist[:-1]*3) #TODO: generalize for s!=3
    ols = ls*ns #get optical path length by multiplying with ior
    opl = ols.sum()
    return opl

def metric_over_r_serial(metric, cavfct, cavpars, rmax=10., Nplt=100, degmodenum=1, **kwargs):
    """
    Compute a metric function for an array of radii
    Args:
        metric (func): metric function to evaluate, must return scalar
        cafct (fct): cavity function
        cavpars (dict): cavity paramters
        rmax (float): maximal radius
        Nplt (int): number of points
        degmodenum (int): 1 or 2, wich paraxial eigenvector to pick
    """
    rs = np.linspace(np.finfo(np.float32).eps, rmax, Nplt)
    Ls = np.zeros_like(rs)

    for i, r in enumerate(rs):
        try:
            d = metric(cavfct=cavfct, cavpars=cavpars, r=r, degmodenum=degmodenum, **kwargs)
        except:
            Ls[i] = np.nan
        else:
            Ls[i] = d
    return rs, Ls

def metric_over_r(metric, cavfct, cavpars, rmax=10., Nplt=100, degmodenum=1, **kwargs):
    """
    Compute a metric function for an array of radii, parallel version
    Args:
        metric (func): metric function to evaluate, must return scalar
        cafct (fct): cavity function
        cavpars (dict): cavity paramters
        rmax (float): maximal radius
        Nplt (int): number of points
        degmodenum (int): 1 or 2, wich paraxial eigenvector to pick
    """
    rs = np.linspace(np.finfo(np.float32).eps, rmax, Nplt)
    
    def get_m(r, metric, cavfct, cavpars, degmodenum):
        try:
            d = metric(cavfct=cavfct, cavpars=cavpars, r=r, degmodenum=degmodenum)
        except:
            return np.nan
        else:
            return d
    fun = partial(get_m, metric=metric, cavfct=cavfct, cavpars=cavpars, degmodenum=degmodenum)
    ms = Parallel(n_jobs=-1)(delayed(fun)(r) for r in rs)
    return rs, np.array(ms)
    
def residual_motion(cavfct, cavpars, ar=2., br=0.1, ap=0., bp=0., s=1, Nrt=100, degmodenum=1, **kwargs):
    """
    Compute motion in the screen plane through propagation of scaled paraxial (abcd) eigenvectors 
    Args:
        cafct (fct): cavity function
        cavpars (dict): cavity paramters
        ar/br (float): radius in units of waists for both eigenvectors
        ap/bp (float): phase for both eigenvectors
        s (int): subset of points to plot; s=1 plots all, s=3 only every third
    """
    elements = cavfct(**cavpars)
    sys = RaySystem(elements)
    system = sys.abcd
    mu1, mu2 = system.q
    mu1, mu2 = (mu1, mu2) if degmodenum==1 else (mu2, mu1)
    waist = system.waist_at(0)[0]
    rmu1 = np.linalg.norm(np.real(mu1[:2]))
    rmu2 = np.linalg.norm(np.real(mu2[:2]))
    mu = np.real(ar*waist/rmu1*mu1*np.exp(1j*ap) + br*waist/rmu2*mu2*np.exp(1j*bp))
        
    ray0 = sys.screen.eigenvectors_to_rays(mu)
    traj_hit = sys.propagate(ray0, Nrt=Nrt, at_screen=True)
    hit_scr = sys.screen.r_to_screen_coords(traj_hit[:,0,0,:])
    #hit_scrs = sys.screen.s_to_screen_coords(traj_hit[:,1,0,:])
    return hit_scr[slice(0, -1, s), :]

def plot_residual_motion(cavfct, cavpars, ar=2., br=0.1, ap=0., bp=0., s=1, Nrt=100, degmodenum=1, ax=None, ms=6., **kwargs):
    """Plotting helper fct"""
    makeidx = lambda ar: np.arange(ar.shape[0])
    
    if ax is None:
        fig, ax = plt.subplots()
    hit = residual_motion(cavfct=cavfct, cavpars=cavpars, ar=ar, br=br, ap=ap, bp=bp, s=s, Nrt=Nrt, degmodenum=degmodenum, **kwargs)
    ax.scatter(hit[:,0], hit[:,1], c=makeidx(hit), cmap='jet', s=ms)
    if ax is None:
        plt.xlabel(r'$x$ [mm]')
        plt.ylabel(r'$y$ [mm]')
        plt.show()
    
def residual_motion_mult(cavfct, cavpars, ar=np.arange(1,10,2), br=0.1, ap=0., bp=0., s=1, Nrt=100, degmodenum=1, **kwargs):
    """
    Compute motion in the screen plane through propagation of scaled paraxial (abcd) eigenvectors.
    Version that takes an array of input radii ar for better visualisation
    Args:
        cafct (fct): cavity function
        cavpars (dict): cavity paramters
        ar/br (float): radius in units of waists for both eigenvectors
        ap/bp (float): phase for both eigenvectors
        s (int): subset of points to plot; s=1 plots all, s=3 only every third
    """
    elements = cavfct(**cavpars)
    sys = RaySystem(elements)
    system = sys.abcd
    mu1, mu2 = system.q
    mu1, mu2 = (mu1, mu2) if degmodenum==1 else (mu2, mu1)
    waist = system.waist_at(0)[0]
    rmu1 = np.linalg.norm(np.real(mu1[:2]))
    rmu2 = np.linalg.norm(np.real(mu2[:2]))
    mu = np.real(ar[:,None]*waist/rmu1*mu1*np.exp(1j*ap) + br*waist/rmu2*mu2*np.exp(1j*bp))
        
    ray0 = sys.screen.eigenvectors_to_rays(mu)
    traj_hit = sys.propagate(ray0, Nrt=Nrt, at_screen=True)
    traj_hit = traj_hit[:,0,:,:]
    shp = traj_hit.shape
    hit_scr = sys.screen.r_to_screen_coords(traj_hit.reshape((-1,3))).reshape((shp[0], shp[1], 2))
    #hit_scrs = sys.screen.s_to_screen_coords(traj_hit[:,1,0,:])
    return hit_scr[slice(0, -1, s),...]

def plot_residual_motion_mult(cavfct, cavpars, ar=np.arange(1,10,2), br=0.0, ap=0., bp=0., s=1, Nrt=100, degmodenum=1, ax=None, ms=6., **kwargs):
    """Plotting helper fct"""
    makeidx = lambda ar: np.arange(ar.shape[0])
    
    if ax is None:
        fig, ax = plt.subplots()
    hit = residual_motion_mult(cavfct=cavfct, cavpars=cavpars, ar=ar, br=br, ap=ap, bp=bp, s=s, Nrt=Nrt, degmodenum=degmodenum, **kwargs)
    Nr = hit.shape[1]
    idx = makeidx(hit)
    for i in range(Nr):
        ax.scatter(hit[:,i,0], hit[:,i,1], c=idx, cmap='jet', s=ms)
    if ax is None:
        plt.xlabel(r'$x$ [mm]')
        plt.ylabel(r'$y$ [mm]')
        plt.show()
    
def plot_metric(metric, cavfct, cavpars, rmax=10., Nplt=100, degmodenum=1, ax=None, **kwargs):
    """Plotting helper fct"""
    if ax is None:
        fig, ax = plt.subplots()
    rs, ms = metric_over_r(metric=metric, cavfct=cavfct, cavpars=cavpars, rmax=rmax, Nplt=Nplt, degmodenum=degmodenum, **kwargs)
    ax.plot(rs, ms)
    plt.xlabel(r'$r_{in}$ [waists]')
    plt.ylabel(r'${}$ [mm]'.format(metric.__name__))
    if ax is None:
        plt.show()

def plot_metrics(cavfct, cavpars, rmax=12., Nplt=100, degmodenum=1, a=8., b=0., s=3, ms=8., **kwargs):
    """
    Composite plot of two metrics and motion in the screen plane
    Args:
        cafct (fct): cavity function
        cavpars (dict): cavity paramters
        rmax (float): radius to plot up to
        Nplt (int): number of plotting point
        degmodenum (int): 1 or 2, wich paraxial eigenvector to pick
        a/b (float): radius in units of waists for both eigenvectors
        s (int): subset of points to plot; s=1 plots all, s=3 only every third
        ms (float): marker size
    """
    makeidx = lambda ar: np.arange(ar.shape[0])
    
    fig, ax = plt.subplots(ncols=3, figsize=(11,3.3))
    
    rs, Ls = metric_over_r(metric=metric_var, cavfct=cavfct, cavpars=cavpars, rmax=rmax, Nplt=Nplt, degmodenum=degmodenum, **kwargs)
    ax[0].plot(rs, Ls)
    
    rs2, Ls2 = metric_over_r(metric=metric_opl, cavfct=cavfct, cavpars=cavpars, rmax=rmax, Nplt=Nplt, degmodenum=degmodenum, **kwargs)
    Ls2 -= Ls2[0]
    ax[1].plot(rs2, Ls2)
    
    if hasattr(a, "__len__"):
        plot_residual_motion_mult(cavfct=cavfct, cavpars=cavpars, ar=a, br=b, ap=0., bp=0., s=s, Nrt=Nplt, degmodenum=degmodenum, ax=ax[2], ms=ms)
    else:
        plot_residual_motion(cavfct=cavfct, cavpars=cavpars, ar=a, br=b, ap=0., bp=0., s=s, Nrt=Nplt, degmodenum=degmodenum, ax=ax[2], ms=ms)
    #hit = residual_motion(cavfct=cavfct, cavpars=cavpars, ar=a, br=b, s=s, Nrt=Nplt, degmodenum=degmodenum, **kwargs)
    #l3 = ax[2].scatter(hit[:,0], hit[:,1], c=makeidx(hit), cmap='jet')
    ax[0].set_xlabel(r'$r_{in}$ [waists]')
    ax[0].set_ylabel(r'deviation [mm]')
    ax[1].set_xlabel(r'$r_{in}$ [waists]')
    ax[1].set_ylabel(r'$\Delta OPL$ [mm]')
    ax[2].set_xlabel(r'$x$ [mm]')
    ax[2].set_ylabel(r'$y$ [mm]')
    ax[0].ticklabel_format(style='sci',scilimits=(-2,2),axis='y')
    ax[1].ticklabel_format(style='sci',scilimits=(-2,2),axis='y')
    plt.tight_layout()
    plt.show()

def plot_metric_par_vs_r(metric, cavfct, cavpars, param, rng = (1e-4, 2.5e-4), rmax=10., Nplt=50, degmodenum=1, ax=None, **kwargs):
    """
    Plot a metric 2d as a function of radius and one parameter
    """
    #assert param in cavpars.keys(), "You have to name a key from cavpars!"
    
    rs = np.linspace(np.finfo(np.float32).eps, rmax, Nplt)
    ps = np.linspace(*rng, Nplt)
    devs = np.zeros((len(rs), len(ps)))
    for i, r in enumerate(rs):
        for j, p in enumerate(ps):
            args = cavpars.copy()
            args[param] = p
            try:
                d = metric(cavfct=cavfct, cavpars=args, r=r, degmodenum=degmodenum, **kwargs)
            except:
                devs[i,j] = np.nan
            else:
                devs[i,j] = d
    
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.pcolormesh(rs, ps, devs.T, norm=LogNorm(vmin=devs.min(), vmax=devs.max()))
    ax.set_xlabel(r'$r_{in}$ [waists]')
    ax.set_ylabel(r'${}$'.format(param))
    ax.ticklabel_format(style='sci',scilimits=(-3,4),axis='y')
    if ax is None:
        plt.colorbar(im, cax=ax)
        plt.show()
    else:
        return im

def plot_metric_par2d_at_r(metric, cavfct, cavpars, parama, rnga, paramb, rngb, r_at=10., Nplt=50, degmodenum=1, ax=None, **kwargs):
    """
    Plot a metric 2d as a function of two parameters at radius r
    """
    #assert parama in cavpars.keys(), "You have to name a key from cavpars!"
    #assert paramb in cavpars.keys(), "You have to name a key from cavpars!"
    
    rs = np.linspace(*rnga, Nplt)
    ps = np.linspace(*rngb, Nplt)
    devs = np.zeros((len(rs), len(ps)))
    for i, r in enumerate(rs):
        for j, p in enumerate(ps):
            args = cavpars.copy()
            args[parama] = r
            args[paramb] = p
            try:
                d = metric(cavfct=cavfct, cavpars=args, r=r_at, degmodenum=degmodenum, **kwargs)
            except:
                devs[i,j] = np.nan
            else:
                devs[i,j] = d
    
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.pcolormesh(rs, ps, devs.T, norm=LogNorm(vmin=devs.min(), vmax=devs.max()))
    ax.set_xlabel(r'${}$'.format(parama))
    ax.set_ylabel(r'${}$'.format(paramb))
    ax.ticklabel_format(style='sci',scilimits=(-3,4),axis='both')
    if ax is None:
        plt.colorbar(im, cax=ax)
        plt.show()
    else:
        return im
    
def metric_interaction_factory(metric, cavfct, cavpars, rmax=12., Nplt=100, degmodenum=1, slrng=(0.8, 1.2), **kwargs):
    """
    Interactive plot of metric and manipulation of coefficients 
    Args:
        metric (fct): metric function
        cafct (fct): cavity function
        cavpars (dict): cavity paramters
        rmax (float): radius to plot up to
        Nplt (int): number of plotting point
        degmodenum (int): 1 or 2, wich paraxial eigenvector to pick
        slrng (tuple): slider range, minimum and maximum value scaled from parameter default value
    """
    magnitude = lambda x: 1. if x==0. else int(np.floor(np.log10(x)))
    
    fig, ax = plt.subplots()
    line = ax.plot([], [])[0]
    plt.xlabel(r'$r_{in}$ [waists]')
    plt.ylabel(r'${}$ [mm]'.format(metric.__name__))
    plt.show()
    
    #Because the coefficients for aspheres/waveplates get super small, there is some rescaling magic here!
    def update(scalings, **kwargs):
        scargs = {}
        for k, v in kwargs.items():
            scargs[k] = v*10**scalings[k]

        rs, Ls = metric_over_r(metric=metric, cavfct=cavfct, cavpars=scargs, rmax=rmax, Nplt=Nplt, degmodenum=degmodenum, **kwargs)
        line.set_xdata(rs)
        line.set_ydata(Ls)
        ax.relim()
        ax.autoscale_view(True,True,True)
        fig.canvas.draw_idle()

    lo = Layout(width='80%', height='30px')
    sliders = {}
    scalings = {}
    for k, v in cavpars.items():
        vs = v*10**(-magnitude(v))
        scalings[k] = magnitude(v)
        sliders[k] = FloatSlider(value=vs, min=slrng[0]*vs, max=slrng[1]*vs, step=1e-3, readout_format='.3e', layout=lo)

    print(scalings)
    return interactive(update, scalings=fixed(scalings), **sliders)

def metrics_interaction_factory(cavfct, cavpars, rmax=12., Nplt=100, slrng=(0.8, 1.2), degmodenum=1, a=8., b=0., s=3, ms=8,**kwargs):
    """
    Interactive composite plot of two metrics and motion in the screen plane of metric and manipulation of coefficients 
    Args:
        metric (fct): metric function
        cafct (fct): cavity function
        cavpars (dict): cavity paramters
        rmax (float): radius to plot up to
        Nplt (int): number of plotting point
        degmodenum (int): 1 or 2, wich paraxial eigenvector to pick
        slrng (tuple): slider range, minimum and maximum value scaled from parameter default value
        s (int): subset of points to plot; s=1 plots all, s=3 only every third
        ms (float): marker size
    """
    magnitude = lambda x: 1. if x==0. else int(np.floor(np.log10(x)))
    makeidx = lambda ar: np.arange(ar.shape[0])
    
    fig, ax = plt.subplots(ncols=3, figsize=(11,3.3))
    l1= ax[0].plot([], [])[0]
    l2= ax[1].plot([], [])[0]
    l3 = ax[2].scatter([], [], c=[], cmap='jet', s=ms)
    ax[0].set_xlabel(r'$r_{in}$ [waists]')
    ax[0].set_ylabel(r'deviation [mm]')
    ax[1].set_xlabel(r'$r_{in}$ [waists]')
    ax[1].set_ylabel(r'$\Delta OPL$ [mm]')
    ax[2].set_xlabel(r'$x$ [mm]')
    ax[2].set_ylabel(r'$y$ [mm]')
    plt.tight_layout()
    
    #Because the coefficients for aspheres/waveplates get super small, there is some rescaling magic here!
    def update(scalings, a, b, **kwargs):
        scargs = {}
        for k, v in kwargs.items():
            scargs[k] = v*10**scalings[k]

        rs, Ls = metric_over_r(metric=metric_var, cavfct=cavfct, cavpars=scargs, rmax=rmax, Nplt=Nplt, degmodenum=degmodenum, **kwargs)
        l1.set_xdata(rs)
        l1.set_ydata(Ls)
        ax[0].relim()
        ax[0].autoscale_view(True,True,True)
        
        rs2, Ls2 = metric_over_r(metric=metric_opl, cavfct=cavfct, cavpars=scargs, rmax=rmax, Nplt=Nplt, degmodenum=degmodenum, **kwargs)
        Ls2 -= Ls2[0]
        l2.set_xdata(rs2)
        l2.set_ydata(Ls2)
        ax[1].relim()
        ax[1].autoscale_view(True,True,True)
        
        hit = residual_motion(cavfct=cavfct, cavpars=scargs, ar=a, br=b, s=s, Nrt=Nplt, degmodenum=degmodenum, **kwargs)
        l3.set_offsets(hit[:,:])
        l3.set_array(makeidx(hit))
        ax[2].set_xlim(1.5*np.nanmin(hit[:,0]), 1.5*np.nanmax(hit[:,0]))
        ax[2].set_ylim(1.5*np.nanmin(hit[:,1]), 1.5*np.nanmax(hit[:,1]))
        l3.set_clim(vmin=0, vmax=Nplt/s)
        
        ax[0].ticklabel_format(style='sci',scilimits=(-2,2),axis='y')
        ax[1].ticklabel_format(style='sci',scilimits=(-2,2),axis='y')
        
        fig.canvas.draw_idle()

    lo = Layout(width='80%', height='30px')
    sliders = {}
    scalings = {}
    for k, v in cavpars.items():
        vs = v*10**(-magnitude(v))
        scalings[k] = magnitude(v)
        sliders[k] = FloatSlider(value=vs, min=slrng[0]*vs, max=slrng[1]*vs, step=1e-3, readout_format='.3e', layout=lo)
    
    
    a = FloatSlider(value=a, min=0., max=10., step=1e-3, readout_format='.3e', layout=lo)
    b = FloatSlider(value=b, min=0., max=10., step=1e-3, readout_format='.3e', layout=lo)
    print(scalings)
    return interactive(update, scalings=fixed(scalings), a=a, b=b, **sliders)

#Optimization routines
def metric_optim(x, metr, cavfct, cavpars, rmax=10., Nplt=100, degmodenum=1)
    """wrapper function for optimization"""
    cavpars.update({'c4': x[0], 'c6': x[1]})
    rs, ds = metric_over_r(metric=metr, cavfct=cavfct, cavpars=cavpars, rmax=rmax, Nplt=Nplt, degmodenum=degmodenum)
    dr = rs[1]-rs[0]
    return np.abs(np.sum(ds))*dr

