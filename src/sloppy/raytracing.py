import numpy as np
import k3d
import matplotlib.pyplot as plt
from functools import reduce
from matplotlib.tri import Triangulation
from .utils import *
from .optics import *

#ray tracing functions

def ray_bundle(p=(0,0,0), n=(0,0,1), n_radii=10, n_angles=2, R=1., divergence=0.):
    p = np.array(p)
    n = norm(n)
    radii = np.linspace(0., 1, n_radii)
    x, s = [[0,0,0],], [[0,0,1],]
    for r in radii[1:]:
        for t in np.linspace(0,2*np.pi,int(2*np.pi*r*n_angles)):
            x.append( [R*r*np.cos(t), R*r*np.sin(t), 0])
            s.append(norm([divergence*r*np.cos(t), divergence*r*np.sin(t), 1]))
       
    rays = np.stack([np.array(x), np.array(s)], axis=0)
    ez = np.array([0,0,1])
    R = getrot(ez, -n)
    rays2 = np.einsum("abi,ij->abj", rays, R)
    rays2[0,:,:] += p
    return rays2

def plot_rays(rays, plot, length=10., **kwargs):
    s = k3d.vectors(origins=rays[0,...], vectors=rays[1,...], colors=[(0xff0000,)*rays.shape[1]*2], head_size=2.)
    plot += s
    ls = [k3d.line([rays[0,i,:], rays[0,i,:]+15*rays[1,i,:]], **kwargs) for i in range(rays.shape[1]) if rays[0,i,0] is not np.nan]
    for l in ls:
        plot += l
        
        
#geometry construction
def geometry(mir):
    """Construct a cavity geometry from mirror positions.

    The function is more general and can also treat other optics like lenses etc.,
    one just has to be careful with the normal vectors of transmitting elements.


    Args:
        mir (ndarray): Array containing the mirror positions with shape (Nmirror, 3).

    Returns:
        A dictonary including all the geometry of the optical elements:
            {
                'M': pairwise distance matrix between all elements,
                'n': normal vector to input-output plane,
                'refl': normal vector of the element,
                'angles': half opening angle at element,
                'xin': x-axis of the input coordinate system,
                'yin': y-axis of the input coordinate system,
                'xout': x-axis of the output coordinate system,
                'yout': y-axis of the output coordinate system,
                'R': transformation matrix between input and output coordinate systems,
                'Ls': distances between the elements,
                'Lrt': total roundtrip distance
            }
            
    """
    
    Nm = len(mir)
    M = mir[:,None,:]-mir[None,:,:]
    m = norm(M)
    
    n = norm(np.array([np.cross(m[j,j-1],m[j,(j+1)%Nm]) for j in range(Nm)]))
    refl = -norm(np.array([0.5*(m[j,j-1]+m[j,(j+1)%Nm]) for j in range(Nm)])) #vectors normal to reflecting mirrors
    angles = np.array([0.5*np.arccos(np.dot(m[j,j-1],m[j,(j+1)%Nm])) for j in range(Nm)])
    xin = n
    xout = n
    yin = norm(np.array([np.cross(n[j],m[j,j-1]) for j in range(Nm)]))
    yout = norm(np.array([np.cross(n[j],m[j,(j+1)%Nm]) for j in range(Nm)]))
    R = np.stack([np.array([[xout[i]@xin[(i+1)%Nm], yout[i]@xin[(i+1)%Nm]],\
                            [xout[i]@yin[(i+1)%Nm], yout[i]@yin[(i+1)%Nm]]]) for i in range(Nm)], axis=0)
    
    Ls = [np.linalg.norm(M[j-1,j]) for j in range(Nm)]
    Lrt = sum(Ls)

    return {'mir': mir, 'M': M, 'n': n, 'refl': refl, 'angles': angles, 'xin': xin, 'xout': xout, 'yin': yin, 'yout': yout, 'R': R, 'Ls': Ls, 'Lrt': Lrt}

def plot_geometry(geom, **kwargs):
    """Plot cavity geometry including coordinate systems.

    Args:
        geom (dict): A dict of cavity geometry from :function:`geometry`.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        k3d.plot: 3D plot of the geometry


    """
    mir, n, refl, yin, yout = geom['mir'], geom['n'], geom['refl'], geom['yin'], geom['yout']
    Nm = len(mir)
    plot = k3d.plot(camera_auto_fit=True, antialias=True)

    col=0xff0000
    pf = 1.
    plt_line = k3d.line(pf*mir, shader='mesh', width=0.5, color=col)
    plt_line2 = k3d.line(pf*mir[(-1,0),...], shader='mesh', width=0.5, color=col)
    plot += plt_line
    plot += plt_line2
    plot += k3d.vectors(origins=pf*mir, vectors=n*2, use_head=True, head_size=3.)#Normals = xIn = xOut
    plot += k3d.vectors(origins=pf*mir, vectors=yin*2, use_head=True, head_size=3., color= 0xff8c00) #yIn
    plot += k3d.vectors(origins=pf*mir, vectors=yout*2, use_head=True, head_size=3., color= 0xff8c00) #yOut
    plot += k3d.vectors(origins=pf*mir, vectors=refl*2, use_head=True, head_size=3., color=0x00ff00)

    ey = np.array([0,1,0])
    ex = np.array([1,0,0])
    for i in range(Nm):
        mirror = Box(size=(1,10,10)).mesh

        mirror.transform.custom_matrix = pad3to4(getrot(ex, refl[i])) #get rotation matrix of mirror
        mirror.transform.translation = pf*mir[i]
        plot += mirror

    return plot

#Propagation
def propagate_system(elements, rays, Nrt=1, clip=True):
    """Propagate rays through an optical system (series of elements).

    Args:
        elements (list): A list of :class:`Optic` objects forming an optical system.
        rays (ndarray): Input rays to propagate of shape (2, Nrays, 3).
        Nrt (int): Number of roundtrips to propagate.
        clip (bool): If true, rays are clipped on apertures of each element.

    Returns:
        allrays (ndarray): Array of rays of shape (Nelements*Nrt+1, 2, Nrays, 3).
            The input ray is always stored as the first element.

    """
    rs = rays.copy()
    ind = np.arange(rs.shape[1])
    Nel = len(elements)
    allrays = np.empty((Nel*Nrt+1,*rays.shape))
    allrays[0,...] = rs
    for i in range(Nrt):
        for j, el in enumerate(elements):
            rs = el.propagate(rs, clip=clip)
            allrays[i*Nel+j+1,...] = rs
        
    return allrays

def propagate_system_at(elements, rays, Nrt=1, which=0, clip=True):
    """Propagate rays through an optical system (series of elements).
    Works exactly like :function:`propagate_system` but only stores intersection at one element (which) for speed.
    Args:
        which (int): Number of the element to store the propagated ray at.


    Returns:
        allrays (ndarray): Array of rays of shape (Nrt, 2, Nrays, 3).
            The input ray is always stored as the first element.

    """
    rs = rays.copy()
    Nel = len(elements)

    allrays = np.empty((Nrt,*rays.shape))
    for i in range(Nrt):
        for j, el in enumerate(elements):
            rs = el.propagate(rs, clip=clip)
            if j==which:
                allrays[i,...] = rs
        
    return allrays

def clip_traj(traj):
    """Utility function to return a trajectory up to where they clip an element (become Nan)."""
    idx = np.where(np.isnan(traj[:,0]))[0]
    if len(idx)>0:
        idx = idx[0]
        return traj[:idx,:]
    else:
        return traj

def plot_trajs(trajs, plot, **kwargs):
    """Plot a trajectory in k3d plot.

    Args:
        trajs (ndarray): Trajectories to plot.
        plot (k3d.plot): Plot object to add trajectories to (added inplace).
    """
    for i in range(trajs.shape[2]):
        t = clip_traj(trajs[:,0,i,:])

        if t.shape[0]>1:
            l = k3d.line(t, **kwargs)
            plot += l
            
def find_eigenray(elements, ray0, lr = 0.05, maxiter=500, tol=1e-5, Nrt=1, debug=False):
    """Find the eigenray (fixed point) of an optical system iteratively.

    Args:
        elements (list): A list of :class:`Optic` objects forming an optical system.
        ray0 (ndarray): Input rays to stabilize of shape (2, Nrays, 3).
        Nrt (int): Number of roundtrips to propagate between iteration steps.
        maxiter (int): Maximum number of iterations.
        tol (float): Tolerance (relative change) down to which to iterate.
            Terminates if tol or maxiter reached.
        lr (float): 'learing rate', must be hand-tuned.
        debug (bool): If true, prints debug messages.
        
    Returns:
        rcur (ndarray): Eigenray of the system (2, Nrays, 3).
    """
    rcur = ray0.copy()
    for i in range(maxiter):
        traj = propagate_system(elements, rcur, Nrt=Nrt)
        rnew = traj[-1,...]
        if np.isnan(rnew).any():
            if debug:
                print("Failed")
            break
        res = np.max(np.abs(rcur.flatten() - rnew.flatten()))

        rcur = (1.-lr)*rcur + lr*rnew
        if res<tol:
            break
    if debug:
        print("Finished in {} steps, reached tol {:.3e}".format(i, res))
    
    return rcur

def find_eigenray_animated(elements, ray0, lr = 0.05, maxiter=500, tol=1e-5, Nrt=1, debug=False, clip=True):
    """Version of :function:`find_eigenray` that stores intermediate trajectories for visualisation.
    
    Returns:
        rcur (ndarray): Eigenray of the system (2, Nrays, 3).
        trajs (ndarray): Eigenray of the system (Nelements*Nrt, 2, Nrays, 3).
    """
    rcur = ray0.copy()
    trajs = []
    for i in range(maxiter):
        traj = propagate_system(elements, rcur, Nrt=Nrt, clip=clip)
        rnew = traj[-1,...]
        if np.isnan(rnew).any():
            if debug:
                print("Failed")
            break
        trajs.append(traj[:,0,:,:])
        res = np.max(np.abs(rcur.flatten() - rnew.flatten()))
        rcur = (1.-lr)*rcur + lr*rnew
        #renormalize normal vector
        nnorm = np.linalg.norm(rcur[1,:,:], axis=-1)
        rcur[1,:,:] *= (1/nnorm)[:,None]
        if res<tol:
            break
    if debug:
        print("Finished in {} steps, reached tol {:.3e}".format(i, res))
            
    return rcur, np.stack(trajs, axis=0)

def _find_eigenray_formpe(elements, ray0, lr = 0.05, maxiter=500, tol=1e-5, Nrt=1, debug=False, clip=True):
    rcur = ray0.copy()
    trajs = []
    tols = []
    for i in range(maxiter):
        traj = propagate_system(elements, rcur, Nrt=Nrt, clip=clip)
        rnew = traj[-1,...]
        if np.isnan(rnew).any():
            if debug:
                print("Failed")
            break
        trajs.append(rnew)
        res = np.max(np.abs(rcur.flatten() - rnew.flatten()))
        #res = np.mean(np.abs(rcur.flatten() - rnew.flatten()))
        tols.append(res)
        rcur = (1.-lr)*rcur + lr*rnew
        #renormalize normal vector
        nnorm = np.linalg.norm(rcur[1,:,:], axis=-1)
        rcur[1,:,:] *= (1/nnorm)[:,None]
        if res<tol:
            break
    if debug:
        print("Finished in {} steps, reached tol {:.3e}".format(i, res))
            
    return rcur, np.stack(trajs, axis=0), np.array(tols)

def MPE(x):
    """Minimal Polynomial Extrapolation Alogirthm.
    Extrapolates a (vector) sequence from a fixpoint iteration procedure to find the true solution.
    Args:
        x (ndarray): Input vector sequence with shape (n, k-1) where n is the spatial dimension.
        
    Returns:
        s (ndarray): Output vector with shape (n).

    """

    U = x[:,1:-1] - x[:,:-2]
    c = -np.linalg.pinv(U)@(x[:,-1]-x[:,-2])
    c = np.append(c, 1.0)
    s = x[:,1:]@c/np.sum(c)
    return s

def find_eigenray_mpe_debug(elements, ray0, lr=0.03, Niter=50, Nmpe=5, tol=1e-9, **kwargs):
    rnew = ray0
    alltols = []
    alltraj = []
    for i in range(Nmpe):
        rconv, rseq, tols = _find_eigenray_formpe(elements, rnew, lr=lr, maxiter=Niter, debug=True, tol=tol*1e-1, **kwargs)
        alltraj.append(rseq)
        alltols.append(tols)
        if rseq.shape[0]<4:
            rnew = rconv
            break
        rseq_rs = np.squeeze(rseq).reshape(-1,6).T #reshape sequence into format for MPE
        rnew = MPE(rseq_rs) #find new starting vector
        rnew = rnew.reshape(2,-1,3)
    return rnew, np.concatenate(alltraj, axis=0), np.concatenate(alltols)

def find_eigenray_mpe(elements, ray0, lr=0.03, Niter=50, Nmpe=5, tol=1e-9, **kwargs):
    """Version of :function:`find_eigenray` that uses MPE to accelerate convergence.
    
    Args:
        Nrt (int): Number of roundtrips to propagate between iteration steps.
        Niter (int): Number of iterations of the inner (iterative) fixpoint algorithm.
        Nmpe (int): Number of iterations of the outer (MPE) fixpoint algorithm.
        tol (float): Tolerance (relative change) down to which to iterate.
        
    Returns:
        rnew (ndarray): Eigenray of the system (2, Nrays, 3).
    """
    rnew = ray0.copy()
    for i in range(Nmpe):
        rconv, rseq, _ = _find_eigenray_formpe(elements, rnew, lr=lr, maxiter=Niter, debug=False, tol=tol*1e-1, **kwargs)
        if rseq.shape[0]<4:
            rnew = rconv
            break
        rseq_rs = np.squeeze(rseq).reshape(-1,6).T #reshape sequence into format for MPE
        rnew = MPE(rseq_rs) #find new starting vector
        rnew = rnew.reshape(2,-1,3)
    return rnew

def traj_to_timeseries(traj_anim, step=5):
     return [{str(t/100): traj_anim[t,:,r,:] for t in range(0, traj_anim.shape[0], step)} for r in range(traj_anim.shape[2])]
    
def modelmats_to_timeseries(modelmats, step=5):
     return [{str(t/100): modelmats[t,i,:,:] for t in range(0, modelmats.shape[0], step)} for i in range(modelmats.shape[1])]
    
def perturb_one_anim(SysFct, screen, reig, traj_eig, which=0, ax=0, rng=np.linspace(-0.5, 0.5, 10)):
    elements, _ = SysFct()
    Nel = len(elements)
    
    eps = np.zeros((Nel,5))
    Nstp = len(rng)

    dists = np.empty((Nstp, *traj_eig.shape[:-1]))
    trajs_pert = np.empty((Nstp, *traj_eig.shape))
    modelmats = np.empty((Nstp, Nel, 4, 4))
    for i, e in enumerate(rng):
        #create pert. system
        eps[which,ax] = e
        elements, _ = SysFct(eps=eps)
        elements.append(screen)

        #find eigenray for this perturbation
        re = find_eigenray(elements, reig, lr=0.1, Nrt=1)
        #propagate new eigenray one roundtrip
        te = propagate_system(elements, re, Nrt=1)[:,0,:,:] #only positions
        trajs_pert[i,...] = te
        dists[i,...] = np.linalg.norm(traj_eig - te, axis=-1)
        for j, elm in enumerate(elements[:-1]):
            modelmats[i,j,:,:] = elm.plot().model_matrix
    return dists, trajs_pert, modelmats

def perturb_one(SysFct, screen, reig, traj_eig, which=0, ax=0, rng=np.linspace(-0.5, 0.5, 10)):
    elements, _ = SysFct()
    Nel = len(elements)
    
    eps = np.zeros((Nel,5))
    Nstp = len(rng)

    dists = np.empty((Nstp, *traj_eig.shape[:-1]))
    for i, e in enumerate(rng):
        #create pert. system
        eps[which,ax] = e
        elements, _ = SysFct(eps=eps)
        elements.append(screen)

        #find eigenray for this perturbation
        re = find_eigenray(elements, reig, lr=0.1, Nrt=1)
        #propagate new eigenray one roundtrip
        te = propagate_system(elements, re, Nrt=1)[:,0,:,:] #only positions
        dists[i,...] = np.linalg.norm(traj_eig - te, axis=-1)
    return dists

def extract_ABCD_fd(elements, x0, n0, epsr = 1e-1, epss = 1e-1, Nrt=1):
    """Extract an ABCD matrix from raytracing.
    Five guiding rays (of the form (eps, 0, 0, 0)...) are generated 
    and the matrix entries are determined by finite difference.
    Args:
        elements (list): List of optical elements.
        x0 (ndarray): Position of ray source.
        n0 (ndarray): Slope (direction) of ray source.
        epsr (float): Epsilon for position offset.
        epss (float): Epsilon for position slope (might be more sensitive).
        Nrt (int): Number of roundtrips ttto extract matrix from.
    """
    rguideray = np.array([[0,0,0], [0,epsr,0], [0,0,epsr], [0,0,0], [0,0,0]]) + x0
    sguideray = norm(n0 + np.array([[0,0,0], [0,0,0], [0,0,0], [0,epss,0], [0,0,epss]]))
    guideray = np.stack([np.atleast_2d(rguideray), np.atleast_2d(sguideray)], axis=0)
    trajgr = propagate_system(elements, guideray, Nrt=Nrt, clip=True)
    
    trajfd = trajgr[-1,:,1:,:]-(trajgr[-1,:,0,:])[:,None,:]
    Mfd = np.concatenate([trajfd[0,:,1:]/epsr, trajfd[1,:,1:]/epss], axis=1).T
    return Mfd

def eigenvector_to_ray(mu, x0, k):
    """Convert eigenvector of ABCD matrix mu into 3D rays for raytracing."""
    r0 = np.array([0, mu[1], mu[0]]) + x0
    a, b = mu[2:]/k
    s0 = np.array([np.sqrt(1-a**2-b**2), b, a])
    ray0 = np.stack([np.atleast_2d(r0), np.atleast_2d(s0)], axis=0)
    return ray0