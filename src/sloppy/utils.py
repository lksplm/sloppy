import numpy as np
import k3d
from k3d.platonic import PlatonicSolid
from itertools import product


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
    m = norm(M)#normal vectors connecting elements
    
    nraw = np.array([np.cross(m[j,j-1],m[j,(j+1)%Nm]) for j in range(Nm)]) #normal vector to in/out plane
    msk = np.linalg.norm(nraw, axis=-1)

    # set normal vector for in-line optics to the previous normal vector
    def go_n_Prev(j):
        if np.linalg.norm(nraw[(j-1)%Nm]>1e-4):
            return nraw[(j-1)%Nm]
        else:
            return go_n_Prev(j-1)
        
    # set normal vector for in-line optics to the next normal vector
    def go_n_Next(j):
        if np.linalg.norm(nraw[(j+1)%Nm]>1e-4):
            return nraw[(j+1)%Nm]
        else:
            return go_n_Next(j+1)
        
    for j in range(Nm):
        if msk[j]<1e-4:
            nraw[j]=go_n_Next(j)
    n = norm(nraw)
    
    refl_raw = np.array([0.5*(m[j,j-1]+m[j,(j+1)%Nm]) for j in range(Nm)]) #vectors normal to reflecting mirrors
    refl_raw = [m[(j+1)%Nm,j] if np.linalg.norm(refl_raw[j])<1e-4 else refl_raw[j] for j in range(Nm)]
    refl = -norm(refl_raw)
    
    angles = np.array([0.5*np.arccos(0.999999*np.dot(m[j,j-1],m[j,(j+1)%Nm])) for j in range(Nm)]) # prevents error with range of np.arccos()
    
    m_in = np.array([m[j,j-1] for j in range(Nm)])
    m_out = np.array([m[j,(j+1)%Nm] for j in range(Nm)])
    
    xin = n
    xout = n
    yin = norm(np.array([np.cross(n[j],m[j,j-1]) for j in range(Nm)]))
    yout = norm(np.array([np.cross(n[j],-m[j,(j+1)%Nm]) for j in range(Nm)]))
    R = np.stack([np.array([[xout[i]@xin[(i+1)%Nm], yout[i]@xin[(i+1)%Nm]],\
                            [xout[i]@yin[(i+1)%Nm], yout[i]@yin[(i+1)%Nm]]]) for i in range(Nm)], axis=0)
    
    #Reflect vector
    def RefV(j, vec):
        return vec - (vec@refl[j])*refl[j]
    
    changeBasisAcross = np.stack([np.array([[RefV(i,xin[i])@xout[i], RefV(i,yin[i])@xout[i]],\
                                            [RefV(i,xin[i])@yout[i], RefV(i,yin[i])@yout[i]]]) for i in range(Nm)], axis=0)
    
    changeBasisAcross = np.round(changeBasisAcross) # fix slight numerical errors of the reflection  
    #make R 4x4
    R4 = np.zeros((R.shape[0], 4, 4))
    for i in range(R.shape[0]):
        m = np.identity(4)
        r = R[i]@changeBasisAcross[i]
        m[0:2,0:2] = r
        m[2:4,2:4] = r
        R4[i,:,:] = m
    
    Ls = [np.linalg.norm(M[j-1,j]) for j in range(Nm)]
    Lrt = sum(Ls)
    
    return {'mir': mir, 'M': M, 'n': n, 'refl': refl, 'm_in': m_in, 'm_out': m_out, 'angles': angles, 'xin': xin, 'xout': xout, 'yin': yin, 'yout': yout, 'R': R4, 'Ls': Ls, 'Lrt': Lrt}

def geometry_old(mir):
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
    
    #make R 4x4
    R4 = np.zeros((R.shape[0], 4, 4))
    for i in range(R.shape[0]):
        m = np.identity(4)
        r = R[i]
        m[0:2,0:2] = r
        m[2:4,2:4] = r
        R4[i,:,:] = m
    
    Ls = [np.linalg.norm(M[j-1,j]) for j in range(Nm)]
    Lrt = sum(Ls)

    return {'mir': mir, 'M': M, 'n': n, 'refl': refl, 'angles': angles, 'xin': xin, 'xout': xout, 'yin': yin, 'yout': yout, 'R': R4, 'Ls': Ls, 'Lrt': Lrt}

def plot_geometry(geom, scale_factor=1., arrow_length=2., **kwargs):
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
    pf = scale_factor
    plt_line = k3d.line(pf*mir, shader='mesh', width=0.5, color=col)
    plt_line2 = k3d.line(pf*mir[(-1,0),...], shader='mesh', width=0.5, color=col)
    plot += plt_line
    plot += plt_line2
    plot += k3d.vectors(origins=pf*mir, vectors=n*arrow_length, use_head=True, head_size=3.)#Normals = xIn = xOut
    plot += k3d.vectors(origins=pf*mir, vectors=yin*arrow_length, use_head=True, head_size=3., color= 0xff8c00) #yIn
    plot += k3d.vectors(origins=pf*mir, vectors=yout*arrow_length, use_head=True, head_size=3., color= 0xff8c00) #yOut
    plot += k3d.vectors(origins=pf*mir, vectors=refl*arrow_length, use_head=True, head_size=3., color=0x00ff00)

    ey = np.array([0,1,0])
    ex = np.array([1,0,0])
    for i in range(Nm):
        mirror = Box(size=(1,10,10)).mesh

        mirror.transform.custom_matrix = pad3to4(getrot(ex, refl[i])) #get rotation matrix of mirror
        mirror.transform.translation = pf*mir[i]
        plot += mirror

    return plot

#helper functions
def getrot(a,b):
    """Find rotation matrix (3x3) that rotates a onto b."""
    if np.allclose(a,b):
        return np.identity(3)
    else:
        v = np.cross(a, b)
        s = np.linalg.norm(v)
        c = np.dot(a,b)
        vx = np.array([[0,-v[2], v[1]],[v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.identity(3) + vx + vx@vx*(1/(1+c))#(1-c)/s**2

def pad3to4(mat):
    """Pad 3x3 roation to 4x4 matrix (for k3d)."""
    m = np.identity(4)
    m[0:3,0:3] = mat
    return m

def norm(arr, axis=-1):
    """Normalize (array of) vectors (last dimension assumed to be vector axis)."""
    arr = np.asarray(arr)
    norm = np.sqrt(np.sum(arr**2, axis=-1))
    norm = np.expand_dims(norm, axis)
    return np.divide(arr, norm, where=(norm!=0))

def vec2rot(a, b):
    """Find rotation axis and angle that rotates a onto b."""
    if np.allclose(a,b):
        return [0, 1, 0, 0]
    else:
        ez = norm(a)
        n = norm(b)
        ax = np.cross(ez,n)
        ang = np.arccos(ez@n)
        return [ang, ax[0], ax[1], ax[2]]
    
def rot_vec(v, ax, ang):
    """Rotate vector v by ang around axis ax."""
    if ang==0.0:
        return v
    else:
        return v*np.cos(ang) + np.cross(ax,v)*np.sin(ang) + ax@v*ax*(1. - np.cos(ang))

def disc_coords(n_radii = 5, n_angles = 10, R=1.):
    """Return polar coordinates of the disk."""
    radii = np.linspace(0., 1, n_radii, endpoint=True)
    x, y = [0], [0]
    for r in radii[1:]:
        x.extend([r*np.cos(p) for p in np.linspace(0,2*np.pi,int(2*np.pi*r*n_angles))])
        y.extend([r*np.sin(p) for p in np.linspace(0,2*np.pi,int(2*np.pi*r*n_angles))])
    x = np.array(x)*R
    y = np.array(y)*R
    return x, y


class Box(PlatonicSolid):
    def __init__(self, origin=[0, 0, 0], size=[1,1,1]):
        origin = np.array(origin, dtype=np.float32)
        size = np.array(size, dtype=np.float32)
        
        if origin.shape == (3,):
            cube_vertices = np.array(list(product([size[0], -size[0]], [size[1], -size[1]], [size[2], -size[2]])), np.float32)
            cube_vertices = np.float32(cube_vertices + origin)

            self.vertices = cube_vertices
            self.indices = [0, 1, 2, 1, 2, 3, 0, 1, 4, 1, 4, 5, 1, 3, 5, 3, 5, 7, 0, 2, 4, 2, 4, 6, 2, 3, 7, 2, 6, 7, 4,
                            5, 6, 5, 6, 7]

        else:
            raise TypeError('Origin attribute should have 3 coordinates.')
            
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
            
def traj_to_timeseries(traj_anim, step=1):
     return [{str(t/100): traj_anim[t,:,r,:] for t in range(0, traj_anim.shape[0], step)} for r in range(traj_anim.shape[2])]
    
def modelmats_to_timeseries(modelmats, step=1):
     return [{str(t/100): modelmats[t,i,:,:] for t in range(0, modelmats.shape[0], step)} for i in range(modelmats.shape[1])]
            
def GramSchmidt(V, prod, norm):
    """Computes orthornormal vectors given k linearly independent vectors of dimension n.
    Args:
        V (ndarray): Input vectors of shape (k, n).
        prod (function): product between two vectors.
        norm (function): norm of a vector.
    Return:
        U (ndarray): Orthonormal output vectors of shape (k, n).
    """
    n = V.shape[1]
    k = V.shape[0]
    #U = np.zeros((k, n), dtype=V.dtype)
    U = np.empty_like(V)
    U[0,:] = V[0,:]/norm(V[0,:])
    for i in range(1,k):
        U[i,:] = V[i,:]
        for j in range(i):
            U[i,:] = U[i,:] - prod(U[j,:], U[i,:])/prod(U[j,:], U[j,:])*U[j,:] 
        U[i,:] = U[i,:]/norm(U[i,:])
    return U

def plot_element_coordinates(el, plot, length=1.):
    #plot += k3d.vectors(el.p, length*el.xin, color=0x9633ff) #purple 
    #plot += k3d.vectors(el.p, length*el.xout, color=0xfec33) #yellow 
    plot += k3d.vectors(el.p, length*el.ax_yin, color=0xff5733) #orange
    plot += k3d.vectors(el.p, length*el.ax_yout, color=0xff33f6) #pink
    plot += k3d.vectors(el.p, length*el.n, color=0x00ff00) #green
    plot += k3d.vectors(el.p, length*el.ax_x, color=0xff0000) #red
    plot += k3d.vectors(el.p, length*el.ax_y, color=0x0000ff) #blue
    
def plot_element_ax(el, plot, length=1.):
    #plot += k3d.vectors(el.p, length*el.xin, color=0x9633ff) #purple 
    #plot += k3d.vectors(el.p, length*el.xout, color=0xfec33) #yellow 
    plot += k3d.vectors(el.p, length*el.n, color=0x00ff00) #green
    plot += k3d.vectors(el.p, length*el.ax, color=0xff0000) #red
    plot += k3d.vectors(el.p, length*el.ay, color=0x0000ff) #blue
