import numpy as np
from k3d.platonic import PlatonicSolid
from itertools import product

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
    radii = np.linspace(0., 1, n_radii)
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
            
