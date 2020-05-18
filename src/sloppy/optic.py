"""Optical elements

This module contains the main definition of all optical elements (mirrors, lenses, etc.).
It contains the ray tracing routines as well as ABCD matrix definitions
for all common optics and wraps the numba jitclass for accelerated raytracing.

Todo:
    * Complete documentation
"""
import numpy as np
import k3d
from matplotlib.tri import Triangulation
from .utils import *
from .joptic import JitOptic

class Optic:
    """Base class for all optical elements.

    It collects common properties (position and direction) and methods like
    coordinate transformations, intersection and plotting.

    Attributes:
        p (ndarray): Position of the element in space (origin point).
        n (ndarray): Direction of the element as normal vector.
        diameter (float): Diameter of the elements aperture.
        ax (ndarray): X axis in element coordinates.
        ay (ndarray): Y axis in element coordinates.
        m (ndarray): ABCD matrix corresponing to this element (4x4).

    """
    def __init__(self, p=(0., 0., 0.), n=(0., 0. ,1.), ax=(1., 0. , 0.), ay=(0., 1., 0.), diameter=1.0, Rbasis=np.eye(4)):
        self.rapt = diameter/2.
        self.p = np.array(p, dtype=np.float64)
        self.n = np.array(norm(n), dtype=np.float64)
        self.ax = np.array(ax, dtype=np.float64)
        self.ay = np.array(ay, dtype=np.float64)
        self.Rot = np.stack((self.ax, self.ay, self.n)).T
        self.Rbasis = Rbasis
        #create JitOptic
        self.jopt = JitOptic(p=self.p, n=self.n, ax=self.ax, ay=self.ay, Rot=self.Rot, rapt=self.rapt, R=0., nratio=1.0, otype=0)
        self.m = np.eye(4)
        
    @property  
    def transformation(self):
        #get translation, rotation in k3d format
        return self.p, vec2rot(norm([0,0,1]), self.n)
        
    def plot(self, n_radii = 5, n_angles = 10, **kwargs):
        """Plots an optical element as a disk.
        The disk is represented by a polar grid of
        Args:
            n_radii: number of radial points.
            n_angles: number of axial points

        Returns:
            k3d.mesh object representing the surface

        """
        x, y = disc_coords(n_radii = n_radii, n_angles = n_angles, R=self.rapt)
        indices = Triangulation(x,y).triangles.astype(np.uint32)
        z = np.zeros_like(x)
        mesh = k3d.mesh(np.vstack([x,y,z]).T, indices, **kwargs)
        #t, r = self.transformation
        #mesh.transform.rotation = r
        #mesh.transform.translation = t
        mesh.transform.translation = self.p
        mesh.transform.custom_matrix = pad3to4(self.Rot).astype(np.float32)
        return mesh  
    
    def intersect(self, ray, clip=True):
        return self.jopt.intersect(ray, clip)
    
    def propagate(self, ray, clip=True):
        return self.jopt.propagate(ray, clip)
    
class Mirror(Optic):
    """Plane mirror, no extra arguments."""
    def __init__(self, p=(0., 0., 0.), n=(0., 0. ,1.), ax=(1., 0. , 0.), ay=(0., 1., 0.), diameter=1.0, Rbasis=np.eye(4)):
        super().__init__(p, n, ax, ay, diameter, Rbasis)
        
class Screen(Optic):
    """Plane mirror, no extra arguments."""
    def __init__(self, p=(0., 0., 0.), n=(0., 0. ,1.), ax=(1., 0. , 0.), ay=(0., 1., 0.), diameter=1.0, Rbasis=np.eye(4)):
        super().__init__(p, n, ax, ay, diameter, Rbasis)
        self.jopt = JitOptic(p=self.p, n=self.n, ax=self.ax, ay=self.ay, Rot=self.Rot, rapt=self.rapt, R=0., nratio=1.0, otype=1)
        
    def r_from_screen_coords(self, v):
        """Turns 2d vectors from screen coordinates with shape (N, 2) into global coords."""
        v = np.atleast_2d(v)
        vs = np.pad(v, pad_width=((0,0), (0,1)))
        vr = np.einsum("ij,kj->ki",self.Rot,vs) + self.p
        return vr
    
    def r_to_screen_coords(self, vr):
        """Turns 3d vectors from global coordinates with shape (N, 3) into screen coords with shape (N, 2)."""
        vr = np.atleast_2d(vr)
        vr = vr-self.p
        vss = np.einsum("ij,kj->ki",self.Rot.T, vr)
        return vss[:,:-1]
    
    def s_from_screen_coords(self, mu):
        """Turns 2d slope vectors from screen coordinates with shape (N, 2) into global normal vectors (N, 3)."""
        mu = np.atleast_2d(mu)
        s0 = np.stack([mu[:,0], mu[:,1], np.sqrt(1 - (mu[:,0])**2 - (mu[:,1])**2)], axis=1) #create vector normal to screen
        s0 = np.einsum("ij,kj->ki",self.Rot,s0) #rotate into global coordinates
        return s0
        
    def s_to_screen_coords(self, vr):
        """Turns 3d normal vectors from global coordinates with shape (N, 3) into 2d slope vectors with shape (N, 2)."""
        vr = np.atleast_2d(vr)
        vss = np.einsum("ij,kj->ki",self.Rot.T, vr)#rotate back to screen coords
        return vss[:,:-1]
    
    def eigenvectors_to_rays(self, mu):
        """Convert eigenvectors of ABCD matrix mu of shape (N,4) into 3D rays for raytracing of shape (2, N, 3)."""
        mu = np.atleast_2d(mu)
        r0 = self.r_from_screen_coords(mu[:,:2])
        s0 = self.s_from_screen_coords(mu[:,2:])
        ray0 = np.stack([np.atleast_2d(r0), np.atleast_2d(s0)], axis=0)
        return ray0
    
    def rays_to_eigenvectors(self, ray):
        """Convert eigenvectors of ABCD matrix mu of shape (N,4) into 3D rays for raytracing of shape (2, N, 3)."""
        r0 = self.r_to_screen_coords(ray[0,:,:])
        s0 = self.s_to_screen_coords(ray[1,:,:])
        mu = np.concatenate([r0, s0], axis=1)
        return mu
        
        
class CurvedMirror(Optic):
    """Curved reflecting mirror, can be conccave ('CC') or convex ('CX').
    
    Args:
        R (float): Radius of curvature, always positive!
        curv (string): 'CC' or 'CX' always w.r.t. to the normal vector.
            Schematically this looks like this (<- is normal):
                CC: ----- <-)
                CX: ----- <-(
        thet (float): angle of reflection for the ABCD matrix in sagital/tangential plane.
        TODO: Implement thet

    """
    def __init__(self, p=(0., 0., 0.), n=(0., 0. ,1.), ax=(1., 0. , 0.), ay=(0., 1., 0.), diameter=1.0, R=1., curv='CC', thet=0., Rbasis=np.eye(4)):
        super().__init__(p, n, ax, ay, diameter, Rbasis)
        self.R = abs(R)
        if curv == 'CC':
            self.jopt = JitOptic(p=self.p, n=self.n, ax=self.ax, ay=self.ay, Rot=self.Rot, rapt=self.rapt, R=abs(self.R), nratio=1.0, otype=2)
        elif curv == 'CX':
            self.jopt = JitOptic(p=self.p, n=self.n, ax=self.ax, ay=self.ay, Rot=self.Rot, rapt=self.rapt, R=abs(self.R), nratio=1.0, otype=3)
        else:
            raise ValueError("Mirror type {} unknown! Curv has to be CC or CX".format(curv))
        self.curv = curv
        self.thet = thet #TODO fix this from coordinate system.
        m = np.identity(4)
        m[2,0] = -2/R*np.cos(thet) #sagital
        m[3,1] = -2/R/np.cos(thet) #tangential
        self.m = m

    def plot(self, n_radii = 10, n_angles = 10, **kwargs):
        x, y = disc_coords(n_radii = n_radii, n_angles = n_angles, R=self.rapt)
        z = self.R-np.sqrt(self.R**2-x**2-y**2)
        if self.curv=='CX':
            z = -z
        indices = Triangulation(x,y).triangles.astype(np.uint32)
        mesh = k3d.mesh(np.vstack([x,y,z]).T, indices, **kwargs)
        mesh.transform.translation = self.p
        mesh.transform.custom_matrix = pad3to4(self.Rot).astype(np.float32)
        return mesh
    
class Glass(Optic):
    """Interface in the indecies of refraction i.e. a glass surface.
    Args:
        n1 (float): Index of refraction before the interface.
        n2 (float): Index of refraction after the interface.

    """
    def __init__(self, p=(0., 0., 0.), n=(0., 0. ,1.), ax=(1., 0. , 0.), ay=(0., 1., 0.), diameter=1.0, n1=1., n2=1., Rbasis=np.eye(4)):
        super().__init__(p, n, ax, ay, diameter, Rbasis)
        self.n1 = n1
        self.n2 = n2
        self.nratio = n1/n2
        self.jopt = JitOptic(p=self.p, n=self.n, ax=self.ax, ay=self.ay, Rot=self.Rot, rapt=self.rapt, nratio=self.nratio, otype=4)
        m = np.identity(4)
        m[2,2] = self.nratio
        m[3,3] = self.nratio
        self.m = m
        
class CurvedGlass(Optic):
    """Curved refracting surface, can be conccave ('CC') or convex ('CX').

    R (float): Radius of curvature, always positive!
    curv (string): 'CC' or 'CX' always w.r.t. to the normal vector.
    Schematically this looks like this:
        CC: ----- <-)
        CX: ----- <-(
    thet (float): angle of reflection for the ABCD matrix in sagital/tangential plane.
    n1 (float): Index of refraction before the interface.
    n2 (float): Index of refraction after the interface.
    
    TODO: Implement thet

    """
    def __init__(self, p=(0., 0., 0.), n=(0., 0. ,1.), ax=(1., 0. , 0.), ay=(0., 1., 0.), diameter=1.0, R=1., curv='CC', n1=1., n2=1., Rbasis=np.eye(4)):
        super().__init__(p, n, ax, ay, diameter, Rbasis)
        self.n1 = n1
        self.n2 = n2
        self.nratio = n1/n2
        self.R = abs(R)
        self.curv = curv
        if curv == 'CC':
            self.jopt = JitOptic(p=self.p, n=self.n, ax=self.ax, ay=self.ay, Rot=self.Rot, rapt=self.rapt, R=abs(self.R), nratio=self.nratio, otype=5)
        elif curv == 'CX':
            self.jopt = JitOptic(p=self.p, n=self.n, ax=self.ax, ay=self.ay, Rot=self.Rot, rapt=self.rapt, R=abs(self.R), nratio=self.nratio, otype=6)
        else:
            raise ValueError("Mirror type {} unknown! Curv has to be CC or CX".format(curv))
        m = np.identity(4)
        m[2,2] = n1/n2
        m[3,3] = n1/n2
        m[2,0] = (n1-n2)/(R*n2) #TODO sagittal and tangential plane
        m[3,1] = (n1-n2)/(R*n2)
        self.m = m
        
    def plot(self, n_radii = 10, n_angles = 10, **kwargs):
        x, y = disc_coords(n_radii = n_radii, n_angles = n_angles, R=self.rapt)
        z = self.R-np.sqrt(self.R**2-x**2-y**2)
        if self.curv=='CX':
            z = -z
        indices = Triangulation(x,y).triangles.astype(np.uint32)
        mesh = k3d.mesh(np.vstack([x,y,z]).T, indices, **kwargs)
        mesh.transform.translation = self.p
        mesh.transform.custom_matrix = pad3to4(self.Rot).astype(np.float32)
        return mesh
    
class FreeFormMirror(Optic):
    """Free form radially symmetric optic of the form z = \sum_i=0^deg coef[i] r**i
    Args:
        coef (ndarray): coefficients
    """
    def __init__(self, p=(0., 0., 0.), n=(0., 0. ,1.), ax=(1., 0. , 0.), ay=(0., 1., 0.), diameter=1.0, Rbasis=np.eye(4), coef=np.zeros(3)):
        super().__init__(p, n, ax, ay, diameter, Rbasis)
        self.coef = coef
        self.jopt = JitOptic(p=self.p, n=self.n, ax=self.ax, ay=self.ay, Rot=self.Rot, rapt=self.rapt, coef=self.coef, otype=7)
        #TODO abcd for quadratic part?
        m = np.identity(4)
        if abs(coef[2])>0.: #quadratic curvature non-zero
            m[2,0] = -coef[2] #sagital
            m[3,1] = -coef[2] #tangential
        self.m = m

    def plot(self, n_radii = 10, n_angles = 10, **kwargs):
        x, y = disc_coords(n_radii = n_radii, n_angles = n_angles, R=self.rapt)
        z = np.polyval(self.coef[::-1], np.sqrt(x**2 + y**2))
        indices = Triangulation(x,y).triangles.astype(np.uint32)
        mesh = k3d.mesh(np.vstack([x,y,z]).T, indices, **kwargs)
        mesh.transform.translation = self.p
        mesh.transform.custom_matrix = pad3to4(self.Rot).astype(np.float32)
        return mesh
    
class FreeFormInterface(FreeFormMirror):
    """Free form radially symmetric optic of the form z = \sum_i=0^deg coef[i] r**i
    Args:
        coef (ndarray): coefficients
    """
    def __init__(self, p=(0., 0., 0.), n=(0., 0. ,1.), ax=(1., 0. , 0.), ay=(0., 1., 0.), diameter=1.0, Rbasis=np.eye(4), coef=np.zeros(3), n1=1., n2=1.):
        super().__init__(p, n, ax, ay, diameter, Rbasis, coef)
        self.n1 = n1
        self.n2 = n2
        self.nratio = n1/n2
        self.jopt = JitOptic(p=self.p, n=self.n, ax=self.ax, ay=self.ay, Rot=self.Rot, rapt=self.rapt, coef=self.coef, nratio=self.nratio, otype=8)
        
        m = np.identity(4)
        m[2,2] = self.nratio
        m[3,3] = self.nratio

        if abs(coef[2])>0.: #quadratic curvature non-zero
            Req_inv = -2*coef[2] #WHY minus -> actually the right sign due to different deffinitions of CX/CC vs coeff
            m[2,0] = (n1-n2)/n2*Req_inv
            m[3,1] = (n1-n2)/n2*Req_inv
        self.m = m