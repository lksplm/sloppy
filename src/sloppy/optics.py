"""Optical elements

This module contains the main definition of all optical elements (mirrors, lenses, etc.).
It contains the ray tracing routines as well as ABCD matrix definitions
for all common optics.

The Ray format used throughout the package is
ray.shape = (2, Nrays, 3) where the first axis denotes position (0) and slope (1)
which are three-vectors each.
In cases where multiple rays are obtained (intersection with different element upon a roundtrip e.g.)
this is prepended as (N, 2, Nrays, 3).

Example:
    Examples can be found in the notebook folder.


Todo:
    * Complete documentation
    * Implement coordinate transformations on a k3d compatible way, especially for Screen.

"""

import numpy as np
import k3d
from matplotlib.tri import Triangulation
from .utils import *


class Optic:
    """Base class for all optical elements.

    It collects common properties (position and direction) and methods like
    coordinate transformations, intersection and plotting.

    Attributes:
        p (ndarray): Position of the element in space (origin point).
        n (ndarray): Direction of the element as normal vector.
        r (float): Radius of the elements aperture.
        m (ndarray): ABCD matrix corresponing to this element (4x4).

    """
    def __init__(self, p=(0,0,0), n=(0,0,1), diam=1.):
        self.p = np.array(p)
        self.n = np.array(norm(n))
        self.r = diam/2
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
        x, y = disc_coords(n_radii = n_radii, n_angles = n_angles, R=self.r)

        z = np.zeros_like(x)
        indices = Triangulation(x,y).triangles.astype(np.uint32)
        mesh = k3d.mesh(np.vstack([x,y,z]).T, indices, **kwargs)
        t, r = self.transformation
        mesh.transform.rotation = r
        mesh.transform.translation = t
        return mesh  
    
    def _intersect(self, ray, clip=True, p=None):
        """Main intersection method for a ray with a plane (optional aperture).
        As this is the fastest check for intersection it used my all elements before the more specific intersection (sphere...).

        The Ray format used throughout the package is
        ray.shape = (2, Nrays, 3) where the first axis denotes position (0) and slope (1)
        which are three-vectors each.
        
        Args:
            ray: Input rays with format described above.
            clip: If true, only intersections within the aperture radius are returned, outside ones return Nan.

        Returns:
            x: poitions of ray intersections.

        """
        x = np.full((ray.shape[1], 3), np.nan)
        
        r, s = ray
        sn = s@self.n
        #prevent ray perpendicular to surface
        msk = np.abs(sn)>np.finfo(np.float32).eps
        t = ((p - r[msk])@self.n)/sn[msk]
        x[msk,:] = r[msk,:] + t[:,None]*s[msk,:]
        
        if clip:
            d = np.linalg.norm(x - p, axis=1)
            #msk[msk] = (d<=self.r)
            x[(d>self.r),:] = np.nan
        return x
    
    def intersect(self, ray, clip=True):
        """
        Wrapper for _intersect that can be overwritten by other functions.
        """
        return self._intersect(ray, clip=clip, p=self.p)
        
    def propagate(self, ray, clip=True):
        """Propagation of input rays through an element.
        For this base class it returns the (clipped) input rays.
        
        Args:
            ray: Input rays with shape (2, Nrays, 3).
            clip: If true, only intersections within the aperture radius are returned, outside ones return Nan.

        Returns:
            rout: Output rays with shape (2, Nrays, 3).
        """
        rout = np.full_like(ray, np.nan)
        
        q = self.intersect(ray, clip=clip)
        msk = ~np.isnan(q[:,0])
        s = ray[1,msk,:]
        
        rout[0,:,:] = q
        rout[1,msk,:] = s
        return rout

class Mirror(Optic):
    """Plane mirror, no extra arguments.

    """
    def __init__(self, p=(0,0,0), n=(0,0,1), diam=1.):
        super().__init__(p, n, diam)
        
    #TODO add opional reflection in sagital plane?
    def propagate(self, ray, clip=True):
        """Propagation of input rays as reflection from plane mirror.
        
        Args:
            ray: Input rays with shape (2, Nrays, 3).
            clip: If true, only intersections within the aperture radius are returned, outside ones return Nan.

        Returns:
            rout: Output rays with shape (2, Nrays, 3).
        """
        rout = np.full_like(ray, np.nan)
        
        q = self.intersect(ray, clip=clip)
        msk = ~np.isnan(q[:,0])
        s = ray[1,msk,:]
        
        sp = s - 2*np.dot(s, self.n)[:,None]*self.n[None,:]
        rout[0,:,:] = q
        rout[1,msk,:] = sp
        return rout
    


class Glass(Optic):
    """Interface in the indecies of refraction i.e. a glass surface.
    Args:
        n1 (float): Index of refraction before the interface.
        n2 (float): Index of refraction after the interface.

    """
    def __init__(self, p=(0,0,0), n=(0,0,1), diam=1., n1=1., n2=1.):
        super().__init__(p, n, diam)
        self.n1 = n1
        self.n2 = n2
        self.nr = n1/n2
        
        m = np.identity(4)
        m[2,2] = n1/n2
        m[3,3] = n1/n2
        self.m = m
        
    def propagate(self, ray, clip=True):
        """Propagation of input rays by transmission through an index change, Snells law.
        
        Args:
            ray: Input rays with shape (2, Nrays, 3).
            clip: If true, only intersections within the aperture radius are returned, outside ones return Nan.

        Returns:
            rout: Output rays with shape (2, Nrays, 3).
        """
        rout = np.full_like(ray, np.nan)
        
        q = self.intersect(ray, clip=clip)
        msk = ~np.isnan(q[:,0])
        s = ray[1,msk,:]
        #make sure there is always transmission and no reflection!
        #c = -np.dot(s, self.n)
        sn = np.dot(s, self.n)
        c = -sn
        #fudge for now to fix this, seems robust!
        f = np.where(sn>0,-1,1)
        
        dis = 1 - self.nr**2*(1 - c**2)
        #prevent total internal reflection
        msk[msk] = (dis>=0.)
        sp = self.nr*s + (f*self.nr*c - f*np.sqrt(dis))[:,None]*self.n[None,:]
        
        rout[0,:,:] = q
        rout[1,msk,:] = sp
        return rout

class CurvedMirror(Optic):
    """Curved reflecting mirror, can be conccave ('CC') or convex ('CX').
    
    Args:
        R (float): Radius of curvature, always positive!
        curv (string): 'CC' or 'CX' always w.r.t. to the normal vector.
            Schematically this looks like this:
                CC: ----- <-)
                CX: ----- <-(
        thet (float): angle of reflection for the ABCD matrix in sagital/tangential plane.
        TODO: Implement thet

    """
    def __init__(self, p=(0,0,0), n=(0,0,1), R=1., curv='CC', diam=25.4, thet=0.):
        super().__init__(p, n, diam)
        self.R = R
        x, dst = self.curved_offsets(self.R, self.r)
        self.intersect_d = dst
        if curv=='CC':
            self.cp = self.p + self.R*self.n
            self.poffs = self.p + self.n*x
        elif curv=='CX':
            self.cp = self.p - self.R*self.n
            self.poffs = self.p - self.n*x
        else:
            raise ValueError("Mirror type {} unknown! Curv has to be CC or CX".format(curv))
        self.curv = curv
        
        self.thet = thet
        m = np.identity(4)
        m[2,0] = -2/R*np.cos(thet) #sagital
        m[3,1] = -2/R/np.cos(thet) #tangential
        self.m = m
    
    @staticmethod
    def curved_offsets(R, rapt):
        """calculates the on axis distance x and diag. distance dst for a curved mirror of finite apperture"""
        x =  R - np.sqrt(R**2 -  rapt**2)
        dst = np.sqrt(rapt**2+x**2)
        return x, dst    

    def plot(self, n_radii = 10, n_angles = 10, **kwargs):
        x, y = disc_coords(n_radii = n_radii, n_angles = n_angles, R=self.r)

        z = self.R-np.sqrt(self.R**2-x**2-y**2)
        if self.curv=='CX':
            z = -z
        indices = Triangulation(x,y).triangles.astype(np.uint32)
        mesh = k3d.mesh(np.vstack([x,y,z]).T, indices, **kwargs)
        t, r = self.transformation
        mesh.transform.rotation = r
        mesh.transform.translation = t
        return mesh
    
    def intersect(self, ray, clip=True):
        #do flat intersection first!
        q = self._intersect(ray, clip=clip, p=self.poffs)
        msk = ~np.isnan(q[:,0])
        r = ray[0,msk,:]
        s = ray[1,msk,:]

        d = self.cp - r
        ds = np.einsum("ji,ji->j",d,s)
        
        dis = ds**2 + self.R**2 - np.einsum("ji,ji->j",d,d)
        msk[msk] = dis>=0
        t12 = np.stack([ds+np.sqrt(dis), ds-np.sqrt(dis)], axis=0) #[2xNrays]
        #find the right intersection closer to the origin!
        x12 = r[None,:] + t12[:,:,None]*s[None,:] #[2xNraysx3]
        dist12 = np.linalg.norm(x12-self.p, axis=2) #[2xNrays]
        which = np.argmin(dist12, axis=0)
        x = x12[which,np.arange(x12.shape[1]),:]
        
        if clip:
            d = np.linalg.norm(x - self.p, axis=1)
            msk[msk] = (d<=self.intersect_d)
        
        xout = np.full_like(q, np.nan)
        xout[msk] = x
        return xout

        
    def propagate(self, ray, clip=True):
        """Propagation of input rays by reflection at the normal vector of the sphere.
        
        Args:
            ray: Input rays with shape (2, Nrays, 3).
            clip: If true, only intersections within the aperture radius are returned, outside ones return Nan.

        Returns:
            rout: Output rays with shape (2, Nrays, 3).
        """
        q = self.intersect(ray, clip=clip)
        msk = ~np.isnan(q[:,0])
        r = ray[0,msk,:]
        s = ray[1,msk,:]
        
        #find normal vectors
        n = self.cp - q[msk]
        nn = np.linalg.norm(n, axis=1)
        n *= 1./nn[:,None]
        
        sp = s - 2*np.einsum("ji,ji->j",s,n)[:,None]*n
        rout = np.full_like(ray, np.nan)
        rout[0,:,:] = q
        rout[1,msk,:] = sp
        return rout

class CurvedGlass(CurvedMirror):
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
    def __init__(self, p=(0,0,0), n=(0,0,1), R=1., curv='CC', diam=25.4, thet=0., n1=1., n2=1.):
        super().__init__(p, n, R, curv, diam)
        self.n1 = n1
        self.n2 = n2
        self.nr = n1/n2
        
        self.thet = thet
        m = np.identity(4)
        m[2,2] = n1/n2
        m[3,3] = n1/n2
        m[2,0] = (n1-n2)/(R*n2) #TODO sagittal and tangential plane
        m[3,1] = (n1-n2)/(R*n2)
        self.m = m
        
    def propagate(self, ray, indices=None, clip=True):
        """Propagation of input rays by transmission through a curved index change, Snells law with the normal vector on a sphere.
        
        Args:
            ray: Input rays with shape (2, Nrays, 3).
            clip: If true, only intersections within the aperture radius are returned, outside ones return Nan.

        Returns:
            rout: Output rays with shape (2, Nrays, 3).
        """
        q = self.intersect(ray, clip=clip)
        msk = ~np.isnan(q[:,0])
        r = ray[0,msk,:]
        s = ray[1,msk,:]
        
        #find normal vectors
        n = self.cp - q[msk]
        nn = np.linalg.norm(n, axis=1)
        n *= 1./nn[:,None]
        #if self.curv=='CX':
        #    n = -n
        
        #make sure there is always transmission and no reflection!
        #c = -np.einsum("ji,ji->j",s,n)
        sn = np.einsum("ji,ji->j",s,n)
        c = -sn
        #fudge for now to fix this, seems robust!
        f = np.where(sn>0,-1,1)
        
        dis = 1 - self.r**2*(1 - c**2)
        msk2 = (dis>=0.)
        msk[msk] = msk2
        sp = self.r*s + (f*self.r*c - f*np.sqrt(dis))[:,None]*n
        
        rout = np.full_like(ray, np.nan)
        rout[0,:,:] = q
        rout[1,msk,:] = sp[msk2]
        return rout    
    
class Screen(Optic):
    def __init__(self, p=(0,0,0), n=(0,0,1), diam=25.4):
        super().__init__(p, n, diam)