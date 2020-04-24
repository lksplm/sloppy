"""
This is the development version of numba accelerated raytracing!
"""

import numpy as np
from numba import jit, jitclass, njit
from numba import boolean, int32, float32, float64    # import the types
import numba as nb
import math
spec_curvm = [
    ('clip', boolean),
    ('otype', int32), #serves as a optic type switch: 0: flat mirror, 1: CC mirror, 2: CX mirror
    ('rapt', float64),
    ('R', float64),
    ('intersect_d', float64),
    ('p', float64[:]),
    ('n', float64[:]),
    ('cp', float64[:]),
    ('poffs', float64[:]),
    ('ray', float64[:,:])
]

#make some faster versions of core numerics
@njit('float64(float64[:], float64[:])')
def dot(a, b):
    N = len(a)
    acc = 0.
    for i in range(N):
        acc += a[i]*b[i]
    return acc
    
@njit('float64(float64[:])')
def fnorm(v):
    return math.sqrt(dot(v,v))

@jitclass(spec_curvm)
class Optic(object):
    def __init__(self, p, n, diameter, R=0., otype=0):
        self.rapt = diameter/2.
        self.p = p
        self.n = n
        self.R = abs(R)
        self.otype = otype
        if self.otype == 1: #CC mirror
            x = self.R - np.sqrt(self.R**2 -  self.rapt**2)
            self.intersect_d = np.sqrt(self.rapt**2+x**2)
            self.cp = self.p + self.R*self.n
            self.poffs = self.p + self.n*x
        elif self.otype==2: #CX mirror
            x = self.R - np.sqrt(self.R**2 -  self.rapt**2)
            self.intersect_d = np.sqrt(self.rapt**2+x**2)
            self.cp = self.p - self.R*self.n
            self.poffs = self.p - self.n*x
        else: #flat mirror
            self.intersect_d = self.rapt
            self.cp = self.p
            self.poffs = self.p

    def _intersect_flat(self, ray, clip=True):
        r = ray[0,:]
        s = ray[1,:]
        sn = dot(s, self.n)
        #prevent ray perpendicular to surface
        if np.abs(sn)<=np.finfo(np.float32).eps:
            return r*np.inf
        t = dot((self.poffs - r), self.n)/sn
        x = r + t*s
        
        if clip>0:
            dist = fnorm(x - self.poffs)
            #d = np.sqrt(np.sum((x - self.poffs)**2))
            #msk[msk] = (d<=self.r)
            if dist>self.rapt:
                return r*np.inf
        return x
    
    def _intersect_curv(self, ray, clip=True):
        #do flat intersection first!
        #q = self._intersect_flat(ray, clip=clip)
        r = ray[0,:]
        s = ray[1,:]

        d = self.cp - r
        ds = dot(d, s)
        
        dis = ds**2 + self.R**2 - dot(d, d)
        if dis < 0.0:
            return r*np.inf

        t1 = ds+np.sqrt(dis)
        t2 = ds-np.sqrt(dis)
        #find the right intersection closer to the origin!
        x1 = r + t1*s
        x2 = r + t2*s
        
        if fnorm(x1-self.p) <= fnorm(x2-self.p):
            x = x1
        else:
            x = x2

        if clip:
            dist = fnorm(x - self.p)
            if dist>self.intersect_d:
                return r*np.inf
        
        return x
        
    def _propagate_flat(self, ray, clip=True):
        q = self._intersect_flat(ray, clip=clip)
        s = ray[1,:]
    
        sp = s - 2*dot(self.n, s)*self.n
        rout = np.vstack((q, sp))
        return rout
    
    def _propagate_curv(self, ray, clip=True):
        q = self._intersect_curv(ray, clip=clip)
        s = ray[1,:]
    
        #find normal vectors
        n = self.cp - q
        n = n/fnorm(n)
        
        sp = s - 2*dot(s, n)*n
        sp = sp/fnorm(sp)
        rout = np.vstack((q, sp))
        return rout
    
    def intersect(self, ray, clip=True):
        if self.otype == 1 or self.otype == 2: #Curved mirror
            return self._intersect_curv(ray, clip)
        else:
            return self._intersect_flat(ray, clip)
        
    def propagate(self, ray, clip=True):
        if self.otype == 1 or self.otype == 2: #Curved mirror
            return self._propagate_curv(ray, clip)
        else:
            return self._propagate_flat(ray, clip)
        
spec_screen = [
    ('rapt', float64),
    ('p', float64[:]),
    ('n', float64[:]),
    ('ax', float64[:]),
    ('ay', float64[:]),
    ('R', float64[:,:]),
    ('RT', float64[:,:]),
    ('ray', float64[:,:]),
    ('rays', float64[:,:,:]),
    ('mu', float64[:]),
    ('mus', float64[:,:]),
    ('v', float64[:])
]
  
@jitclass(spec_screen)
class Screen(object):
    def __init__(self, p, n, diameter, ax, ay):
        self.rapt = diameter/2.
        self.p = p
        self.n = n
        self.ax = ax
        self.ay = ay
        self.RT = np.stack((self.ax, self.ay, self.n))
        self.R = self.RT.T
        
    def intersect(self, ray, clip=True):
        r = ray[0,:]
        s = ray[1,:]
        sn = dot(s, self.n)
        #prevent ray perpendicular to surface
        if np.abs(sn)<=np.finfo(np.float32).eps:
            return r*np.inf
        t = dot((self.p - r), self.n)/sn
        x = r + t*s
        
        if clip>0:
            dist = fnorm(x - self.p)
            #d = np.sqrt(np.sum((x - self.poffs)**2))
            #msk[msk] = (d<=self.r)
            if dist>self.rapt:
                return r*np.inf
        return x
    
    def propagate(self, ray, clip=True):
        q = self.intersect(ray, clip=clip)
        s = ray[1,:]
    
        rout = np.vstack((q, s))
        return rout
        
    def r_from_screen_coords(self, v):
        """Turns 2d vectors from screen coordinates with shape (2) into global coords."""
        vs = np.array([v[0], v[1], 0.0])
        vr = np.dot(self.R,vs) + self.p
        return vr
    
    def r_to_screen_coords(self, v):
        """Turns 3d vectors from global coordinates with shape (3) into screen coords with shape (2)."""
        vss = np.dot(self.RT, v-self.p)
        return vss[:-1]
    
    def s_from_screen_coords(self, mu):
        """Turns 2d slope vectors from screen coordinates with shape (2) into global normal vectors (3)."""
        s0 = np.array([mu[0], mu[1], np.sqrt(1 - (mu[0])**2 - (mu[1])**2)]) #create vector normal to screen
        s0 = np.dot(self.R,s0) #rotate into global coordinates
        return s0
        
    def s_to_screen_coords(self, v):
        """Turns 3d normal vectors from global coordinates with shape (3) into 2d slope vectors with shape (2)."""
        vss = np.dot(self.RT, v)#rotate back to screen coords
        return vss[:-1]
    
    def eigenvector_to_ray(self, mu):
        """Convert eigenvectors of ABCD matrix mu of shape (4) into 3D rays for raytracing of shape (2, 3)."""
        r0 = self.r_from_screen_coords(mu[:2])
        s0 = self.s_from_screen_coords(mu[2:])
        ray0 = np.stack((r0, s0), axis=0)
        return ray0
    
    def ray_to_eigenvector(self, ray):
        """Convert eigenvectors of ABCD matrix mu of shape (4) into 3D rays for raytracing of shape (2, 3)."""
        r0 = self.r_to_screen_coords(ray[0,:])
        s0 = self.s_to_screen_coords(ray[1,:])
        mu = np.concatenate((r0, s0), axis=0)
        return mu
    
    def eigenvectors_to_rays(self, mus):
        """Convert eigenvectors of ABCD matrix mu of shape (N,4) into 3D rays for raytracing of shape (N, 2, 3)."""
        Nrays = mus.shape[0]
        rays = np.empty((Nrays, 2, 3), dtype=np.float64)
        for i in range(Nrays):
            rays[i,:,:] = self.eigenvector_to_ray(mus[i,:])
        return rays
        
    def rays_to_eigenvectors(self, rays):
        """Convert eigenvectors of ABCD matrix mu of shape (N,4) into 3D rays for raytracing of shape (N, 2, 3)."""
        Nrays = rays.shape[0]
        mus = np.empty((Nrays, 4), dtype=np.float64)
        for i in range(Nrays):
            mus[i,:] = self.ray_to_eigenvector(rays[i,:,:])
        return mus
    
@njit
def propagate_system_screen(elements, screen, ray, Nrt=1, clip=True):
    """Propagate rays through an optical system (series of elements).
    Works exactly like :function:`propagate_system` but only stores intersection at screen for speed.

    Returns:
        trajs (ndarray): Array of rays of shape (Nrt, 2, Nrays, 3).
            The input ray is always stored as the first element.

    """
    rcur = ray.copy()
    trajs = np.empty((Nrt+1, 2, 3))
    trajs[0,:,:] = rcur
    for i in range(Nrt):
        for el in elements:
            rcur = el.propagate(rcur, clip)
        rcur = screen.propagate(rcur, clip)
        trajs[i+1,:,:] = rcur
    return trajs

@njit
def propagate_system(elements, screen, rays, Nrt=1, clip=True):
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
    rcur = ray.copy()
    Nel = len(elements)+1 #screen not counted in elements
    trajs = np.empty((Nel*Nrt+1, 2, 3))
    trajs[0,:,:] = rcur
    for i in range(Nrt):
        for j in range(Nel-1):
            rcur = elements[j].propagate(rcur, clip=clip)
            trajs[i*Nel+j+1,:,:] = rcur
        rcur = screen.propagate(rcur, clip)
        trajs[i*Nel+Nel,:,:] = rcur
        
    return allrays