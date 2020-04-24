import numpy as np
from numba import jit, jitclass, njit
from numba import boolean, int32, float32, float64    # import the types
import numba as nb
import math

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
"""
Monolithic optic class accelerated with numba
"""
spec = [
    ('clip', boolean),
    ('otype', int32), #serves as a optic type switch: 0: flat mirror, 1: CC mirror, 2: CX mirror
    ('rapt', float64), #aperture radius
    ('R', float64),
    ('intersect_d', float64),
    ('nratio', float64), #index of refraction ratio n1/n2 (left and right of the interface)
    ('p', float64[:]),
    ('n', float64[:]),
    ('cp', float64[:]),
    ('ax', float64[:]),
    ('ay', float64[:]),
    ('Rot', float64[:,:]), #rotation matrix into local coordinate system
    ('RotT', float64[:,:]),
    ('poffs', float64[:]),
    ('ray', float64[:,:]),
    ('rays', float64[:,:,:]),
    ('mu', float64[:]),
    ('mus', float64[:,:]),
    ('v', float64[:])
]
"""
otype: Main switch to set the type of optic considered>
    0 [default]: flat mirror
    1: screen
    2: CC concave mirror
    3: CX convex mirror
    4: flat interface (refractive index)
    5: CC concave interface
    6: CX convex interface
    7: free form function TODO
"""
@jitclass(spec)
class JitOptic(object):
    def __init__(self, p, n, ax, ay, Rot, rapt, R=0., nratio=1.0, otype=0):
        self.rapt = rapt
        self.p = p
        self.n = n
        self.R = abs(R)
        self.nratio = nratio
        self.ax = ax
        self.ay = ay
        self.Rot = Rot
        self.RotT = Rot.T
        
        self.otype = otype
        if self.otype == 2 or self.otype == 5: #CC
            x = self.R - np.sqrt(self.R**2 - self.rapt**2)
            self.intersect_d = np.sqrt(self.rapt**2+x**2)
            self.cp = self.p + self.R*self.n
            self.poffs = self.p + self.n*x
        elif self.otype== 3 or self.otype == 6: #CX 
            x = self.R - np.sqrt(self.R**2 - self.rapt**2)
            self.intersect_d = np.sqrt(self.rapt**2+x**2)
            self.cp = self.p - self.R*self.n
            self.poffs = self.p - self.n*x
        else: #flat
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
    
    def _propagate_screen(self, ray, clip=True):
        q = self._intersect_flat(ray, clip=clip)
        s = ray[1,:]
    
        rout = np.vstack((q, s))
        return rout
    
    def _propagate_curv(self, ray, clip=True):
        q = self._intersect_curv(ray, clip=clip)
        s = ray[1,:]
    
        #find normal vectors
        if self.otype ==2: #CC mirror
            n = self.cp - q
        else: #CX mirror
            n = q - self.cp
        n = n/fnorm(n)
        
        sp = s - 2*dot(s, n)*n
        sp = sp/fnorm(sp)
        rout = np.vstack((q, sp))
        return rout
    
    def _propagate_flat_interface(self, ray, clip=True):
        q = self._intersect_flat(ray, clip=clip)
        s = ray[1,:]
        
        #make sure there is always transmission and no reflection!
        c = -dot(s, self.n)
        if c>0.0: #normal case, ray is coming from medium 1 and refracted into medium 2
            r = self.nratio
            dis = 1 - r**2*(1 - c**2)#prevent total internal reflection?
            sp = r*s + (r*c - math.sqrt(dis))*self.n
        else: #reversed case, ray is comming from the other direction! reverse normal vec and ior ratio!
            r = 1.0/self.nratio
            c = -c
            dis = 1 - r**2*(1 - c**2)#prevent total internal reflection?
            sp = r*s + (math.sqrt(dis) - r*c)*self.n
        
        sp = sp/fnorm(sp)
        rout = np.vstack((q, sp))
        return rout
    
    def _propagate_curv_interface(self, ray, clip=True):
        q = self._intersect_curv(ray, clip=clip)
        s = ray[1,:]
    
        #find normal vectors
        if self.otype == 5: #CC interface
            n = self.cp - q
        else: #CX intterface
            n = q - self.cp
            
        n = n/fnorm(n)
        
        #make sure there is always transmission and no reflection!
        c = -dot(s, n)
        if c>0.0: #normal case, ray is coming from medium 1 and refracted into medium 2
            r = self.nratio
            dis = 1 - r**2*(1 - c**2)#prevent total internal reflection?
            sp = r*s + (r*c - math.sqrt(dis))*n
        else: #reversed case, ray is comming from the other direction! reverse normal vec and ior ratio!
            r = 1.0/self.nratio
            c = -c
            dis = 1 - r**2*(1 - c**2)#prevent total internal reflection?
            sp = r*s + (math.sqrt(dis) - r*c)*n

        sp = sp/fnorm(sp)
        rout = np.vstack((q, sp))
        return rout
    
    def intersect(self, ray, clip=True):
        if self.otype == 2 or self.otype == 3 or self.otype == 5 or self.otype == 6: #Curved mirror
            return self._intersect_curv(ray, clip)
        else:
            return self._intersect_flat(ray, clip)
        
    def propagate(self, ray, clip=True):
        if self.otype == 1: #screen
            return self._propagate_screen(ray, clip)
        elif self.otype == 2 or self.otype == 3: #Curved mirror
            return self._propagate_curv(ray, clip)
        elif self.otype == 4: #flat interface 
            return self._propagate_flat_interface(ray, clip)
        elif self.otype == 5 or self.otype == 6: #Curved interface
            return self._propagate_curv_interface(ray, clip)
        else:
            return self._propagate_flat(ray, clip)
        
    def _r_from_screen_coords(self, v):
        """Turns 2d vectors from screen coordinates with shape (2) into global coords."""
        vs = np.array([v[0], v[1], 0.0])
        vr = np.dot(self.Rot,vs) + self.p
        return vr
    
    def _r_to_screen_coords(self, v):
        """Turns 3d vectors from global coordinates with shape (3) into screen coords with shape (2)."""
        vss = np.dot(self.RotT, v-self.p)
        return vss[:-1]
    
    def _s_from_screen_coords(self, mu):
        """Turns 2d slope vectors from screen coordinates with shape (2) into global normal vectors (3)."""
        s0 = np.array([mu[0], mu[1], np.sqrt(1 - (mu[0])**2 - (mu[1])**2)]) #create vector normal to screen
        s0 = np.dot(self.Rot,s0) #rotate into global coordinates
        return s0
        
    def _s_to_screen_coords(self, v):
        """Turns 3d normal vectors from global coordinates with shape (3) into 2d slope vectors with shape (2)."""
        vss = np.dot(self.RotT, v)#rotate back to screen coords
        return vss[:-1]
    
