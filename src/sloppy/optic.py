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
    def __init__(self, p=(0., 0., 0.), n=(0., 0. ,1.), ax=(1., 0. , 0.), ay=(0., 1., 0.), diameter=1.0, Rbasis=np.identity(4, dtype=np.float64)):
        self.rapt = diameter/2.
        self.p = np.array(p, dtype=np.float64)
        self.n = np.array(norm(n), dtype=np.float64)
        self.ax = np.array(norm(ax), dtype=np.float64)
        self.ay = np.array(norm(ay), dtype=np.float64)
        cross_product = np.cross(self.ax, self.ay)
        # asser the coordinate system is right handed
        if not np.allclose(cross_product, self.n):
            raise RuntimeError("Coordinate system is not right handed")
        
        self.Rot = np.stack((self.ax, self.ay, self.n)).T
        self.Rbasis = Rbasis
        #create JitOptic
        self._init_jopt()
        self.m = np.identity(4, dtype=np.float64)

    def _init_jopt(self):
        self.jopt = JitOptic(p=self.p, n=self.n, ax=self.ax, ay=self.ay, Rot=self.Rot, rapt=self.rapt, R=0., nratio=1.0, otype=0)
        
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
    
    def get_abcd(self, traversal_direction):
        """Get the ABCD matrix adjusted for the given traversal direction.
        
        Attributes:
            traversal_direction (ndarray): Vector indicating the ray propagation direction
            
        Returns:
            Modified ABCD matrix for the correct traversal direction
        """
        # Check if ray is traveling with or against the element's normal
        traversal_direction = np.asarray(traversal_direction)
        aligned_with_normal = np.dot(traversal_direction, self.n) > 0
        
        # For most elements (mirrors, screens), direction doesn't matter
        return self.m
    
class Mirror(Optic):
    """Plane mirror, no extra arguments."""
    def __init__(self, p=(0., 0., 0.), n=(0., 0. ,1.), ax=(1., 0. , 0.), ay=(0., 1., 0.), diameter=1.0, Rbasis=np.identity(4, dtype=np.float64)):
        super().__init__(p=p, n=n, ax=ax, ay=ay, diameter=diameter, Rbasis=Rbasis)
        
class Screen(Optic):
    """Plane mirror, no extra arguments."""
    def __init__(self, p=(0., 0., 0.), n=(0., 0. ,1.), ax=(1., 0. , 0.), ay=(0., 1., 0.), diameter=1.0, Rbasis=np.identity(4, dtype=np.float64)):
        super().__init__(p=p, n=n, ax=ax, ay=ay, diameter=diameter, Rbasis=Rbasis)
    
    def _init_jopt(self):
        self.jopt = JitOptic(p=self.p, n=self.n, ax=self.ax, ay=self.ay, Rot=self.Rot, rapt=self.rapt, R=0., nratio=1.0, otype=1)
    
    def aberrations(self, chi):
        raise NotImplementedError("aberrations not defined for this element")

class Source(Screen):
    """A special screen that marks the start of a cavity traversal."""
    
    def __init__(self, p=(0., 0., 0.), n=(0., 0., 1.), ax=(1., 0., 0.), 
                 ay=(0., 1., 0.), diameter=1.0, Rbasis=np.identity(4, dtype=np.float64)):
        
        # Forward direction is along the normal
        self.forward_direction = np.array(n)
        super().__init__(p=p, n=n, ax=ax, ay=ay, diameter=diameter, Rbasis=Rbasis)
        

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
    def __init__(self, p=(0., 0., 0.), n=(0., 0. ,1.), ax=(1., 0. , 0.), ay=(0., 1., 0.), diameter=1.0, R=1., curv='CC', thet=0., Rbasis=np.identity(4, dtype=np.float64)):
        
        self.R = abs(R)
        
        self.curv = curv
        self.thet = thet #TODO fix this from coordinate system.
        super().__init__(p=p, n=n, ax=ax, ay=ay, diameter=diameter, Rbasis=Rbasis)

    def _init_jopt(self):
        if self.curv == 'CC':
            self.jopt = JitOptic(p=self.p, n=self.n, ax=self.ax, ay=self.ay, Rot=self.Rot, rapt=self.rapt, R=abs(self.R), nratio=1.0, otype=2)
        elif self.curv == 'CX':
            self.jopt = JitOptic(p=self.p, n=self.n, ax=self.ax, ay=self.ay, Rot=self.Rot, rapt=self.rapt, R=abs(self.R), nratio=1.0, otype=3)
        else:
            raise ValueError("Mirror type {} unknown! Curv has to be CC or CX".format(self.curv))
        
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
    def __init__(self, p=(0., 0., 0.), n=(0., 0. ,1.), ax=(1., 0. , 0.), ay=(0., 1., 0.), diameter=1.0, n1=1., n2=1., Rbasis=np.identity(4, dtype=np.float64)):
        self.n1 = n1
        self.n2 = n2
        self.nratio = n1/n2
        super().__init__(p=p, n=n, ax=ax, ay=ay, diameter=diameter, Rbasis=Rbasis)

    def _init_jopt(self):
        self.jopt = JitOptic(p=self.p, n=self.n, ax=self.ax, ay=self.ay, Rot=self.Rot, rapt=self.rapt, nratio=self.nratio, otype=4)

    def get_abcd(self, traversal_direction):
        """Get the ABCD matrix for the correct traversal direction."""
        # Check if ray is traveling with or against the element's normal
        aligned_with_normal = np.dot(np.asarray(traversal_direction), self.n) > 0
        
        # Create the corrected matrix
        m = np.identity(4, dtype=np.float64)
        
        if aligned_with_normal:
            # Ray traversing WITH normal: n2 to n1
            m[2,2] = self.n2/self.n1
            m[3,3] = self.n2/self.n1
        else:
            # Ray traversing AGAINST normal: n1 to n2
            m[2,2] = self.n1/self.n2
            m[3,3] = self.n1/self.n2

        return m
        
class CurvedGlass(Glass):
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
    def __init__(self, p=(0., 0., 0.), n=(0., 0. ,1.), ax=(1., 0. , 0.), ay=(0., 1., 0.), diameter=1.0, R=1., curv='CC', n1=1., n2=1., Rbasis=np.identity(4, dtype=np.float64)):
        self.n1 = n1
        self.n2 = n2
        self.nratio = n1/n2
        self.R = R
        self.curv = curv
        super().__init__(p=p, n=n, ax=ax, ay=ay, diameter=diameter, n1=n1, n2=n2, Rbasis=Rbasis)


    def _init_jopt(self):
        if self.curv == 'CC':
            self.jopt = JitOptic(p=self.p, n=self.n, ax=self.ax, ay=self.ay, Rot=self.Rot, rapt=self.rapt, R=abs(self.R), nratio=self.nratio, otype=5)
        elif self.curv == 'CX':
            self.jopt = JitOptic(p=self.p, n=self.n, ax=self.ax, ay=self.ay, Rot=self.Rot, rapt=self.rapt, R=abs(self.R), nratio=self.nratio, otype=6)
        else:
            raise ValueError("Mirror type {} unknown! Curv has to be CC or CX".format(self.curv))

    @property
    def Reff(self):
        """Calculate effective radius of curvature for ABCD matrix.
        
        Returns:
            float: Effective radius with sign indicating curvature
        """
        # Return signed radius based on curvature type
        return -self.R if self.curv == 'CC' else self.R
    
    def get_abcd(self, traversal_direction):
        """Get the ABCD matrix for a curved interface with direction awareness."""
        # First get the basic refractive matrix from parent
        m = super().get_abcd(traversal_direction) 
        
        # Check ray alignment with normal
        aligned_with_normal = np.dot(traversal_direction, self.n) > 0
        
        # Add curvature effect with proper sign
        # The effective Reff already has the right sign for CC/CX
        if aligned_with_normal:
            Reff = -self.Reff
        else:
            Reff = self.Reff

        # print('curved glass, aligned_with_normal', aligned_with_normal, Reff)
        if aligned_with_normal:
            # Ray traversing WITH normal: n1 to n2
            m[2,0] = (self.n2-self.n1)/(Reff*self.n1)
            m[3,1] = (self.n2-self.n1)/(Reff*self.n1)
        else:
            # Ray traversing AGAINST normal: n2 to n1
            m[2,0] = (self.n1-self.n2)/(Reff*self.n2)
            m[3,1] = (self.n1-self.n2)/(Reff*self.n2)
            
        return m
        
    def plot(self, n_radii = 10, n_angles = 10, **kwargs):
        x, y = disc_coords(n_radii = n_radii, n_angles = n_angles, R=self.rapt)
        R = abs(self.R)
        z = R-np.sqrt(R**2-x**2-y**2)
        if self.curv=='CX':
            z = -z
        #if self.curv=='CC':
        #    z = self.R-np.sqrt(self.R**2-x**2-y**2)
        #else:
        #    z = -self.R+np.sqrt(self.R**2-x**2-y**2)
        indices = Triangulation(x,y).triangles.astype(np.uint32)
        mesh = k3d.mesh(np.vstack([x,y,z]).T, indices, **kwargs)
        mesh.transform.translation = self.p
        mesh.transform.custom_matrix = pad3to4(self.Rot).astype(np.float32)
        return mesh
    
    def aberrations(self, chi):
        #shape of chi is (4, N, N)
        x, y, sx, sy = chi
        c2 = 1./(2*self.R)
        c4 = 1./(8*self.R**3)
        n, m = self.n1, self.n2
        px, py = self.n1*sx, self.n1*sy
        x2, y2 = x@x, y@y
        x2py2 = x2+y2
        pxx, pyy = px@x, py@y
        
        sph4 = (n-m)*( 2*c2**2/m*(pxx+pyy)@x2py2\
                    + c2/3.*(1./(2*m*n)-1)*(px@px+py@py)@x2py2\
                    + c2/3.*(1+1./(m*n))*(pxx+pyy)@(pxx+pyy)\
                    + (2*(n/m-1)*c2**3+c4)*x2py2@x2py2)
        
        return sph4
    
class FreeFormMirror(Mirror):
    """Free form radially symmetric optic of the form z = \sum_i=0^deg coef[i] r**i
    Args:
        coef (ndarray): coefficients
    """
    def __init__(self, p=(0., 0., 0.), n=(0., 0. ,1.), ax=(1., 0. , 0.), ay=(0., 1., 0.), diameter=1.0, Rbasis=np.identity(4, dtype=np.float64), coef=np.zeros(3)):
        self.coef = coef
        super().__init__(p=p, n=n, ax=ax, ay=ay, diameter=diameter, Rbasis=Rbasis)
        
        
    def _init_jopt(self):
        self.jopt = JitOptic(p=self.p, n=self.n, ax=self.ax, ay=self.ay, Rot=self.Rot, rapt=self.rapt, coef=self.coef, otype=7)


    def plot(self, n_radii = 10, n_angles = 10, **kwargs):
        x, y = disc_coords(n_radii = n_radii, n_angles = n_angles, R=self.rapt)
        z = np.polyval(self.coef[::-1], np.sqrt(x**2 + y**2))
        indices = Triangulation(x,y).triangles.astype(np.uint32)
        mesh = k3d.mesh(np.vstack([x,y,z]).T, indices, **kwargs)
        mesh.transform.translation = self.p
        mesh.transform.custom_matrix = pad3to4(self.Rot).astype(np.float32)
        return mesh
    
class FreeFormInterface(CurvedGlass):
    """Free form radially symmetric optic of the form z = \sum_i=0^deg coef[i] r**i
    Args:
        coef (ndarray): coefficients
    """
    def __init__(self, p=(0., 0., 0.), n=(0., 0. ,1.), ax=(1., 0. , 0.), ay=(0., 1., 0.), diameter=1.0, Rbasis=np.identity(4, dtype=np.float64), coef=np.zeros(3), n1=1., n2=1.):
        self.coef = np.array(coef, dtype=np.float64)
        # call the parent constructor with named arguments
        super().__init__(p=p, n=n, ax=ax, ay=ay, diameter=diameter, Rbasis=Rbasis, n1=n1, n2=n2)

    def _init_jopt(self):
        self.jopt = JitOptic(p=self.p, n=self.n, ax=self.ax, ay=self.ay, Rot=self.Rot, rapt=self.rapt, coef=self.coef, nratio=self.nratio, otype=8)

    @property
    def Reff(self):
        """Calculate effective radius of curvature for ABCD matrix.
        
        Returns:
            float: Effective radius with sign indicating curvature
        """
        return -1/(2*self.coef[2]) # match sag of spherical surface

    def plot(self, n_radii = 10, n_angles = 10, **kwargs):
        x, y = disc_coords(n_radii = n_radii, n_angles = n_angles, R=self.rapt)
        z = np.polyval(self.coef[::-1], np.sqrt(x**2 + y**2))
        indices = Triangulation(x,y).triangles.astype(np.uint32)
        mesh = k3d.mesh(np.vstack([x,y,z]).T, indices, **kwargs)
        mesh.transform.translation = self.p
        mesh.transform.custom_matrix = pad3to4(self.Rot).astype(np.float32)
        return mesh
        
    def aberrations(self, chi):
        #shape of chi is (4, N, N)
        x, y, sx, sy = chi
        c2 = -self.coef[2]
        c4 = -self.coef[4]
        n, m = self.n1, self.n2
        px, py = self.n1*sx, self.n1*sy
        x2, y2 = x@x, y@y
        x2py2 = x2+y2
        pxx, pyy = px@x, py@y
        
        sph4 = (n-m)*( 2*c2**2/m*(pxx+pyy)@x2py2\
                    + c2/3.*(1./(2*m*n)-1)*(px@px+py@py)@x2py2\
                    + c2/3.*(1+1./(m*n))*(pxx+pyy)@(pxx+pyy)\
                    + (2*(n/m-1)*c2**3+c4)*x2py2@x2py2)
        
        return sph4
    
class ThorlabsAsphere(FreeFormInterface):
    """Free form radially symmetric optic of the form z = \sum_i=0^deg coef[i] r**i
    Args:
        coef (ndarray): coefficients [R, k, A2, A4, ...]
    """
    def __init__(self, p=(0., 0., 0.), n=(0., 0. ,1.), ax=(1., 0. , 0.), ay=(0., 1., 0.), diameter=1.0, Rbasis=np.identity(4, dtype=np.float64), coef=np.zeros(3), n1=1., n2=1.): 
        self.curv = 'CX'
        super().__init__(p=p, n=n, ax=ax, ay=ay, diameter=diameter, Rbasis=Rbasis, coef=coef, n1=n1, n2=n2)

    def _init_jopt(self):
        self.jopt = JitOptic(p=self.p, n=self.n, ax=self.ax, ay=self.ay, Rot=self.Rot, rapt=self.rapt, coef=self.coef, nratio=self.nratio, otype=9)
    
    @property
    def Reff(self):
        """Calculate effective radius of curvature for ABCD matrix.
        
        Returns:
            float: Effective radius with sign indicating curvature
        """
        
        return -self.coef[0] # match sag of spherical surface compared to thorlabs definition
        
    def aberrations(self, chi):
        #shape of chi is (4, N, N)
        x, y, sx, sy = chi
        c2 = -self.coef[2]
        c4 = -self.coef[4]
        n, m = self.n1, self.n2
        px, py = self.n1*sx, self.n1*sy
        x2, y2 = x@x, y@y
        x2py2 = x2+y2
        pxx, pyy = px@x, py@y
        
        sph4 = (n-m)*( 2*c2**2/m*(pxx+pyy)@x2py2\
                    + c2/3.*(1./(2*m*n)-1)*(px@px+py@py)@x2py2\
                    + c2/3.*(1+1./(m*n))*(pxx+pyy)@(pxx+pyy)\
                    + (2*(n/m-1)*c2**3+c4)*x2py2@x2py2)
        
        return sph4
    
class MicroLensArray(CurvedGlass):
    """A microlens array with spherical lenslets arranged in a grid pattern."""
    
    def __init__(self, p=(0., 0., 0.), n=(0., 0., 1.), ax=(1., 0., 0.), ay=(0., 1., 0.), 
                 diameter=1.0, R=1., curv='CC', n1=1., n2=1., 
                 a1=(0.1, 0), a2=(0, 0.1), origin_centered=True, Rbasis=np.identity(4, dtype=np.float64)):
        """
        Args:
            p: Position of the MLA
            n: Normal vector 
            ax, ay: Local coordinate system vectors
            diameter: Overall diameter of the MLA
            R: Radius of curvature of individual lenslets
            curv: 'CC' or 'CX' - curvature type for lenslets
            n1: Refractive index on incident side
            n2: Refractive index on exit side
            a1, a2: Lattice vectors defining the grid of lenslets
            origin_centered: If True, the first lenslet is centered at the origin
        """
        self.a1 = np.array(a1, dtype=np.float64)
        self.a2 = np.array(a2, dtype=np.float64)
        self.origin_centered = origin_centered
        self.curv = curv
        super().__init__(p=p, n=n, ax=ax, ay=ay, diameter=diameter, Rbasis=Rbasis, curv=curv, R=R, n1=n1, n2=n2)
        
        # Create JitOptic with MLA type
    def _init_jopt(self):
        if self.curv == 'CC':
            self.jopt = JitOptic(p=self.p, n=self.n, ax=self.ax, ay=self.ay, Rot=self.Rot, 
                               rapt=self.rapt, R=self.R, nratio=self.nratio, otype=11,
                               a1=self.a1, a2=self.a2, origin_centered=self.origin_centered)
        elif self.curv == 'CX':
            self.jopt = JitOptic(p=self.p, n=self.n, ax=self.ax, ay=self.ay, Rot=self.Rot, 
                               rapt=self.rapt, R=self.R, nratio=self.nratio, otype=12,
                               a1=self.a1, a2=self.a2, origin_centered=self.origin_centered)
        else:
            raise ValueError(f"Curvature type {self.curv} unknown! Must be CC or CX")
            
    def plot(self, n_radii=10, n_angles=10, lattice_size=5, **kwargs):
        """Plot the microlens array with multiple lenslets."""
        import k3d
        from matplotlib.tri import Triangulation
        
        # Calculate lattice extent
        half_size = lattice_size // 2
        points = []
        faces = []
        vertex_count = 0
        
        # Generate a mesh for each lenslet
        for i in range(-half_size, half_size + 1):
            for j in range(-half_size, half_size + 1):
                # Calculate lenslet center
                lenslet_center = i * self.a1 + j * self.a2
                if self.origin_centered:
                    lenslet_center += 0.5 * (self.a1 + self.a2)
                
                # Create lenslet surface
                x, y = disc_coords(n_radii=n_radii, n_angles=n_angles, 
                                   R=min(np.linalg.norm(self.a1), np.linalg.norm(self.a2)) / 2)
                
                # Shift to lenslet center
                x = x + lenslet_center[0]
                y = y + lenslet_center[1]
                
                # Calculate z based on spherical surface
                r_sqr = (x - lenslet_center[0])**2 + (y - lenslet_center[1])**2
                mask = r_sqr < self.R**2  # Only valid points
                
                if not np.any(mask):
                    continue  # Skip this lenslet if no valid points
                
                # Calculate z values based on curvature
                z = np.zeros_like(x)
                if self.curv == 'CC':
                    z[mask] = self.R - np.sqrt(self.R**2 - r_sqr[mask])
                else:  # 'CX'
                    z[mask] = -self.R + np.sqrt(self.R**2 - r_sqr[mask])
                
                # Create triangulation for this lenslet
                tri = Triangulation(x, y).triangles
                
                # Add vertices and faces to the global lists
                points.append(np.vstack([x, y, z]).T)
                faces.append(tri + vertex_count)
                vertex_count += len(x)
        
        # Combine all vertices and faces
        points = np.vstack(points)
        faces = np.vstack(faces).astype(np.uint32)
        
        # Create the mesh
        mesh = k3d.mesh(points, faces, **kwargs)
        mesh.transform.translation = self.p
        mesh.transform.custom_matrix = pad3to4(self.Rot).astype(np.float32)
        return mesh
    
class OffAxisParabolicMirror(CurvedMirror):
    """Off-axis parabolic mirror segment.
    
    This implements an off-axis segment of a parabolic mirror as shown in the diagram.
    The point P (position parameter) corresponds to the center of the mirror segment.
    
    Attributes:
        efl: Effective focal length (from mirror center to focal point)
        angle: Angle between central ray and optical axis
        diameter: Diameter of the mirror segment
    """
    
    def __init__(self, p=(0., 0., 0.), n=(0., 0., 1.), ax=(1., 0., 0.), ay=(0., 1., 0.), 
                 diameter=1.0, efl=1.0, angle=np.radians(30), Rbasis=np.identity(4, dtype=np.float64)):
        """Initialize the off-axis parabolic mirror.
        
        Args:
            p: Position of the mirror segment center (point P in diagram)
            n: Normal vector at the center of the mirror segment
            ax, ay: Local coordinate system vectors
            diameter: Diameter of the mirror segment
            efl: Effective focal length (reflected EFL in diagram)
            angle: Angle between central ray and axis in radians
            Rbasis: Rotation basis
        """
        # Store the parameters
        self.efl = efl
        self.angle = angle
        
        # Calculate parent focal length and y-offset
        self.parent_focal_length = self.efl / np.cos(self.angle)
        self.y_offset = self.parent_focal_length * np.sin(self.angle)
        
        # Calculate parameters A, B, C from the diagram
        self.A = self.parent_focal_length * np.sin(self.angle) * np.tan(self.angle/2)
        self.B = self.efl  # Reflected EFL is B in the diagram
        self.C = self.parent_focal_length * np.sin(self.angle)**2 / (1 + np.cos(self.angle))
        
        # Calculate the effective radius of curvature at the center point
        # For a parabola, local R = 2 * parent_focal_length / (cos(θ))^3 at off-axis point
        R_effective = 2 * self.parent_focal_length / (np.cos(self.angle)**3)
        
        # Initialize the parent class with the effective radius
        super().__init__(p=p, n=n, ax=ax, ay=ay, diameter=diameter, R=R_effective, curv='CC', Rbasis=Rbasis)
        
        # Override the standard jopt with our custom JitOptic
        self.jopt = JitOptic(
            p=self.p, n=self.n, ax=self.ax, ay=self.ay, Rot=self.Rot, 
            rapt=self.rapt, parent_focal_length=self.parent_focal_length, angle=self.angle, otype=13)
        
    def get_focal_point(self):
        """Calculate the focal point location in global coordinates."""
        # Focal point is efl distance from P along the central reflected ray
        # Central reflected ray direction is at angle 2*theta from axis
        central_ray_dir = np.array([
            np.sin(2*self.angle), 
            0, 
            np.cos(2*self.angle)
        ])
        # Transform to global coordinates
        central_ray_dir = self.Rot @ central_ray_dir
        
        # Focal point is efl distance from P along central ray direction
        focal_point = self.p + self.efl * central_ray_dir
        return focal_point
        
    def plot(self, n_radii=10, n_angles=10, **kwargs):
        """Plot the off-axis parabolic mirror segment."""
        import k3d
        from matplotlib.tri import Triangulation
        
        # Generate coordinates in local frame
        x, y = disc_coords(n_radii=n_radii, n_angles=n_angles, R=self.rapt)
        
        # Calculate z based on parabolic equation in local coordinates
        # The equation of a parabola with focus at origin is: z = (x² + y²)/(4*f)
        # Adjusted for off-axis segment where vertex is at (-y_offset, 0, 0)
        # and point P is at (0, 0, 0) in local coordinates
        
        # Shift coordinates to place vertex at origin
        x_shifted = x 
        y_shifted = y + self.y_offset
        
        # Calculate z using parabolic equation relative to vertex
        z = (x_shifted**2 + y_shifted**2) / (4 * self.parent_focal_length)
        
        # Shift z to place point P at z=0
        z_center = (self.y_offset**2) / (4 * self.parent_focal_length)
        z = z - z_center
        
        indices = Triangulation(x, y).triangles.astype(np.uint32)
        mesh = k3d.mesh(np.vstack([x, y, z]).T, indices, **kwargs)
        mesh.transform.translation = self.p
        mesh.transform.custom_matrix = pad3to4(self.Rot).astype(np.float32)
        return mesh
