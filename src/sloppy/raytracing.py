import numpy as np
from numba import jit, njit, prange #jitclass,
from numba.experimental import jitclass
from numba import boolean, int32, float32, float64    # import the types
import numba as nb
import math
from .abcd import *
from .optic import *
from .joptic import *

class ConvergenceError(Exception):
    """Base class for exceptions in this module."""
    pass

class RaySystem:
    def __init__(self, elements, add_screen=True):
        if add_screen:
            #add screen to elements
            x0 = 0.5*(elements[0].p + elements[-1].p)
            n0 = norm(elements[0].p - elements[-1].p)

            screen = Screen(p=x0, n=n0, diameter=10., ax=elements[0].ax, ay=elements[0].ay)
            elements.append(screen)
            self.screen = screen
        self.elements = elements
        self.jelements = tuple((el.jopt for el in elements)) #homogenous tuple to support jitted routines
        
    # @property
    # def abcd(self):
    #     pos = [e.p for e in self.elements]

    #     abcd = []
    #     for i, el in enumerate(self.elements):
    #         d = np.linalg.norm(pos[i-1]-pos[i])
    #         if isinstance(el, Glass) or isinstance(el, CurvedGlass) or isinstance(el, FreeFormInterface):
    #             #modify index of refraction in propagation according to Glass element
    #             abcd.extend([Prop(d, n=el.n1), ABCD(el.m), ABCD(el.Rbasis)])
    #         else:
    #             abcd.extend([Prop(d), ABCD(el.m), ABCD(el.Rbasis)])
    #     return ABCDSystem(abcd)

    @property
    def abcd(self):
        """Calculate the ABCD matrix respecting element order and direction."""
        abcd_matrices = []
        
        # Start from element after source
        for i in range(1, len(self.elements)):
            prev_el = self.elements[i-1]
            curr_el = self.elements[i]
            # Calculate direction vector from previous to current element
            direction = norm(curr_el.p - prev_el.p)
            
            # Calculate propagation distance
            d = np.linalg.norm(curr_el.p - prev_el.p)
            
            aligned_with_normal = np.dot(direction, prev_el.n) > 0
            # print('directions ',curr_el, curr_el.p, prev_el.p, direction, aligned_with_normal)

            # Determine propagation index based on previous element
            n_prop = 1.0  # Default to air
            if isinstance(prev_el, Glass):
                # Determine which index to use based on ray direction
                n_prop = prev_el.n1 if aligned_with_normal else prev_el.n2

            # print('n_prop from ', prev_el, 'to ', curr_el, 'is ', n_prop)
            # Add propagation matrix
            abcd_matrices.append(Prop(d, n=n_prop))
            
            # Add element's ABCD matrix
            _m = curr_el.get_abcd(direction)
            abcd_matrices.append(ABCD(_m))
            # print('m ', _m)
            
            # Add any basis rotation
            abcd_matrices.append(ABCD(curr_el.Rbasis))
        
        return ABCDSystem(abcd_matrices)
    
    def propagate(self, rays, Nrt=1, at_screen=False, clip=True):
        """Propagate rays through an optical system (series of elements).

        Args:
            rays (ndarray): Input rays to propagate of shape (2, Nrays, 3).
            Nrt (int): Number of roundtrips to propagate.
            clip (bool): If true, rays are clipped on apertures of each element.
            at_screen (bool): If true, only stores intersection at screen for speed.

        Returns:
            trajs (ndarray): 
                If at_screen:
                    Array of rays of shape (Nrt, 2, Nrays, 3).
                Else
                    Array of rays of shape (Nelements*Nrt+1, 2, Nrays, 3).
                    The input ray is always stored as the first element.
        """
        rays = rays.astype(np.float64)
        if rays.shape[1]>1:
            #many rays
            if at_screen:
                return self._propagate_system_screen_many(self.jelements, rays, Nrt=Nrt, clip=clip)
            else:
                return self._propagate_system_many(self.jelements, rays, Nrt=Nrt, clip=clip)
        else:
            #one ray
            ray = np.squeeze(rays)
            if at_screen:
                trajs = self._propagate_system_screen(self.jelements, ray.copy(), Nrt=Nrt, clip=clip)
                return trajs.reshape((-1, 2, 1, 3))
            else:
                trajs = self._propagate_system(self.jelements, ray.copy(), Nrt=Nrt, clip=clip)
                return trajs.reshape((-1, 2, 1, 3))
    
    @staticmethod
    @jit(nopython=True)  
    def _propagate_system_screen(elements, ray, Nrt=1, clip=True):
        rcur = ray
        trajs = np.empty((Nrt+1, 2, 3), dtype=np.float64)
        trajs[0,:,:] = rcur
        for i in range(Nrt):
            for el in elements:
                rcur = el.propagate(rcur, clip)
            trajs[i+1,:,:] = rcur
        return trajs

    @staticmethod
    @jit(nopython=True)
    def _propagate_system(elements, ray, Nrt=1, clip=True):
        rcur = ray
        Nel = len(elements)
        trajs = np.empty((Nel*Nrt+1, 2, 3), dtype=np.float64)
        trajs[0,:,:] = rcur
        for i in range(Nrt):
            for j in range(Nel):
                rcur = elements[j].propagate(rcur, clip=clip)
                trajs[i*Nel+j+1,:,:] = rcur
        return trajs
    
    @staticmethod
    @jit(nopython=True)  
    def _propagate_system_screen_many(elements, rays, Nrt=1, clip=True):
        rcurs = rays
        Nrays = rcurs.shape[1]
        trajs = np.empty((Nrt+1, 2, Nrays, 3), dtype=np.float64)
        trajs[0,:,:,:] = rcurs
        for k in range(Nrays):#TODO parallelize this!
            rcur = rcurs[:,k,:]
            for i in range(Nrt):
                for el in elements:
                    rcur = el.propagate(rcur, clip)
                trajs[i+1,:,k,:] = rcur
        return trajs
    
    @staticmethod
    @jit(nopython=True)  
    def _propagate_system_many(elements, rays, Nrt=1, clip=True):
        rcurs = rays
        Nrays = rcurs.shape[1]
        Nel = len(elements)
        trajs = np.empty((Nel*Nrt+1, 2, Nrays, 3), dtype=np.float64)
        trajs[0,:,:,:] = rcurs
        for k in range(Nrays):
            rcur = rcurs[:,k,:]
            for i in range(Nrt):
                for j in range(Nel):
                    rcur = elements[j].propagate(rcur, clip)
                    trajs[i*Nel+j+1,:,k,:] = rcur
        return trajs
    
    def extract_ABCD(self, epsr = 1e-1, epss = 1e-1, Nrt=1):
        """Extract an ABCD matrix from raytracing.
        Five guiding rays (of the form (eps, 0, 0, 0)...) are generated 
        and the matrix entries are determined by finite difference.
        Args:
            epsr (float): Epsilon for position offset.
            epss (float): Epsilon for position slope (might be more sensitive).
            Nrt (int): Number of roundtrips ttto extract matrix from.
        """
        mue = np.array([[0, 0, 0, 0], [epsr, 0, 0, 0], [0, epsr, 0, 0], [0, 0, epss, 0], [0, 0, 0, epss]], dtype=np.float64)
        guideray = self.screen.eigenvectors_to_rays(mue)
        trajgr = self.propagate(guideray, Nrt=Nrt, clip=True)
        Mfd = self.screen.rays_to_eigenvectors(trajgr[-1,:,1:,:]) - self.screen.rays_to_eigenvectors(trajgr[-1,:,0,:][:,None,:])  
        Mfd = Mfd.T
        Mfd[:2, :] /= epsr
        Mfd[2:, :] /= epss
        return Mfd
    
    def find_eigenray(self, ray0, lr = 0.05, maxiter=500, tol=1e-5, Nrt=1, debug=False, animated=False):
        """Find the eigenray (fixed point) of an optical system iteratively.

        Args:
            ray0 (ndarray): Input rays to stabilize of shape (2, Nrays, 3).
            Nrt (int): Number of roundtrips to propagate between iteration steps.
            maxiter (int): Maximum number of iterations.
            tol (float): Tolerance (relative change) down to which to iterate.
                Terminates if tol or maxiter reached.
            lr (float): 'learing rate', must be hand-tuned.
            debug (bool): If true, prints debug messages.
            animated (bool): save all intermediate trajectories for animation.

        Returns:
            rcur (ndarray): Eigenray of the system (2, Nrays, 3).
        """
        rcur = ray0.copy()
        trajs = []
        for i in range(maxiter):
            traj = self.propagate(rcur, Nrt=Nrt)
            rnew = traj[-1,...]
            if np.isnan(rnew).any():
                raise ConvergenceError
            if animated:
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
            
        if animated:
            return rcur, np.stack(trajs, axis=0)
        else:
            return rcur
        
    def _find_eigenray_formpe(self, ray0, lr = 0.05, maxiter=500, tol=1e-5, Nrt=1, debug=False, clip=True):
        rcur = ray0.copy()
        trajs = []
        tols = []
        for i in range(maxiter):
            traj = self.propagate(rcur, Nrt=Nrt)
            rnew = traj[-1,...]
            if np.isnan(rnew).any():
                raise ConvergenceError
            trajs.append(rnew)
            res = np.max(np.abs(rcur.flatten() - rnew.flatten()))
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
        
    @staticmethod
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
    
    def find_eigenray_mpe(self, ray0, lr=0.03, Niter=50, Nmpe=5, tol=1e-9, get_tols=False, **kwargs):
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
        alltols = []
        for i in range(Nmpe):
            rconv, rseq, tols = self._find_eigenray_formpe(rnew, lr=lr, maxiter=Niter, tol=tol*1e-1, **kwargs)
            if get_tols:
                alltols.append(tols)
            #if inner loop terminates early, break and don't do MPE step
            if rseq.shape[0]<4:
                rnew = rconv
                break
                
            #if inner loop reaches desired tolerance, break and don't do MPE step
            if tols[-1]<tol:
                rnew = rconv
                break
                
            rseq_rs = np.squeeze(rseq).reshape(-1,6).T #reshape sequence into format for MPE
            rnew = RaySystem.MPE(rseq_rs) #find new starting vector
            rnew = rnew.reshape(2,-1,3)
        if get_tols:
            return rnew, np.concatenate(alltols)
        else:
            return rnew
        
    def find_eigenray_mpe_dev(self, ray0, lr=0.03, Niter=50, Nmpe=5, tol=1e-9, get_tols=False, lr_decay=0.9, **kwargs):
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
        alltols = []
        lr_i = lr
        tolmin = np.inf
        raymin = ray0.copy()
        for i in range(Nmpe):
            rconv, rseq, tols = self._find_eigenray_formpe(rnew, lr=lr_i, maxiter=Niter, tol=tol*1e-1, **kwargs)
            if get_tols:
                alltols.append(tols)
                
            #find "best" eigenray (lowest tolerance)
            
            
            #if inner loop terminates early, break and don't do MPE step
            if rseq.shape[0]<10:
                rnew = rconv
                break
                
            #if inner loop reaches desired tolerance, break and don't do MPE step
            if tols[-1]<tol:
                rnew = rconv
                break
                
            rseq_rs = np.squeeze(rseq).reshape(-1,6).T #reshape sequence into format for MPE
            rnew = RaySystem.MPE(rseq_rs) #find new starting vector
            rnew = rnew.reshape(2,-1,3)
            #decay learning rate/relaxation constant with each iteration
            lr_i = lr_i*lr_decay
            
        if get_tols:
            return rnew, np.concatenate(alltols)
        else:
            return rnew