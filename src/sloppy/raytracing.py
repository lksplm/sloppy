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
    def __init__(self, elements, add_screen=True, old_mode=False):
        self.old_mode = old_mode
        if add_screen:
            #check if last element is a screen already
            if isinstance(elements[-1], Screen):
                screen = elements[-1]
            else:
                #add screen to elements
                x0 = 0.5*(elements[0].p + elements[-1].p)
                # n0 = norm(elements[0].p - elements[-1].p)
                n0 = elements[0].n

                screen = Screen(p=x0, n=n0, diameter=10., ax=elements[0].ax, ay=elements[0].ay)
                elements.append(screen)
            self.screen = screen
        self.elements = elements
        self.jelements = tuple((el.jopt for el in elements)) #homogenous tuple to support jitted routines
        
    # @property
    def _old_abcd(self):
        pos = [e.p for e in self.elements]

        abcd = []
        for i, el in enumerate(self.elements):
            d = np.linalg.norm(pos[i-1]-pos[i])
            if isinstance(el, Glass) or isinstance(el, CurvedGlass) or isinstance(el, FreeFormInterface):
                #modify index of refraction in propagation according to Glass element
                abcd.extend([Prop(d, n=el.n1), ABCD(el.m), ABCD(el.Rbasis)])
            else:
                abcd.extend([Prop(d), ABCD(el.m), ABCD(el.Rbasis)])
        return ABCDSystem(abcd)

    @property
    def abcd(self):
        """Calculate the ABCD matrix respecting element order and direction."""
        if self.old_mode:
            return self._old_abcd()
        abcd_matrices = []
        
        # Start from element after source
        # for i in range(1, len(self.elements)):
        for i in range(len(self.elements)):
            prev_el = self.elements[i-1]
            curr_el = self.elements[i]
            # Calculate direction vector from previous to current element
            direction = norm(curr_el.p - prev_el.p)
            
            # Calculate propagation distance
            d = np.linalg.norm(curr_el.p - prev_el.p)
            
            
            aligned_with_normal = np.dot(direction, curr_el.n) > 0
            # print('directions ',curr_el, curr_el.p, prev_el.p, direction, aligned_with_normal)
            # print(f"element {curr_el}, pos {curr_el.p}, dist {d}, prev pos {prev_el.p}, normal {curr_el.n}, prev normal {prev_el.n}, direction {direction}, aligned {aligned_with_normal}")

            # Determine propagation index based on previous element
            n_prop = 1.0  # Default to air
            prev_aligned_with_normal = np.dot(direction, prev_el.n) > 0
            if isinstance(prev_el, Glass):
                # Determine which index to use based on ray direction
                # n_prop = prev_el.n1 if aligned_with_normal else prev_el.n2
                n_prop = prev_el.n1 if prev_aligned_with_normal else prev_el.n2
                # print(f"n_prop from {prev_el} to {curr_el} is {n_prop}")

            # print('n_prop from ', prev_el, 'to ', curr_el, 'is ', n_prop)
            # Add propagation matrix
            _m = Prop(d, n=n_prop)
            # print(f"propagation matrix {_m.m}")
            abcd_matrices.append(_m)
            
            # Add element's ABCD matrix
            _m2 = curr_el.get_abcd(direction)
            abcd_matrices.append(ABCD(_m2))
            # print('m ', _m)
            
            # Add any basis rotation
            abcd_matrices.append(ABCD(curr_el.Rbasis))
        
        return ABCDSystem(abcd_matrices)
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def convert_layout(trajs, from_layout, to_layout):
        """Convert trajectory arrays between different memory layouts.
        
        Args:
            trajs: Input trajectory array
            from_layout: Source layout ('standard', 'optimized')  
            to_layout: Target layout ('standard', 'optimized')
            
        Returns:
            Converted trajectory array
        """
        if from_layout == to_layout:
            return trajs
            
        if from_layout == 'optimized' and to_layout == 'standard':
            # (Nrays, Ntime, 2, 3) -> (Ntime, 2, Nrays, 3)
            Nrays, Ntime, _, _ = trajs.shape
            standard = np.empty((Ntime, 2, Nrays, 3), dtype=np.float64)
            for k in range(Nrays):
                for t in range(Ntime):
                    standard[t, :, k, :] = trajs[k, t, :, :]
            return standard
        elif from_layout == 'standard' and to_layout == 'optimized':
            # (Ntime, 2, Nrays, 3) -> (Nrays, Ntime, 2, 3)
            Ntime, _, Nrays, _ = trajs.shape
            optimized = np.empty((Nrays, Ntime, 2, 3), dtype=np.float64)
            for k in range(Nrays):
                for t in range(Ntime):
                    optimized[k, t, :, :] = trajs[t, :, k, :]
            return optimized
        else:
            raise ValueError(f"Unsupported conversion from {from_layout} to {to_layout}")
    
    def propagate(self, rays, Nrt=1, at_screen=False, clip=True, layout='standard'):
        """Propagate rays through an optical system (series of elements).

        Args:
            rays (ndarray): Input rays to propagate of shape (2, Nrays, 3).
            Nrt (int): Number of roundtrips to propagate.
            clip (bool): If true, rays are clipped on apertures of each element.
            at_screen (bool): If true, only stores intersection at screen for speed.
            layout (str): Memory layout optimization:
                'standard': (Ntime, 2, Nrays, 3) - backward compatible, good for single-threaded
                'optimized': (Nrays, Ntime, 2, 3) - better parallel performance with rays-first layout

        Returns:
            trajs (ndarray): Ray trajectories with shape depending on layout parameter.
                The input ray is always stored as the first element.
        """
        rays = rays.astype(np.float64)
        if rays.shape[1]>1:
            #many rays
            if layout == 'optimized':
                # Use optimized memory layout versions
                if at_screen:
                    return self._propagate_system_screen_many_optimized(self.jelements, rays, Nrt=Nrt, clip=clip)
                else:
                    return self._propagate_system_many_optimized(self.jelements, rays, Nrt=Nrt, clip=clip)
            else: # layout == 'standard'
                # Use standard layout versions
                if at_screen:
                    return self._propagate_system_screen_many(self.jelements, rays, Nrt=Nrt, clip=clip)
                else:
                    return self._propagate_system_many(self.jelements, rays, Nrt=Nrt, clip=clip)
        else:
            #one ray
            ray = np.squeeze(rays)
            if at_screen:
                trajs = self._propagate_system_screen(self.jelements, ray.copy(), Nrt=Nrt, clip=clip)
                if layout == 'optimized':
                    return trajs.reshape((1, -1, 2, 3))
                else:
                    return trajs.reshape((-1, 2, 1, 3))
            else:
                trajs = self._propagate_system(self.jelements, ray.copy(), Nrt=Nrt, clip=clip)
                if layout == 'optimized':
                    return trajs.reshape((1, -1, 2, 3))
                else:
                    return trajs.reshape((-1, 2, 1, 3))
    
    @staticmethod
    @jit(nopython=True, cache=True)  
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
    @jit(nopython=True, cache=True)
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
    
    # @staticmethod
    # @jit(nopython=True)  
    # def _propagate_system_screen_many(elements, rays, Nrt=1, clip=True):
    #     rcurs = rays
    #     Nrays = rcurs.shape[1]
    #     trajs = np.empty((Nrt+1, 2, Nrays, 3), dtype=np.float64)
    #     trajs[0,:,:,:] = rcurs
    #     for k in range(Nrays):#TODO parallelize this!
    #         rcur = rcurs[:,k,:]
    #         for i in range(Nrt):
    #             for el in elements:
    #                 rcur = el.propagate(rcur, clip)
    #             trajs[i+1,:,k,:] = rcur
    #     return trajs
    
    @staticmethod
    @jit(nopython=True, cache=True)  
    def _propagate_system_screen_many(elements, rays, Nrt=1, clip=True):
        rcurs = rays
        Nrays = rcurs.shape[1]
        trajs = np.empty((Nrt+1, 2, Nrays, 3), dtype=np.float64)
        trajs[0,:,:,:] = rcurs
        # parallelize this using prange
        for k in prange(Nrays):
            rcur = rcurs[:,k,:]
            for i in range(Nrt):
                for el in elements:
                    rcur = el.propagate(rcur, clip)
                trajs[i+1,:,k,:] = rcur
        return trajs
    
    @staticmethod
    @jit(nopython=True, cache=True)  
    def _propagate_system_many(elements, rays, Nrt=1, clip=True):
        rcurs = rays
        Nrays = rcurs.shape[1]
        Nel = len(elements)
        trajs = np.empty((Nel*Nrt+1, 2, Nrays, 3), dtype=np.float64)
        trajs[0,:,:,:] = rcurs
        for k in prange(Nrays):
            rcur = rcurs[:,k,:]
            for i in range(Nrt):
                for j in range(Nel):
                    rcur = elements[j].propagate(rcur, clip)
                    trajs[i*Nel+j+1,:,k,:] = rcur
        return trajs
    
    @staticmethod
    @jit(nopython=True, cache=True, parallel=True)  
    def _propagate_system_screen_many_optimized(elements, rays, Nrt=1, clip=True):
        rcurs = rays
        Nrays = rcurs.shape[1]
        # Optimized layout: (Nrays, Nrt+1, 2, 3) - rays first for better cache locality
        trajs = np.empty((Nrays, Nrt+1, 2, 3), dtype=np.float64)
        # Copy initial rays
        for k in range(Nrays):
            trajs[k, 0, :, :] = rcurs[:, k, :]
        
        # parallelize over rays with better memory layout
        for k in prange(Nrays):
            rcur = rcurs[:,k,:]
            for i in range(Nrt):
                for el in elements:
                    rcur = el.propagate(rcur, clip)
                trajs[k, i+1, :, :] = rcur
        return trajs
    
    @staticmethod
    @jit(nopython=True, cache=True, parallel=True)  
    def _propagate_system_many_optimized(elements, rays, Nrt=1, clip=True):
        rcurs = rays
        Nrays = rcurs.shape[1]
        Nel = len(elements)
        # Optimized layout: (Nrays, Nel*Nrt+1, 2, 3) - rays first for better cache locality
        trajs = np.empty((Nrays, Nel*Nrt+1, 2, 3), dtype=np.float64)
        # Copy initial rays
        for k in range(Nrays):
            trajs[k, 0, :, :] = rcurs[:, k, :]
            
        for k in prange(Nrays):
            rcur = rcurs[:,k,:]
            for i in range(Nrt):
                for j in range(Nel):
                    rcur = elements[j].propagate(rcur, clip)
                    trajs[k, i*Nel+j+1, :, :] = rcur
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
    
    def find_eigenray_mpe(self, ray0, lr=0.03, Niter=50, Nmpe=5, tol=1e-9, get_tols=False, lr_decay=0.9, **kwargs):
        """Find eigenray using Minimal Polynomial Extrapolation.
        
        This method uses the MPE function to accelerate convergence.
        
        Args:
            ray0 (ndarray): Input rays to stabilize of shape (2, Nrays, 3).
            Niter (int): Number of iterations.
            Nmpe (int): Number of MPE steps.
            tol (float): Tolerance (relative change) down to which to iterate.
            lr (float): 'learning rate', must be hand-tuned.
            lr_decay (float): Decay factor for learning rate.
            get_tols (bool): If true, return tolerance history.
            
        Returns:
            rcur (ndarray): Eigenray of the system (2, Nrays, 3).
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
        """Development version of find_eigenray_mpe.
        
        This method uses the MPE function to accelerate convergence.
        
        Args:
            ray0 (ndarray): Input rays to stabilize of shape (2, Nrays, 3).
            Niter (int): Number of iterations.
            Nmpe (int): Number of MPE steps.
            tol (float): Tolerance (relative change) down to which to iterate.
            lr (float): 'learning rate', must be hand-tuned.
            lr_decay (float): Decay factor for learning rate.
            get_tols (bool): If true, return tolerance history.
            
        Returns:
            rcur (ndarray): Eigenray of the system (2, Nrays, 3).
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

    def benchmark_propagate(self, rays, Nrt=1, at_screen=False, clip=True, num_runs=3):
        """Benchmark different propagation methods for performance comparison.
        
        Args:
            rays (ndarray): Input rays to propagate of shape (2, Nrays, 3).
            Nrt (int): Number of roundtrips to propagate.
            at_screen (bool): If true, only stores intersection at screen for speed.
            clip (bool): If true, rays are clipped on apertures of each element.
            num_runs (int): Number of benchmark runs to average.
            
        Returns:
            dict: Timing results for different methods.
        """
        import time
        
        print(f"Benchmarking propagation with {rays.shape[1]} rays, {Nrt} roundtrips")
        print("Warming up JIT compilation...")
        
        # Warm up all methods
        _ = self.propagate(rays, Nrt=1, at_screen=at_screen, clip=clip, layout='standard')
        if rays.shape[1] > 1:
            _ = self.propagate(rays, Nrt=1, at_screen=at_screen, clip=clip, layout='optimized')
        
        results = {}
        
        # Benchmark standard layout
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.propagate(rays, Nrt=Nrt, at_screen=at_screen, clip=clip, layout='standard')
            times.append(time.perf_counter() - start)
        results['standard'] = np.mean(times)
        
        # Benchmark optimized layout (only for multiple rays)
        if rays.shape[1] > 1:
            # Optimized layout
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = self.propagate(rays, Nrt=Nrt, at_screen=at_screen, clip=clip, layout='optimized')
                times.append(time.perf_counter() - start)
            results['optimized'] = np.mean(times)
        
        # Print results
        print("\nBenchmark Results:")
        print("-" * 40)
        for layout, time_taken in results.items():
            speedup = results['standard'] / time_taken if layout != 'standard' else 1.0
            print(f"{layout:>12}: {time_taken:.4f}s (speedup: {speedup:.2f}x)")
        
        return results

    def analyze_numba_compilation(self, rays, Nrt=1, at_screen=False, clip=True, save_to_file=False):
        """Analyze Numba compilation details for the propagation functions.
        
        This function provides detailed information about:
        - Type inference
        - LLVM intermediate representation
        - Generated assembly code
        - Control flow graphs
        
        Args:
            rays (ndarray): Sample rays to trigger compilation
            Nrt (int): Number of roundtrips
            at_screen (bool): Whether to analyze screen-only propagation
            clip (bool): Whether to analyze with clipping
            save_to_file (bool): Whether to save analysis to files
            
        Returns:
            dict: Analysis results for different functions
        """
        print("=== Numba Compilation Analysis ===")
        print(f"Analyzing with {rays.shape[1]} rays, Nrt={Nrt}")
        
        # Force compilation by calling the functions
        print("Triggering compilation...")
        _ = self.propagate(rays, Nrt=1, at_screen=at_screen, clip=clip, layout='standard')
        if rays.shape[1] > 1:
            _ = self.propagate(rays, Nrt=1, at_screen=at_screen, clip=clip, layout='optimized')
        
        results = {}
        
        # Analyze the different propagation functions
        functions_to_analyze = []
        
        if rays.shape[1] > 1:
            if at_screen:
                functions_to_analyze.extend([
                    ('standard_screen_many', self._propagate_system_screen_many),
                    ('optimized_screen_many', self._propagate_system_screen_many_optimized)
                ])
            else:
                functions_to_analyze.extend([
                    ('standard_many', self._propagate_system_many),
                    ('optimized_many', self._propagate_system_many_optimized)
                ])
        else:
            if at_screen:
                functions_to_analyze.append(('single_screen', self._propagate_system_screen))
            else:
                functions_to_analyze.append(('single', self._propagate_system))
        
        for func_name, func in functions_to_analyze:
            print(f"\n{'='*60}")
            print(f"Analyzing: {func_name}")
            print(f"{'='*60}")
            
            try:
                # Get the compiled function signatures
                signatures = list(func.signatures)
                print(f"Compiled signatures: {len(signatures)}")
                for i, sig in enumerate(signatures):
                    print(f"  {i+1}: {sig}")
                
                if signatures:
                    # Get the first signature for detailed analysis
                    signature = signatures[0]
                    
                    result = {
                        'signatures': signatures,
                        'function_name': func_name
                    }
                    
                    # Type analysis - call on the dispatcher function with signature
                    print(f"\n--- Type Information for {func_name} ---")
                    try:
                        types_info = func.inspect_types(signature)
                        result['types'] = types_info
                        print("Type inference successful - see returned dict for details")
                        if save_to_file:
                            with open(f"numba_types_{func_name}.txt", "w") as f:
                                f.write(types_info)
                            print(f"Types saved to numba_types_{func_name}.txt")
                    except Exception as e:
                        print(f"Type inspection failed: {e}")
                        result['types'] = None
                    
                    # LLVM IR analysis
                    print(f"\n--- LLVM IR for {func_name} ---")
                    try:
                        llvm_ir = func.inspect_llvm(signature)
                        result['llvm_ir'] = llvm_ir
                        # Print first few lines as preview
                        lines = llvm_ir.split('\n')[:10]
                        print("LLVM IR preview (first 10 lines):")
                        for line in lines:
                            print(f"  {line}")
                        print(f"  ... ({len(llvm_ir.split(chr(10)))} total lines)")
                        if save_to_file:
                            with open(f"numba_llvm_{func_name}.ll", "w") as f:
                                f.write(llvm_ir)
                            print(f"LLVM IR saved to numba_llvm_{func_name}.ll")
                    except Exception as e:
                        print(f"LLVM IR inspection failed: {e}")
                        result['llvm_ir'] = None
                    
                    # Assembly analysis
                    print(f"\n--- Assembly Code for {func_name} ---")
                    try:
                        asm_code = func.inspect_asm(signature)
                        result['assembly'] = asm_code
                        # Print first few lines as preview
                        lines = asm_code.split('\n')[:15]
                        print("Assembly code preview (first 15 lines):")
                        for line in lines:
                            print(f"  {line}")
                        print(f"  ... ({len(asm_code.split(chr(10)))} total lines)")
                        if save_to_file:
                            with open(f"numba_asm_{func_name}.s", "w") as f:
                                f.write(asm_code)
                            print(f"Assembly saved to numba_asm_{func_name}.s")
                    except Exception as e:
                        print(f"Assembly inspection failed: {e}")
                        result['assembly'] = None
                    
                    # Control flow graph (if available)
                    print(f"\n--- Control Flow Analysis for {func_name} ---")
                    try:
                        # Try different CFG methods that might be available
                        cfg_info = None
                        try:
                            cfg_info = func.inspect_cfg(signature)
                        except:
                            # Alternative method for CFG inspection
                            try:
                                # Get the compiled result and try to access CFG from there
                                compiled_result = func.overloads[signature]
                                if hasattr(compiled_result, 'library'):
                                    cfg_info = f"Function compiled successfully. Library: {compiled_result.library}"
                                else:
                                    cfg_info = f"Function compiled successfully. Signature: {signature}"
                            except:
                                cfg_info = "CFG inspection not available for this Numba version"
                        
                        result['cfg'] = cfg_info
                        print(f"Control flow info: {cfg_info}")
                        if save_to_file and cfg_info:
                            with open(f"numba_cfg_{func_name}.txt", "w") as f:
                                f.write(str(cfg_info))
                            print(f"CFG info saved to numba_cfg_{func_name}.txt")
                    except Exception as e:
                        print(f"CFG inspection failed: {e}")
                        result['cfg'] = None
                    
                    results[func_name] = result
                
            except Exception as e:
                print(f"Error analyzing {func_name}: {e}")
                results[func_name] = {'error': str(e)}
        
        print(f"\n{'='*60}")
        print("Analysis Summary:")
        print(f"{'='*60}")
        for func_name, result in results.items():
            if 'error' in result:
                print(f"{func_name}: ERROR - {result['error']}")
            else:
                print(f"{func_name}: ✓ Analyzed successfully")
                print(f"  - Signatures: {len(result.get('signatures', []))}")
                print(f"  - Types: {'✓' if result.get('types') else '✗'}")
                print(f"  - LLVM IR: {'✓' if result.get('llvm_ir') else '✗'}")
                print(f"  - Assembly: {'✓' if result.get('assembly') else '✗'}")
                print(f"  - CFG: {'✓' if result.get('cfg') else '✗'}")
        
        return results

    def plot_compilation_metrics(self, analysis_results, save_plots=False):
        """Create visual plots comparing compilation characteristics.
        
        Args:
            analysis_results (dict): Results from analyze_numba_compilation
            save_plots (bool): Whether to save plots to files
            
        Returns:
            dict: Generated plots and metrics
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from collections import defaultdict
        
        print("Creating compilation analysis plots...")
        
        # Extract metrics from analysis results
        metrics = defaultdict(list)
        function_names = []
        
        for func_name, result in analysis_results.items():
            if 'error' in result:
                continue
                
            function_names.append(func_name)
            
            # Count signatures
            metrics['num_signatures'].append(len(result.get('signatures', [])))
            
            # Count LLVM IR lines
            llvm_lines = 0
            if result.get('llvm_ir'):
                llvm_lines = len(result['llvm_ir'].split('\n'))
            metrics['llvm_lines'].append(llvm_lines)
            
            # Count assembly lines
            asm_lines = 0
            if result.get('assembly'):
                asm_lines = len(result['assembly'].split('\n'))
            metrics['asm_lines'].append(asm_lines)
            
            # Estimate complexity (rough metric)
            complexity = llvm_lines + asm_lines * 0.1
            metrics['complexity'].append(complexity)
        
        if not function_names:
            print("No valid analysis results to plot")
            return {}
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Numba Compilation Analysis Comparison', fontsize=16)
        
        # Plot 1: Number of signatures
        axes[0,0].bar(function_names, metrics['num_signatures'], color='skyblue')
        axes[0,0].set_title('Number of Compiled Signatures')
        axes[0,0].set_ylabel('Count')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Plot 2: LLVM IR lines
        axes[0,1].bar(function_names, metrics['llvm_lines'], color='lightgreen')
        axes[0,1].set_title('LLVM IR Lines')
        axes[0,1].set_ylabel('Lines of Code')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Assembly lines
        axes[1,0].bar(function_names, metrics['asm_lines'], color='lightcoral')
        axes[1,0].set_title('Assembly Lines')
        axes[1,0].set_ylabel('Lines of Code')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Complexity comparison
        axes[1,1].bar(function_names, metrics['complexity'], color='orange')
        axes[1,1].set_title('Estimated Complexity')
        axes[1,1].set_ylabel('Complexity Score')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('numba_compilation_analysis.png', dpi=300, bbox_inches='tight')
            print("Plots saved to numba_compilation_analysis.png")
        
        plt.show()
        
        # Print detailed metrics
        print("\n" + "="*60)
        print("Detailed Compilation Metrics:")
        print("="*60)
        for i, func_name in enumerate(function_names):
            print(f"\n{func_name}:")
            print(f"  Signatures: {metrics['num_signatures'][i]}")
            print(f"  LLVM IR lines: {metrics['llvm_lines'][i]}")
            print(f"  Assembly lines: {metrics['asm_lines'][i]}")
            print(f"  Complexity score: {metrics['complexity'][i]:.1f}")
        
        return {
            'metrics': dict(metrics),
            'function_names': function_names,
            'figure': fig
        }