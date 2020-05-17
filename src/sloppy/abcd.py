import numpy as np
from math import floor
import matplotlib.pyplot as plt
from itertools import product


class ABCD:
    """Base class for ABCD objects.
    In addition to the matrix it can store element properties.

    """
    def __init__(self, m):
        if m is None:
            self.m = np.eye(4)
        else:
            self.m = np.array(m)
        
    def __matmul__(self, other):
        return self.m@other.m


class Prop(ABCD):
    """Propagation through space.
    Args:
        L (float): Propagation distance.
        n (float): Index of refraction of the medium.
    """
    def __init__(self, L, n=1.):
        # Returns the ABCD array for propagation through dielectric constant n
        self.L = L
        self.n = n
        m = np.identity(4)
        m[0,2] = L
        m[1,3] = L
        self.m = m

class Rot(ABCD):
    """Coordinate transformation through rotation.
    Args:
        r (ndarray): 2x2 rotation matrix.
    """
    def __init__(self, r):
        # Returns the ABCD array for basis rotation with 2x2 matrix r
        self.r = r
        m = np.identity(4)
        m[0:2,0:2] = r
        m[2:4,2:4] = r
        self.m = m
        
def ThickLens(d, n1=1., n2=1., R1=None, R2=None):
    """Composite ABCD matrix for thick lens.
    Args:
        R1 (float): Radius of curvature of the first surface (None is flat).
        R2 (float): Radius of curvature of the second surface (None is flat).
        n1 (float): Index of refraction outside the lens.
        n2 (float): Index of refraction inside the lens.
    """
    return [Interface(n1, n2, R1), Prop(d, n2), Interface(n2, n1, R2)]

class ABCDSystem:
    """System comprised of 4x4 ABCD matrices of optical elements and propagation in 2D.
    
    Can obtain different mode properties (waists, eigenvectors) at positions on the optical axis.
    Instead of the complex q parameter for one-dimensional system (2x2 matrices) the eigenvector matrix BiK is used throughout the class.
    
    Attributes:
        elements (list): List of :class:`Optic` elements and :class:`Prop` propagation.
        wl (float): Wavelength of the light.

    """
    @staticmethod
    def abcd_from_x(x, distlist, abcdlist, nlist):
        """ABCD matrix as a function of position x along the optical axis."""
        idx = np.where(x>=np.array(distlist))[0][-1]
        return Prop(x-distlist[idx], nlist[idx]).m@abcdlist[idx]
    
    @staticmethod
    def get_conj_pairs(ev):
        """Get both conjugate eigen pairs with the right normalization."""
        ind = [0, 1, 2, 3]
        indout = []
        evout = []
        while len(ind)>0:
            i = ind.pop(0)
            j = ind[np.argsort(np.abs(ev[i].conj()-ev[ind]))[0]]
            if ev[i].imag<ev[j].imag:
                i, j = j, i
            indout.append(i)
            evout.append(ev[i])
            ind.remove(j)
        return indout, evout
    
    @staticmethod
    def wrapAroundFSR(fsr,f):
        return (f%(fsr/2) - (fsr/2)*(1 if floor(f/(fsr/2))%2==1 else 0))        
    
    @staticmethod
    def stability(Mrt):
        """Stability parameter."""
        # seems like maybe the second dimension gives a factor of 2? Maybe we should just use the norm of the eigenvalues
        return np.trace(Mrt)*0.5/2
    
    @staticmethod
    def stability_bool(Mrt):
        """Is the cavity stable?."""
        eps=0.0001
        return all([np.abs(np.linalg.norm(eva)-1)<eps for eva in np.linalg.eigvals(Mrt)])
    
    def __init__(self, elements, wl = 780e-6):
        #unwarp possible nested elements
        self.elements = []
        for ele in elements:
            if isinstance(ele, list):
                self.elements.extend(ele)
            else:
                self.elements.append(ele)
        self.wl = wl
        #build functions
        distlist=[] #cumulative distances
        optDistlist=[] #cumulative optical distances
        abcdlist=[] #cumulative abcd matrices
        nlist = []
        dtot=0.0
        dOptTot=0.0
        abcd=np.eye(4)
        for ele in self.elements:
            # check if ele is a propagation or other ABCD matrix
            if isinstance(ele, Prop):
                abcdlist.append(abcd.copy())
                distlist.append(dtot)
                nlist.append(ele.n)
                dtot=dtot+ele.L
                dOptTot=dOptTot+ele.L*ele.n
                abcd=ele.m@abcd
            else:
                abcd=ele.m@abcd
        #print(nlist)
        abcdlist.append(abcd.copy())
        distlist.append(dtot)
        optDistlist.append(dOptTot)
        nlist.append(nlist[-1]) #TODO: Check this!
        
        self.abcd_rt = abcd
        self.distlist = distlist
        self.optDistlist = optDistlist
        self.abcdlist = abcdlist
        self.nlist = nlist
        self.Ltot = dtot
        self.LOptot = dOptTot
        self.fsr = 2.99792458e8/(dOptTot*1e-3)
        
        self.abcd_fct = lambda x: self.abcd_from_x(x, distlist, abcdlist, nlist)
        
        #compute mode
        #self.q, self.m = self.solve_mode(self.abcd_rt)
        self.q = self.M2BiK(self.abcd_rt)
        self.m = self.stability(self.abcd_rt)
        self.is_stable = self.stability_bool(self.abcd_rt)
        #assert self.m<1 , "Mode not stable!"
        #print("No eigenmode found!")
        
    def M2BiK(self, Mrt):
        """Get eigenvectors from ABCD matrix and normalize."""
        G = np.array([[0,0,1,0], [0,0,0,1], [-1,0,0,0], [0,-1,0,0]])
        ev, evec = np.linalg.eig(Mrt)
        ind, es = ABCDSystem.get_conj_pairs(ev)
        #print(ev, ind, es)
        mus = evec[:,ind]
        xsi = np.angle(es)
        N = [mus[:,i].T.conj()@G@mus[:,i] for i in range(2)]
        musN = np.stack([mus[:,i]*np.sqrt(2/N[i]) for i in range(2)], axis=0)
        NN = [musN[i].T.conj()@G@musN[i] for i in range(2)]
        #print(NN)
        return musN
    

    def M2freq(self, Mrt, Lrt):
        """Get transverse mode frequencies from ABCD matrix and roundtrip lentgh."""
        if not self.is_stable:
            return [np.nan, np.nan], [np.nan, np.nan]
        ev = np.linalg.eigvals(Mrt)
        freqs = np.angle(ev)*self.fsr/(2*np.pi)
        freqsA = np.sort(freqs)[-1:-3:-1]
        freqsB = freqsA - self.fsr
        #freqsC = 3.*freqsA - self.fsr
        freqsC = np.array(list(self.wrapAroundFSR(self.fsr, 3.*f) for f in freqsA))
        return freqsA, freqsC
    
    def solve_mode(self, BiK, n=1.):
        """Get waists (diagonal) from eigenvectors."""
        #BiK = ABCDSystem.M2BiK(abcd)
        #B = BiK[:2,:]
        #K = -1j*BiK[2:,:]
        if not self.is_stable:
            return [0, 0]
        B = BiK[:,:2]
        K = -1j*BiK[:,2:]
        #ensure normalisation
        #print(K@(B.T.conj()) + B@(K.T.conj()))
        lam = self.wl
        Q = np.linalg.solve(B,K) # Q = B^-1 K
        di = np.linalg.eigvals(1j*Q)
        ws = np.sqrt(lam/(-n*np.pi)*((1/di.imag) if all(di.imag<0) else np.array([0, 0])))
        return ws
    

    def propBiK(self, BiK, M):
        """Propagate eigenvector matrix BiK with ABCD matrix M."""
        BiKz = BiK@M.T
        return BiKz
        
    def abcd_at(self, x):
        """Get cumulative ABCD matrix up to position x on the optical axis."""
        idx = np.where(x>=np.array(self.distlist))[0][-1]
        return Prop(x-self.distlist[idx], self.nlist[idx]).m@self.abcdlist[idx]
    
    def n_at(self, x):
        """Get index of refraction at position x to get right waists."""
        idx = np.where(x>=np.array(self.distlist))[0][-1]
        return self.nlist[idx]
    
    def waist_at(self, x):
        """Get waists at position x on the optical axis."""
        qp = self.propBiK(self.q, self.abcd_at(x))
        return self.solve_mode(qp, self.n_at(x))
    
    def compute_waists(self, x, qin=None):
        """Get waists at array of positions x on the optical axis."""
        if qin is None:
            qin = self.q
        ws = np.zeros((x.shape[0], 2))
        for i, l in enumerate(x):
            qp = self.propBiK(qin, self.abcd_at(l))
            ws[i,:] = self.solve_mode(qp, self.n_at(l))
        return ws
    
    def q_at(self, x):
        self.propBiK(self.q, self.abcd_at(x))
        
    def get_freqs(self):
        return self.M2freq(self.abcd_rt, self.Ltot)
    
def propagate_ABCD(mu, M, Nrt=100):
    rs = np.empty((Nrt+1,4))
    r = mu.copy()
    rs[0,:] = r
    for i in range(Nrt):
        r = M@r
        rs[i+1,:] = r
    return rs
