import numpy as np
from math import floor
import matplotlib.pyplot as plt
from itertools import product


class ABCD:
    """Base class for ABCD matrices in 2D.

    Parameters
    ----------
    m : ndarray
        4x4 ABCD matrix. If None, returns identity matrix.
    """
    def __init__(self, m):
        if m is None:
            self.m = np.identity(4, dtype=np.float64)
        else:
            self.m = np.array(m)
        
    def __matmul__(self, other):
        """Matrix multiplication of ABCD matrices.

        Parameters
        ----------
        other : ABCD
            Another ABCD matrix object

        Returns
        -------
        ndarray
            Result of matrix multiplication
        """
        return self.m@other.m


class Prop(ABCD):
    """Propagation through space.

    Parameters
    ----------
    L : float
        Propagation distance
    n : float, optional
        Index of refraction of the medium, by default 1.0
    """
    def __init__(self, L, n=1.):
        # Returns the ABCD array for propagation through dielectric constant n
        self.L = L
        self.n = n
        m = np.identity(4, dtype=np.float64)
        m[0,2] = L
        m[1,3] = L
        self.m = m
        
    def aberrations(self, chi, k):
        """Calculate nonlinear phase terms from propagation.

        Parameters
        ----------
        chi : ndarray
            Array of shape (4, N, N) containing field parameters
        k : float
            Wavenumber

        Returns
        -------
        float
            Sum of nonlinear phase terms
        """
        #shape of chi is (4, N, N)
        x, y, sx, sy = chi
        sx2, sy2 = sx@sx, sy@sy
        sx2psy2 = sx2+sy2
        sx2psy2_2 = sx2psy2@sx2psy2
        nonP4 = -self.L*k/8*(sx2psy2_2)
        nonP6 = -self.L*k/16*(sx2psy2_2@sx2psy2)
        #nonP8 = -5*self.dist*k/128*(sx2psy2@sx2psy2@sx2psy2@sx2psy2)
        
        return sum([nonP4, nonP6]) #nonP6, nonP8 

class Rot(ABCD):
    """Coordinate transformation through rotation.

    Parameters
    ----------
    r : ndarray
        2x2 rotation matrix
    """
    def __init__(self, r):
        # Returns the ABCD array for basis rotation with 2x2 matrix r
        self.r = r
        m = np.identity(4, dtype=np.float64)
        m[0:2,0:2] = r
        m[2:4,2:4] = r
        self.m = m

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
        """Get ABCD matrix at position x along optical axis.

        Parameters
        ----------
        x : float
            Position along optical axis
        distlist : list
            List of cumulative distances
        abcdlist : list
            List of cumulative ABCD matrices
        nlist : list
            List of refractive indices

        Returns
        -------
        ndarray
            ABCD matrix at position x
        """
        idx = np.where(x>=np.array(distlist))[0][-1]
        return Prop(x-distlist[idx], nlist[idx]).m@abcdlist[idx]
    
    @staticmethod
    def get_conj_pairs(ev):
        """Get conjugate eigenvector pairs with proper normalization.

        Parameters
        ----------
        ev : ndarray
            Array of eigenvectors

        Returns
        -------
        tuple
            Indices and normalized eigenvectors of conjugate pairs
        """
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
        """Calculate stability parameter of resonator.

        Parameters
        ----------
        Mrt : ndarray
            Round-trip ABCD matrix

        Returns
        -------
        float
            Stability parameter
        """
        # seems like maybe the second dimension gives a factor of 2? Maybe we should just use the norm of the eigenvalues
        return np.trace(Mrt)*0.5/2
    
    @staticmethod
    def stability_bool(Mrt):
        """Is the cavity stable?."""
        eps=0.001
        #return all([np.abs(np.linalg.norm(eva)-1)<eps for eva in np.linalg.eigvals(Mrt)])
        return all([(np.abs(eva)-1)<eps for eva in np.linalg.eigvals(Mrt)])
    
    @staticmethod
    def wrap_fsr(f, fsr):
        return np.mod(f,0.5*fsr) - 0.5*fsr*np.where(np.floor(2.*f/fsr)%2==1, 1, 0)
    
    def __init__(self, elements, wl = 780e-6):
        #unwarp possible nested elements
        self.elements = []
        for ele in elements:
            if isinstance(ele, list):
                self.elements.extend(ele)
            else:
                self.elements.append(ele)
        self.wl = wl
        self.k = 2*np.pi/wl
        #build functions
        distlist=[] #cumulative distances
        #optDistlist=[] #cumulative optical distances
        abcdlist=[] #cumulative abcd matrices
        nlist = []
        dtot=0.0
        dOptTot=0.0
        abcd=np.identity(4, dtype=np.float64)
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
                # set element position
                ele.x = dtot

        abcdlist.append(abcd.copy())
        distlist.append(dtot)
        #optDistlist.append(dOptTot)
        nlist.append(nlist[-1]) #TODO: Check this!
        
        self.abcd_rt = abcd
        self.distlist = distlist
        #self.optDistlist = optDistlist
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
        #self.is_stable = np.abs(self.stability(self.abcd_rt))<1.
        self.is_stable = self.stability_bool(self.abcd_rt)
        #assert self.m<1 , "Mode not stable!"
        #print("No eigenmode found!")
        
    def M2BiK(self, Mrt):
        """Convert ABCD matrix to normalized eigenvectors.

        Parameters
        ----------
        Mrt : ndarray
            Round-trip ABCD matrix

        Returns
        -------
        ndarray
            Normalized eigenvector matrix
        """
        # choose properly normalized pair if we can
        G = np.array([[0,0,1,0], [0,0,0,1], [-1,0,0,0], [0,-1,0,0]])
        ev, mus = np.linalg.eig(Mrt)
        nn0 = [mus[:,i].T.conj()@G@mus[:,i] for i in range(len(mus))]
        if any([ ni == 0 for ni in nn0]):
            nn0 = np.array([1,1,1,1])
        musN = [mus[:,i]*np.sqrt(2/nn0[i]) for i in range(len(mus))]
        nn1 = np.array([musN[i].T.conj()@G@musN[i] for i in range(len(musN))])
        goodIdx = [np.abs((ni-(-2j)))<0.0001 for ni in nn1]
        if np.sum(goodIdx)==2:
            return np.array(musN)[goodIdx]
        else:   # But if unstable or some weird numerics or something,....
            ind, es = ABCDSystem.get_conj_pairs(ev)
            mus = mus[:,ind]
            N = [mus[:,i].T.conj()@G@mus[:,i] for i in range(2)]
            if any([ Ni == 0 for Ni in N]):
                N = np.ones(np.shape(N))
            musN = np.stack([mus[:,i]*np.sqrt(2/N[i]) for i in range(2)], axis=0)
            return musN
        
    def M2waistNew(self, Mrt):
        ev, mus = np.linalg.eig(Mrt)
        G = np.array([[0,0,1,0], [0,0,0,1], [-1,0,0,0], [0,-1,0,0]])
        #Normalize the eigenvectors and take the ones that are normalized to -2j
        N = [mus[:,i].T.conj()@G@mus[:,i] for i in range(4)]
        musN = np.stack([mus[:,i]*np.sqrt(2/N[i]) for i in range(4)], axis=1)
        N = np.array([musN[:,i].T.conj()@G@musN[:,i] for i in range(4)])
        #print(N)
        idx = np.where(np.abs(N.imag+2)<1e-5)[0]
        idx = idx[::-1]
        musN = musN[:,idx]
        #Build B and K matrix, this is very sensitive! Double check again, see https://journals.aps.org/pra/pdf/10.1103/PhysRevA.75.033819
        B = np.array([[ musN[0,0], musN[1,0] ],[ musN[0,1], musN[1,1] ]])
        K = -1j*np.array([[ musN[2,0], musN[3,0] ],[ musN[2,1], musN[3,1] ]])
        # Q = B^-1 K
        Q = np.linalg.solve(B,K) 
        di = np.linalg.eigvals(1j*Q)
        ws = np.sqrt(self.wl/(-n*np.pi)*((1/di.imag) if all(di.imag!=0) else np.array([np.nan, np.nan])))
        return {'ws': ws, 'B': B, 'K': K, 'Q': Q, 'mu': musN}

    def M2freq(self, Mrt, s=3):
        """Get transverse mode frequencies from ABCD matrix and roundtrip lentgh."""
        if not self.is_stable:
            return [np.nan, np.nan], [np.nan, np.nan]
        ev = np.linalg.eigvals(Mrt)
        freqs = np.angle(ev)*self.fsr/(2*np.pi)
        freqsA = np.sort(freqs)[-1:-3:-1]
        #freqsB = freqsA - fsr
        freqsC = ABCDSystem.wrap_fsr(s*freqsA, self.fsr) #same as matts version, only vectorized freqsC = np.array(list(self.wrapAroundFSR(self.fsr, 3.*f) for f in freqsA))
        return freqsA, freqsC
    
    def solve_mode(self, BiK, n=1.):
        """Get waists (diagonal) from eigenvectors."""
        #if not self.is_stable:
        #    return [0, 0]
        B = BiK[:,:2]
        K = -1j*BiK[:,2:]
        lam = self.wl
        Q = np.linalg.solve(B,K) # Q = B^-1 K
        di = np.linalg.eigvals(1j*Q)
        ws = np.sqrt(lam/(-n*np.pi)*((1/di.imag) if all(di.imag!=0) else np.array([np.nan, np.nan])))
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
        """Calculate beam waists at position x.

        Parameters
        ----------
        x : float
            Position along optical axis

        Returns
        -------
        ndarray
            Beam waists in both transverse dimensions
        """
        qp = self.propBiK(self.q, self.abcd_at(x))
        return self.solve_mode(qp, self.n_at(x))
    
    def compute_waists(self, x, qin=None):
        """Calculate beam waists at multiple positions.

        Parameters
        ----------
        x : ndarray
            Array of positions along optical axis
        qin : ndarray, optional
            Input q parameter, by default None

        Returns
        -------
        ndarray
            Array of beam waists at each position
        """
        if qin is None:
            qin = self.q
        ws = np.zeros((x.shape[0], 2))
        for i, l in enumerate(x):
            qp = self.propBiK(qin, self.abcd_at(l))
            ws[i,:] = self.solve_mode(qp, self.n_at(l))
        return ws
    
    def q_at(self, x):
        return self.propBiK(self.q, self.abcd_at(x))
        
    def get_freqs(self, s=3):
        """Get transverse mode frequencies.

        Parameters
        ----------
        s : int, optional
            Scaling factor, by default 3

        Returns
        -------
        tuple
            Frequencies and wrapped frequencies
        """
        return self.M2freq(self.abcd_rt, s=s)
    
    def make_realspace(self, a=5., b=None, N=200):
        if b is None:
            b=a
        ws = self.waist_at(0)
        xx=np.linspace(-a*ws[0], a*ws[0], N)
        dx = xx[1]-xx[0]
        yy=np.linspace(-b*ws[1], b*ws[1], N)
        dy = yy[1]-y[0]
        x, y = np.meshgrid(xx,yy, indexing='ij')
        r = np.dstack([x,y])
        self.r = r
    
    @staticmethod
    def chi_wf(phi, k):
        pdx = np.gradient(phi, dx, axis=0)
        pdy = np.gradient(phi, dx, axis=1)
        px = phi*x
        py = phi*y
        return np.stack([px, py, 1j*pdx/k, 1j*pdy/k], axis=0)

    
def propagate_ABCD(mu, M, Nrt=100):
    rs = np.empty((Nrt+1,4))
    r = mu.copy()
    rs[0,:] = r
    for i in range(Nrt):
        r = M@r
        rs[i+1,:] = r
    return rs
