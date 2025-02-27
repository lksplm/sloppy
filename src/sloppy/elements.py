import numpy as np
from .optic import Optic, Mirror, Screen, Glass, CurvedGlass, CurvedMirror, FreeFormInterface, FreeFormMirror, ThorlabsAsphere

class OpticalElement:
    """Base class for complete optical elements composed of multiple interfaces.
    
    An OpticalElement represents a complete optical component (lens, mirror, etc.)
    composed of one or more interfaces. It handles consistent positioning,
    orientation, and refractive indices across interfaces.
    """
    
    def __init__(self, position=(0., 0., 0.), orientation=(0., 0., 1.), 
                 ax=(1., 0., 0.), ay=(0., 1., 0.), diameter=1.0, 
                 position_reference='center'):
        """
        Args:
            position: Reference position of the element
            orientation: Direction normal vector
            ax, ay: Local coordinate system vectors
            diameter: Outer diameter of the element
            position_reference: Where the position is referenced from ('center', 'front', 'back')
        """
        self.position = np.array(position, dtype=np.float64)
        self.orientation = np.array(norm(orientation), dtype=np.float64)
        self.ax = np.array(ax, dtype=np.float64)
        self.ay = np.array(ay, dtype=np.float64)
        self.diameter = diameter
        self.position_reference = position_reference
        self.interfaces = []
        
        # Create rotation matrix for coordinate transforms
        self.Rot = np.stack((self.ax, self.ay, self.orientation)).T
        
    def get_interfaces(self):
        """Return all interfaces that make up this optical element."""
        return self.interfaces
    
    def _calculate_surface_positions(self, thickness):
        """Calculate surface positions based on reference position and thickness."""
        if self.position_reference == 'center':
            front_pos = self.position - (thickness/2) * self.orientation
            back_pos = self.position + (thickness/2) * self.orientation
        elif self.position_reference == 'front':
            front_pos = self.position
            back_pos = self.position + thickness * self.orientation
        elif self.position_reference == 'back':
            front_pos = self.position - thickness * self.orientation
            back_pos = self.position
        else:
            raise ValueError(f"Invalid position reference: {self.position_reference}")
            
        return front_pos, back_pos

class SphericalLens(OpticalElement):
    """A spherical lens with two interfaces (front and back surfaces)."""
    
    def __init__(self, position=(0., 0., 0.), orientation=(0., 0., 1.), 
                 ax=(1., 0., 0.), ay=(0., 1., 0.), diameter=1.0,
                 thickness=5.0, n_glass=1.5, R1=float('inf'), R2=float('inf'),
                 position_reference='center'):
        """
        Args:
            thickness: Center thickness of the lens
            n_glass: Refractive index of the lens material
            R1: Radius of first surface (positive: convex to incident light)
            R2: Radius of second surface (positive: convex to incident light)
        """
        super().__init__(position, orientation, ax, ay, diameter, position_reference)
        self.thickness = thickness
        self.n_glass = n_glass
        self.R1 = R1
        self.R2 = R2
        
        # Calculate surface positions
        self.front_pos, self.back_pos = self._calculate_surface_positions(thickness)
        
        # Create interfaces
        self._create_interfaces()
        
    def _create_interfaces(self):
        """Create the front and back interfaces of the lens."""
        # Front surface (air to glass)
        if np.isinf(self.R1):
            # Flat surface
            self.interfaces.append(
                Glass(p=self.front_pos, n=self.orientation, ax=self.ax, ay=self.ay,
                     diameter=self.diameter, n1=1.0, n2=self.n_glass)
            )
        else:
            # Curved surface
            curv = 'CX' if self.R1 > 0 else 'CC'
            self.interfaces.append(
                CurvedGlass(p=self.front_pos, n=self.orientation, ax=self.ax, ay=self.ay,
                           diameter=self.diameter, R=abs(self.R1), curv=curv,
                           n1=1.0, n2=self.n_glass)
            )
            
        # Back surface (glass to air)
        if np.isinf(self.R2):
            # Flat surface
            self.interfaces.append(
                Glass(p=self.back_pos, n=self.orientation, ax=self.ax, ay=self.ay,
                     diameter=self.diameter, n1=self.n_glass, n2=1.0)
            )
        else:
            # Curved surface - note orientation is reversed for back surface
            curv = 'CC' if self.R2 > 0 else 'CX'
            self.interfaces.append(
                CurvedGlass(p=self.back_pos, n=self.orientation, ax=self.ax, ay=self.ay,
                           diameter=self.diameter, R=abs(self.R2), curv=curv,
                           n1=self.n_glass, n2=1.0)
            )

class AsphericLens(OpticalElement):
    """Aspheric lens with potentially aspheric front and back surfaces."""
    
    def __init__(self, position=(0., 0., 0.), orientation=(0., 0., 1.), 
                 ax=(1., 0., 0.), ay=(0., 1., 0.), diameter=1.0,
                 thickness=5.0, n_glass=1.5, 
                 R1=float('inf'), coef1=None, 
                 R2=float('inf'), coef2=None,
                 position_reference='center'):
        """
        Args:
            coef1: Aspheric coefficients for front surface [R, k, A2, A4, ...]
            coef2: Aspheric coefficients for back surface [R, k, A2, A4, ...]
        """
        super().__init__(position, orientation, ax, ay, diameter, position_reference)
        self.thickness = thickness
        self.n_glass = n_glass
        self.R1 = R1
        self.R2 = R2
        self.coef1 = coef1
        self.coef2 = coef2
        
        # Calculate surface positions
        self.front_pos, self.back_pos = self._calculate_surface_positions(thickness)
        
        # Create interfaces
        self._create_interfaces()
        
    def _create_interfaces(self):
        """Create the front and back interfaces of the lens."""
        # Front surface (air to glass)
        if self.coef1 is not None:
            # Aspheric surface
            self.interfaces.append(
                ThorlabsAsphere(p=self.front_pos, n=self.orientation, ax=self.ax, ay=self.ay,
                               diameter=self.diameter, coef=self.coef1,
                               n1=1.0, n2=self.n_glass)
            )
        elif not np.isinf(self.R1):
            # Spherical surface
            curv = 'CX' if self.R1 > 0 else 'CC'
            self.interfaces.append(
                CurvedGlass(p=self.front_pos, n=self.orientation, ax=self.ax, ay=self.ay,
                           diameter=self.diameter, R=abs(self.R1), curv=curv,
                           n1=1.0, n2=self.n_glass)
            )
        else:
            # Flat surface
            self.interfaces.append(
                Glass(p=self.front_pos, n=self.orientation, ax=self.ax, ay=self.ay,
                     diameter=self.diameter, n1=1.0, n2=self.n_glass)
            )
            
        # Back surface (glass to air) - similar logic but with reversed indices
        if self.coef2 is not None:
            self.interfaces.append(
                ThorlabsAsphere(p=self.back_pos, n=self.orientation, ax=self.ax, ay=self.ay,
                               diameter=self.diameter, coef=self.coef2,
                               n1=self.n_glass, n2=1.0)
            )
        elif not np.isinf(self.R2):
            curv = 'CC' if self.R2 > 0 else 'CX'
            self.interfaces.append(
                CurvedGlass(p=self.back_pos, n=self.orientation, ax=self.ax, ay=self.ay,
                           diameter=self.diameter, R=abs(self.R2), curv=curv,
                           n1=self.n_glass, n2=1.0)
            )
        else:
            self.interfaces.append(
                Glass(p=self.back_pos, n=self.orientation, ax=self.ax, ay=self.ay,
                     diameter=self.diameter, n1=self.n_glass, n2=1.0)
            )

class CementedDoublet(OpticalElement):
    """A cemented doublet lens consisting of two lenses with three interfaces."""
    
    def __init__(self, position=(0., 0., 0.), orientation=(0., 0., 1.), 
                 ax=(1., 0., 0.), ay=(0., 1., 0.), diameter=1.0,
                 thickness1=3.0, n_glass1=1.5, R1=float('inf'),
                 thickness2=3.0, n_glass2=1.6, R2=float('inf'), R3=float('inf'),
                 position_reference='center'):
        """
        Args:
            thickness1, thickness2: Thicknesses of first and second lens elements
            n_glass1, n_glass2: Refractive indices of first and second glasses
            R1: Radius of first surface
            R2: Radius of cemented interface between elements
            R3: Radius of last surface
        """
        super().__init__(position, orientation, ax, ay, diameter, position_reference)
        self.thickness1 = thickness1
        self.thickness2 = thickness2
        self.total_thickness = thickness1 + thickness2
        self.n_glass1 = n_glass1
        self.n_glass2 = n_glass2
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        
        # Calculate interface positions
        self._calculate_interface_positions()
        
        # Create interfaces
        self._create_interfaces()
        
    def _calculate_interface_positions(self):
        """Calculate all three interface positions."""
        if self.position_reference == 'center':
            self.front_pos = self.position - (self.total_thickness/2) * self.orientation
            self.cement_pos = self.front_pos + self.thickness1 * self.orientation
            self.back_pos = self.cement_pos + self.thickness2 * self.orientation
        elif self.position_reference == 'front':
            self.front_pos = self.position
            self.cement_pos = self.front_pos + self.thickness1 * self.orientation
            self.back_pos = self.cement_pos + self.thickness2 * self.orientation
        elif self.position_reference == 'back':
            self.back_pos = self.position
            self.cement_pos = self.back_pos - self.thickness2 * self.orientation
            self.front_pos = self.cement_pos - self.thickness1 * self.orientation
        else:
            raise ValueError(f"Invalid position reference: {self.position_reference}")
    
    def _create_interfaces(self):
        """Create all three interfaces of the cemented doublet."""
        # Front surface (air to glass1)
        if np.isinf(self.R1):
            self.interfaces.append(
                Glass(p=self.front_pos, n=self.orientation, ax=self.ax, ay=self.ay,
                     diameter=self.diameter, n1=1.0, n2=self.n_glass1)
            )
        else:
            curv = 'CX' if self.R1 > 0 else 'CC'
            self.interfaces.append(
                CurvedGlass(p=self.front_pos, n=self.orientation, ax=self.ax, ay=self.ay,
                           diameter=self.diameter, R=abs(self.R1), curv=curv,
                           n1=1.0, n2=self.n_glass1)
            )
        
        # Cement interface (glass1 to glass2)
        if np.isinf(self.R2):
            self.interfaces.append(
                Glass(p=self.cement_pos, n=self.orientation, ax=self.ax, ay=self.ay,
                     diameter=self.diameter, n1=self.n_glass1, n2=self.n_glass2)
            )
        else:
            curv = 'CX' if self.R2 > 0 else 'CC'
            self.interfaces.append(
                CurvedGlass(p=self.cement_pos, n=self.orientation, ax=self.ax, ay=self.ay,
                           diameter=self.diameter, R=abs(self.R2), curv=curv,
                           n1=self.n_glass1, n2=self.n_glass2)
            )
        
        # Back surface (glass2 to air)
        if np.isinf(self.R3):
            self.interfaces.append(
                Glass(p=self.back_pos, n=self.orientation, ax=self.ax, ay=self.ay,
                     diameter=self.diameter, n1=self.n_glass2, n2=1.0)
            )
        else:
            curv = 'CC' if self.R3 > 0 else 'CX'
            self.interfaces.append(
                CurvedGlass(p=self.back_pos, n=self.orientation, ax=self.ax, ay=self.ay,
                           diameter=self.diameter, R=abs(self.R3), curv=curv,
                           n1=self.n_glass2, n2=1.0)
            )