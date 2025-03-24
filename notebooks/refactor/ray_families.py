import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from numba import njit, prange
import numpy as np

class GaussianModeRayFamily:
    def __init__(self, mode_type='LG', N=2, mu=2, w0=1.0, k=1.0):
        """
        Initialize a Gaussian mode ray family.
        
        Parameters:
        - mode_type: 'LG', 'HG', or 'GG'
        - N: mode order
        - mu: specific mode parameter (e.g., angular momentum for LG)
        - w0: beam waist width
        - k: wavenumber
        """
        self.mode_type = mode_type
        self.N = N
        self.mu = mu
        self.w0 = w0
        self.k = k
        
        # Parameters for GG mode (only used if mode_type is 'GG')
        self.theta = np.pi/4  # Default angle on Poincaré sphere
        self.phi = 0.0        # Default azimuth on Poincaré sphere

    def solif_angle(self, l):
        if abs(l) > self.N:
            return ValueError(f"l = {l} is greater than N = {self.N}")
        
        Omega = 2*np.pi*(self.N + 1 -l)/(self.N+1)
        return Omega
        
    def set_gg_parameters(self, theta, phi):
        """Set the direction parameters for a GG mode."""
        self.theta = theta
        self.phi = phi
    
    def generate_ray_family(self, num_eta=50, num_tau=50):
        """
        Generate a family of rays representing the Gaussian mode.
        
        Parameters:
        - num_eta: number of points along the Poincaré curve
        - num_tau: number of points around each ellipse
        
        Returns:
        - Dictionary containing ray positions, momenta, and Poincaré path
        """
        # Initialize arrays
        ray_positions = np.zeros((num_eta, num_tau, 2))  # (x,y) positions
        ray_momenta = np.zeros((num_eta, num_tau, 2))   # (px,py) momenta
        poincare_points = np.zeros((num_eta, 3))        # (s1,s2,s3) Stokes params
        
        # Parameter arrays
        eta_values = np.linspace(0, 2*np.pi, num_eta)
        phi_of_eta_values = np.zeros_like(eta_values)
        vartheta_of_eta_values = np.zeros_like(eta_values)
        tau_values = np.linspace(0, 2*np.pi, num_tau)
        
        # Define the Poincaré curve based on mode_type
        if self.mode_type == 'LG':
            # For LG: circle around vertical (s3) axis at latitude determined by mu/(N+1)
            latitude = np.arccos(self.mu/(self.N+1))  # θ = arccos(μ/(N+1))
            
            for i, eta in enumerate(eta_values):
                phi = eta  # Azimuthal angle on Poincaré sphere
                phi_of_eta_values[i] = phi
                vartheta_of_eta_values[i] = np.arcsin(self.mu/(self.N+1))
                poincare_points[i] = [
                    np.sin(latitude) * np.cos(phi),  # s1
                    np.sin(latitude) * np.sin(phi),  # s2
                    np.cos(latitude)                # s3
                ]
                
        elif self.mode_type == 'HG':
            # For HG: circle around horizontal (s1) axis at latitude determined by mu/(N+1)
            latitude = np.arccos(self.mu/(self.N+1))
            
            for i, eta in enumerate(eta_values):
                phi = eta  # Azimuthal angle on Poincaré sphere
                phi_of_eta_values[i] = phi
                vartheta_of_eta_values[i] = np.arcsin(self.mu/(self.N+1))
                poincare_points[i] = [
                    np.cos(latitude),                # s1
                    np.sin(latitude) * np.sin(eta),  # s2
                    np.sin(latitude) * np.cos(eta)   # s3
                ]
                
        elif self.mode_type == 'GG':
            # For GG: circle centered at (θ,φ) with radius determined by mu/(N+1)
            # The circle is on a plane perpendicular to the direction (θ,φ)
            center_direction = np.array([
                np.sin(self.theta) * np.cos(self.phi),
                np.sin(self.theta) * np.sin(self.phi),
                np.cos(self.theta)
            ])
            
            # Create two orthogonal unit vectors in the plane perpendicular to center_direction
            if np.abs(center_direction[2]) < 0.9:
                v1 = np.array([0, 0, 1])
            else:
                v1 = np.array([1, 0, 0])
                
            v1 = v1 - np.dot(v1, center_direction) * center_direction
            v1 = v1 / np.linalg.norm(v1)
            v2 = np.cross(center_direction, v1)
            
            # Calculate the radius on the Poincaré sphere
            radius = np.arccos(self.mu/(self.N+1))
            
            for i, eta in enumerate(eta_values):
                # Point on the circle around center_direction
                point = (np.cos(radius) * center_direction + 
                         np.sin(radius) * np.cos(eta) * v1 + 
                         np.sin(radius) * np.sin(eta) * v2)
                poincare_points[i] = point
        
        # Convert Poincaré points to ray parameters
        for i in range(num_eta):
            # Extract spherical coordinates from Poincaré point
            s1, s2, s3 = poincare_points[i]
            
            if self.mode_type == 'LG':
                phi = phi_of_eta_values[i]
                vartheta = vartheta_of_eta_values[i]

            else:
                # Convert to spherical coordinates
                phi = np.arctan2(s2, s1)
                theta = np.arccos(s3)
                
                # Convert to latitude angle (ϑ = π/2 - θ)
                vartheta = np.pi/2 - theta
                
            
            
            # Create rays for each point on the elliptic orbit
            for j, tau in enumerate(tau_values):
                # Create Jones vector components as in Eq. 3.4
                c = np.cos(vartheta/2)  # cos(ϑ/2)
                s = np.sin(vartheta/2)  # sin(ϑ/2)
                
                # vx = c * np.cos(phi/2) + 1j * s * np.sin(phi/2)
                # vy = s * np.cos(phi/2) - 1j * c * np.sin(phi/2)

                vx = c * np.cos(phi/2) - 1j * s * np.sin(phi/2)
                vy = c * np.sin(phi/2) + 1j * s * np.cos(phi/2)
                
                # Calculate Q and P from Jones vector using Eq. 3.5
                v = np.array([vx, vy])
                e_minus_i_tau = np.exp(-1j * tau)

                # Convert to physical position and momentum (Eq. 3.3)
                #2d version

                Q = np.real(v * e_minus_i_tau)
                P = np.imag(v * e_minus_i_tau)

                ray_positions[i,j,:] = self.w0 * np.sqrt(self.N+1) * Q
                ray_momenta[i,j,:] = (2*np.sqrt(self.N+1)/(self.k*self.w0)) * P
                
                # Q = np.real(vx * e_minus_i_tau)
                # P = np.imag(vx * e_minus_i_tau)
                
                # # Convert to physical position and momentum (Eq. 3.3)
                # ray_positions[i,j,0] = self.w0 * np.sqrt(self.N+1) * Q
                # ray_momenta[i,j,0] = (2*np.sqrt(self.N+1)/(self.k*self.w0)) * P
                
                # # Repeat for y-component
                # Q = np.real(vy * e_minus_i_tau)
                # P = np.imag(vy * e_minus_i_tau)
                
                # ray_positions[i,j,1] = self.w0 * np.sqrt(self.N+1) * Q
                # ray_momenta[i,j,1] = (2*np.sqrt(self.N+1)/(self.k*self.w0)) * P
        
        return {
            'positions': ray_positions,
            'momenta': ray_momenta,
            'poincare_points': poincare_points,
            'eta_values': eta_values,
            'tau_values': tau_values,
            'phi_of_eta_values': phi_of_eta_values,
            'vartheta_of_eta_values': vartheta_of_eta_values
        }
    
    def calculate_ROPL(self, ray_family):
        """
        Calculate the Reduced Optical Path Length for each ray.
        
        Parameters:
        - ray_family: output from generate_ray_family
        
        Returns:
        - S: array of ROPL values for each ray
        """
        poincare_points = ray_family['poincare_points']
        eta_values = ray_family['eta_values']
        tau_values = ray_family['tau_values']
        phi_of_eta_values = ray_family['phi_of_eta_values']
        vartheta_of_eta_values = ray_family['vartheta_of_eta_values']
        
        num_eta = len(eta_values)
        num_tau = len(tau_values)
        S = np.zeros((num_eta, num_tau))
        
        # Calculate the geometric phase Γ(η) - Equation 4.5
        Gamma = np.zeros(num_eta)
        
        for i in range(1, num_eta):
            # Extract parameters from Poincaré points
            s1_prev, s2_prev, s3_prev = poincare_points[i-1]
            s1_curr, s2_curr, s3_curr = poincare_points[i]
            
            # Convert to spherical coordinates
            # phi_prev = np.arctan2(s2_prev, s1_prev)
            # phi_curr = np.arctan2(s2_curr, s1_curr)
            # theta_prev = np.arccos(s3_prev)
            # theta_curr = np.arccos(s3_curr)
            phi_prev = phi_of_eta_values[i-1]
            phi_curr = phi_of_eta_values[i]
            theta_prev = vartheta_of_eta_values[i-1]
            theta_curr = vartheta_of_eta_values[i]


            
            # Handle 2π jumps in phi
            if phi_curr - phi_prev > np.pi:
                phi_curr -= 2*np.pi
            elif phi_curr - phi_prev < -np.pi:
                phi_curr += 2*np.pi
            
            # Geometric phase increment (Eq. 4.5)
            d_eta = eta_values[i] - eta_values[i-1]
            
            # Average value over the interval
            sin_theta = 0.5 * (np.sin(theta_prev) + np.sin(theta_curr))
            d_phi = (phi_curr - phi_prev) #/ d_eta
            
            Gamma[i] = Gamma[i-1] + 0.5 * sin_theta * d_phi #* d_eta
        
        # Calculate S for each ray (Eq. 4.7)
        for i in range(num_eta):
            # theta = np.arccos(poincare_points[i,2])
            # vartheta = np.pi/2 - theta  # Latitude angle
            vartheta = vartheta_of_eta_values[i]
            
            for j, tau in enumerate(tau_values):
                # S₁,η(τ) component (Eq. 4.3)
                S1 = (self.N+1) * (tau - 0.5 * np.cos(vartheta) * np.sin(2*tau))
                
                # S₂(η) component (Eq. 4.6)
                S2 = (self.N+1) * Gamma[i]
                
                # Total ROPL (Eq. 4.7)
                S[i,j] = S1 + S2
        
        return S, S1, S2, Gamma
    
    def visualize_poincare_sphere(self, ray_family, ax=None):
        """
        Visualize the Poincaré sphere with the path representing the mode.
        
        Parameters:
        - ray_family: output from generate_ray_family
        - ax: Optional matplotlib 3D axis to plot on
        """
        poincare_points = ray_family['poincare_points']
        
        if ax is None:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        # Draw the Poincaré sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)
        
        # Draw coordinate axes
        ax.plot([-1.5, 1.5], [0, 0], [0, 0], 'k-', linewidth=1)
        ax.plot([0, 0], [-1.5, 1.5], [0, 0], 'k-', linewidth=1)
        ax.plot([0, 0], [0, 0], [-1.5, 1.5], 'k-', linewidth=1)
        
        # Label axes
        ax.text(1.6, 0, 0, '$s_1$')
        ax.text(0, 1.6, 0, '$s_2$')
        ax.text(0, 0, 1.6, '$s_3$')
        
        # Plot the equator
        phi_eq = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(phi_eq), np.sin(phi_eq), np.zeros_like(phi_eq), 'k-', linewidth=1, alpha=0.5)
        
        # Plot the Poincaré path with color gradient
        num_eta = len(poincare_points)
        colors = cm.jet(np.linspace(0, 1, num_eta))
        
        for i in range(num_eta-1):
            ax.plot(
                [poincare_points[i,0], poincare_points[i+1,0]],
                [poincare_points[i,1], poincare_points[i+1,1]],
                [poincare_points[i,2], poincare_points[i+1,2]],
                color=colors[i], linewidth=2
            )
        
        # Connect the last point to the first
        ax.plot(
            [poincare_points[-1,0], poincare_points[0,0]],
            [poincare_points[-1,1], poincare_points[0,1]],
            [poincare_points[-1,2], poincare_points[0,2]],
            color=colors[-1], linewidth=2
        )
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-1.2, 1.2)
        
        # Remove ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        return ax
    
    def visualize_ray_family(self, ray_family, ax=None, arrow_length=0.0):
        """
        Visualize the ray family in real space, colored by eta.
        
        Parameters:
        - ray_family: output from generate_ray_family
        - ax: Optional matplotlib axis to plot on
        """
        positions = ray_family['positions']
        momenta = ray_family['momenta']
        
        num_eta, num_tau = positions.shape[0:2]
        
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
        
        # Draw the ray family with color gradient by eta
        colors = cm.hsv(np.linspace(0, 1, num_eta))
        
        if arrow_length >0:
            for i in range(num_eta):
                for j in range(num_tau):
                    # Get the ray position and momentum
                    pos = positions[i,j]
                    mom = momenta[i,j]
                    
                    # Plot the ray as an arrow
                    ax.arrow(
                        pos[0], pos[1],
                        mom[0], mom[1],
                        head_width=0.05, head_length=0.1,
                        fc=colors[i], ec=colors[i],
                        alpha=0.7
                    )
        else:
            for i in range(num_eta):
                ax.scatter(positions[i,:,0], positions[i,:,1], color=colors[i], alpha=0.7)
                ax.plot(positions[i,:,0], positions[i,:,1], c=colors[i], alpha=0.7)
            
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Add axis labels
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def visualize_ray_family_tau(self, ray_family, ax=None):
        """
        Improved visualization of the ray family in real space with colors by eta.
        
        Parameters:
        - ray_family: output from generate_ray_family
        - ax: Optional matplotlib axis to plot on
        """
        positions = ray_family['positions']
        momenta = ray_family['momenta']
        
        num_eta, num_tau = positions.shape[0:2]
        
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
        
        # Draw the ray positions as ellipses (orbits)
        for i in range(num_eta):
            # Create color gradient around each orbit
            orbit_colors = cm.hsv(np.linspace(0, 1, num_tau))
            
            # Extract x and y positions for this orbit
            x = positions[i,:,0]
            y = positions[i,:,1]
            
            # Draw rays as line segments from each position in the direction of momentum
            for j in range(num_tau):
                # Scale factor for arrow length
                scale = 0.3
                
                # Draw the ray
                ax.arrow(
                    x[j], y[j],
                    scale * momenta[i,j,0], scale * momenta[i,j,1],
                    head_width=0.05, head_length=0.05,
                    color=orbit_colors[j], alpha=0.7
                )
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def visualize_all(self, fig=None):
        """
        Generate and visualize a complete set of ray family information.
        
        Returns:
        - fig: matplotlib figure containing the visualizations
        """
        if fig is None:
            fig = plt.figure(figsize=(18, 8))
        
        # Generate ray family
        ray_family = self.generate_ray_family(num_eta=30, num_tau=20)
        
        # Plot Poincaré sphere
        ax1 = fig.add_subplot(121, projection='3d')
        self.visualize_poincare_sphere(ray_family, ax=ax1)
        ax1.set_title(f'{self.mode_type}$_{{{self.N},{self.mu}}}$ Mode: Poincaré Sphere Path')
        
        # Plot ray family
        ax2 = fig.add_subplot(122)
        self.visualize_ray_family_(ray_family, ax=ax2)
        ax2.set_title(f'{self.mode_type}$_{{{self.N},{self.mu}}}$ Mode: Ray Family')
        
        plt.tight_layout()
        return fig
    
@njit(parallel=True)
def _cache_optimized_field_reconstruction(positions, momenta, eta_values, tau_values, 
                                         phi_values, theta_values, ropl, 
                                         grid_extent, grid_size, sigma=1.0, k=1.0):
    # Create evaluation grid
    x = np.linspace(grid_extent[0], grid_extent[1], grid_size)
    y = np.linspace(grid_extent[2], grid_extent[3], grid_size)

    sigma_squared = sigma**2
    norm = 1/(sigma * np.sqrt(np.pi)) 
    
    # Initialize complex field
    field = np.zeros((grid_size, grid_size), dtype=np.complex128)
    
    num_eta = len(eta_values)
    num_tau = len(tau_values)
    d_eta = 2*np.pi / num_eta
    d_tau = 2*np.pi / num_tau
    
    # Calculate cutoff distance for Gaussian (beyond which contribution is negligible)
    # (e.g., 4*sigma means we capture ~99.994% of the Gaussian)
    cutoff_distance = 5.0 * sigma
    cutoff_squared = cutoff_distance * cutoff_distance
    
    # Calculate grid spacing
    dx = (grid_extent[1] - grid_extent[0]) / (grid_size - 1)
    dy = (grid_extent[3] - grid_extent[2]) / (grid_size - 1)
    
    # Pre-compute all jacobian factors (can be reused across tau values)
    jacobian_factors = np.zeros(num_eta, dtype=np.complex128)
    for i in range(num_eta):
        phi = phi_values[i]
        theta = theta_values[i]
        
        phi_next = phi_values[(i+1) % num_eta]
        theta_next = theta_values[(i+1) % num_eta]
        
        # Handle 2π jumps
        if phi_next - phi > np.pi:
            phi_next -= 2*np.pi
        elif phi_next - phi < -np.pi:
            phi_next += 2*np.pi
        
        # Derivatives & Jacobian
        dphi_deta = (phi_next - phi) / d_eta
        dtheta_deta = (theta_next - theta) / d_eta
        # jac_real = 0.5 * (-dphi_deta * np.cos(theta))
        # jac_imag = 0.5 * dtheta_deta
        # jacobian_factors[i] = np.sqrt(complex(jac_real, jac_imag))
        jacobian_factors[i] = np.sqrt(0.5j * (dphi_deta * np.cos(theta) + 1j * dtheta_deta)) 
    
    # Outer parallel loop over y-grid values (each thread handles a row)
    for y_idx in prange(grid_size):
        y_val = y[y_idx]
        
        # Process this row sequentially
        for x_idx in range(grid_size):
            x_val = x[x_idx]
            r = np.array([x_val, y_val])
            
            # Initialize accumulator for this grid point
            field_sum = 0.0j
            
            # Process rays in blocks for better cache utilization of ray data
            block_size = 32  # Adjust based on your specific problem
            for i_block in range(0, num_eta, block_size):
                i_end = min(i_block + block_size, num_eta)
                
                # Process each ray in the current block
                for i in range(i_block, i_end):
                    jacobian_factor = jacobian_factors[i]
                    
                    for j in range(num_tau):
                        # Get ray position
                        q = positions[i, j]
                        p = momenta[i, j]
                        
                        # Check if within cutoff distance
                        # dist_squared = (r[0] - q[0])**2 + (r[1] - q[1])**2
                        # Calculate distance from center
                        dx = r[0] - q[0]
                        dy = r[1] - q[1]
                        dist_squared = dx**2 + dy**2
                        
                        # Phase from the ray direction
                        phase = k * (dx * p[0] + dy * p[1])
                        
                        if dist_squared <= cutoff_squared:
                            # Phase factor from ROPL
                            tau = tau_values[j]
                            phase_factor = np.exp(1j * ( phase + ropl[i, j] -  tau))
                            
                            # Gaussian
                            gaussian = norm*np.exp(-dist_squared / (sigma_squared))
                            
                            # Add contribution
                            field_sum += jacobian_factor * phase_factor * gaussian * d_eta * d_tau
            
            # Store the accumulated value for this grid point
            field[y_idx, x_idx] = field_sum
    
    return field

class GaussianFieldReconstructor:
    def __init__(self, gaussian_mode):
        """
        Initialize the field reconstructor with a GaussianModeRayFamily instance.
        """
        self.mode = gaussian_mode
    
    def fundamental_gaussian(self, r_grid, q, p):
        """
        Create a fundamental Gaussian beam centered at q with direction p.
        """
        # Calculate distance from center
        dx = r_grid[:, :, 0] - q[0]
        dy = r_grid[:, :, 1] - q[1]
        r_squared = dx**2 + dy**2
        
        # Phase from the ray direction
        phase = self.mode.k * (dx * p[0] + dy * p[1])
        
        # Gaussian amplitude
        amplitude = 1/(self.mode.w0 * np.sqrt(np.pi)) * np.exp(-r_squared / self.mode.w0**2)
        
        return amplitude * np.exp(1j * phase)
    
    def reconstruct_field(self, ray_family=None, grid_size=200, grid_extent=(-5, 5, -5, 5)):
        """
        Reconstruct the field using the SAFE method (Gaussian-dressed rays).
        """
        # Generate ray family if not provided
        if ray_family is None:
            ray_family = self.mode.generate_ray_family(num_eta=50, num_tau=40)
        
        # Calculate ROPL
        ropl, _, _, _ = self.mode.calculate_ROPL(ray_family)

        # Create evaluation grid
        x = np.linspace(grid_extent[0], grid_extent[1], grid_size)
        y = np.linspace(grid_extent[2], grid_extent[3], grid_size)
        x_grid, y_grid = np.meshgrid(x, y)

        if True:

            field = _cache_optimized_field_reconstruction(ray_family['positions'], 
                                                        ray_family['momenta'], 
                                                        ray_family['eta_values'], 
                                                        ray_family['tau_values'], 
                                                        ray_family['phi_of_eta_values'], 
                                                        ray_family['vartheta_of_eta_values'], 
                                                        ropl, 
                                                        grid_extent, 
                                                        grid_size)
            
            return field, x_grid, y_grid
        
        else:
            
            
            r_grid = np.stack([x_grid, y_grid], axis=-1)
            
            # Initialize complex field
            field = np.zeros((grid_size, grid_size), dtype=complex)
            
            # Extract parameters
            positions = ray_family['positions']
            momenta = ray_family['momenta']
            # poincare_points = ray_family['poincare_points']
            eta_values = ray_family['eta_values']
            tau_values = ray_family['tau_values']
            phi_of_eta_values = ray_family['phi_of_eta_values']
            vartheta_of_eta_values = ray_family['vartheta_of_eta_values']

            
            num_eta = len(eta_values)
            num_tau = len(tau_values)
            d_eta = 2*np.pi / num_eta
            d_tau = 2*np.pi / num_tau
            
            print(f"Reconstructing field with {num_eta}×{num_tau} rays...")
            
            # Equation 6.6 - Integrate over all rays
            for i in range(num_eta):
                if i % 10 == 0:
                    print(f"Processing ray family {i}/{num_eta}...")
                    
                # Extract spherical coordinates from Poincaré point
                # s1, s2, s3 = poincare_points[i]
                
                # Convert to spherical coordinates
                # phi = np.arctan2(s2, s1)
                # theta = np.arccos(s3)
                phi = phi_of_eta_values[i]
                theta = vartheta_of_eta_values[i]
                
                # Calculate Jacobian factor (Eq. 6.2)
                if i < num_eta - 1:
                    phi_next = phi_of_eta_values[i+1]
                    theta_next = vartheta_of_eta_values[i+1]
                else:
                    phi_next = phi_of_eta_values[0]
                    theta_next = vartheta_of_eta_values[0]
                    
                # Handle 2π jumps in phi
                if phi_next - phi > np.pi:
                    phi_next -= 2*np.pi
                elif phi_next - phi < -np.pi:
                    phi_next += 2*np.pi
                    
                # Derivatives
                dphi_deta = (phi_next - phi) / d_eta
                dtheta_deta = (theta_next - theta) / d_eta
                
                # Jacobian square root term (Eq. 6.2)
                jacobian_factor = np.sqrt(0.5j * (dphi_deta * np.cos(theta) + 1j * dtheta_deta)) 
                # jacobian_factor = 1.0
                
                for j in range(num_tau):
                    # Get ray position and momentum
                    q = positions[i,j]
                    p = momenta[i,j]
                    

                    tau = tau_values[j]
                    # Phase factor from ROPL
                    phase_factor = np.exp(1j * ropl[i,j] - 1j*tau)
                    
                    # Create Gaussian beam for this ray
                    gaussian = self.fundamental_gaussian(r_grid, q, p)
                    
                    # Add contribution to the field
                    field += jacobian_factor * phase_factor * gaussian * d_eta * d_tau
                    
            return field, x_grid, y_grid
    
    def visualize_field(self, field, x_grid, y_grid, ax=None, plot_type='intensity'):
        """
        Visualize the reconstructed field.
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            
        if plot_type == 'intensity':
            # Plot intensity |ψ|²
            intensity = np.abs(field)**2
            intensity_normalized = intensity / np.max(intensity)
            
            im = ax.pcolormesh(x_grid, y_grid, intensity_normalized, shading='auto', cmap='viridis')
            plt.colorbar(im, ax=ax, label='Normalized Intensity')
            ax.set_title('Field Intensity')
            
        elif plot_type == 'phase':
            # Plot phase arg(ψ)
            phase = np.angle(field)
            mask = np.abs(field)**2 < 0.01 * np.max(np.abs(field)**2)  # Mask low intensity regions
            masked_phase = np.ma.masked_where(mask, phase)
            
            im = ax.pcolormesh(x_grid, y_grid, masked_phase, shading='auto', cmap='hsv', 
                              vmin=-np.pi, vmax=np.pi)
            plt.colorbar(im, ax=ax, label='Phase (rad)')
            ax.set_title('Field Phase')
            
        elif plot_type == 'real_part':
            # Plot real part Re(ψ)
            real_part = np.real(field)
            
            im = ax.pcolormesh(x_grid, y_grid, real_part, shading='auto', cmap='RdBu_r')
            plt.colorbar(im, ax=ax, label='Re(ψ)')
            ax.set_title('Field Real Part')
            
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        
        return ax
    
    def visualize_all(self, ray_family=None, field=None, grid_size=200, grid_extent=(-5, 5, -5, 5)):
        """
        Complete pipeline to generate, reconstruct, and visualize a field.
        """
        # Generate ray family if not provided
        if ray_family is None:
            ray_family = self.mode.generate_ray_family(num_eta=50, num_tau=40)
            
        # Reconstruct field if not provided
        if field is None:
            field, x_grid, y_grid = self.reconstruct_field(ray_family, grid_size, grid_extent)
        else:
            x = np.linspace(grid_extent[0], grid_extent[1], grid_size)
            y = np.linspace(grid_extent[2], grid_extent[3], grid_size)
            x_grid, y_grid = np.meshgrid(x, y)
        
        # Create visualization figure
        fig = plt.figure(figsize=(18, 12))
        
        # Plot the Poincaré sphere
        ax1 = fig.add_subplot(231, projection='3d')
        self.mode.visualize_poincare_sphere(ray_family, ax=ax1)
        ax1.set_title(f'Poincaré Path: {self.mode.mode_type}$_{{{self.mode.N},{self.mode.mu}}}$')
        
        # Plot ray family (positions)
        ax2 = fig.add_subplot(232)
        self.mode.visualize_ray_family(ray_family, ax=ax2)
        ax2.set_title('Ray Positions')
        
        # Plot ray family with velocities
        ax3 = fig.add_subplot(233)
        self.mode.visualize_ray_family_tau(ray_family, ax=ax3)
        ax3.set_title('Ray Positions and Directions')
        
        # Plot field intensity
        ax4 = fig.add_subplot(234)
        self.visualize_field(field, x_grid, y_grid, ax=ax4, plot_type='intensity')
        
        # Plot field phase
        ax5 = fig.add_subplot(235)
        self.visualize_field(field, x_grid, y_grid, ax=ax5, plot_type='phase')
        
        # Plot field real part
        ax6 = fig.add_subplot(236)
        self.visualize_field(field, x_grid, y_grid, ax=ax6, plot_type='real_part')
        
        plt.tight_layout()
        mode_type = self.mode.mode_type
        N = self.mode.N
        mu = self.mode.mu
        fig.suptitle(f'{mode_type}$_{{{N},{mu}}}$ Mode Reconstruction', fontsize=16, y=0.99)
        
        return fig, field, x_grid, y_grid

