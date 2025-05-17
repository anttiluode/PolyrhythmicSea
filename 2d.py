import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d
from scipy.ndimage import label, center_of_mass # For pattern detection
import time

# --- Parameters ---
GRID_SIZE_2D = 128  # Can be larger for 2D
NUM_FIELDS = 50
DT = 0.05

# PolyrhythmicSea Parameters (from your Ursina version)
COUPLING = 0.100
NONLINEARITY_A = 1.000
NONLINEARITY_B = 1.000
DAMPING = 0.005
TENSION = 5.000
BASE_FREQUENCIES_MIN = 0.5
BASE_FREQUENCIES_MAX = 2.5
DIFFUSION_COEFFS_MIN = 0.05
DIFFUSION_COEFFS_MAX = 0.1
DRIVING_AMPLITUDE = 0.005 # from PolyrhythmicSea.step

# Pattern Detection (Simplified for 2D)
ENABLE_PATTERN_TRACKING = True
PATTERN_THRESHOLD = 0.7 # Threshold on summed phi for pattern detection (tune this)
MIN_PATTERN_AREA = 10   # Min pixels for a pattern

class PolyrhythmicSea2D:
    def __init__(self, N=GRID_SIZE_2D, dt=DT, num_initial_fields=NUM_FIELDS):
        self.N = N
        self.dt = dt
        self.num_fields = num_initial_fields

        # Core Substrate Parameters
        self.polyrhythm_coupling = COUPLING
        self.nonlinearity_A = NONLINEARITY_A
        self.nonlinearity_B = NONLINEARITY_B
        self.damping_factor = DAMPING
        self.tension = TENSION
        
        self.base_frequencies_min = BASE_FREQUENCIES_MIN
        self.base_frequencies_max = BASE_FREQUENCIES_MAX
        self.diffusion_coeffs_min = DIFFUSION_COEFFS_MIN
        self.diffusion_coeffs_max = DIFFUSION_COEFFS_MAX
        self.driving_amplitude = DRIVING_AMPLITUDE

        self.laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        self._initialize_fields_and_params()

        # Pattern tracking
        self.patterns = {} # {pid: {'pos': (r,c), 'age': age, 'area': area}}
        self.next_pattern_id = 0
        self.pattern_update_interval = 20 # steps
        self.step_count = 0


    def _initialize_fields_and_params(self):
        N = self.N
        self.phi_fields = [(np.random.rand(N, N).astype(np.float32) - 0.5) * 0.5
                           for _ in range(self.num_fields)]
        self.phi_o_fields = [np.copy(phi) for phi in self.phi_fields]
        
        self.base_frequencies = np.linspace(self.base_frequencies_min, self.base_frequencies_max, self.num_fields)
        self.diffusion_coeffs = np.linspace(self.diffusion_coeffs_max, self.diffusion_coeffs_min, self.num_fields) # Max to min
        self.field_phases = np.random.uniform(0, 2 * np.pi, self.num_fields)

        self.phi_sum = np.zeros((N, N), dtype=np.float32) # Main summed field
        self._update_summed_fields()
        self.t_sim = 0.0
        self.patterns = {}
        self.next_pattern_id = 0
        self.step_count = 0


    def _update_summed_fields(self):
        if self.phi_fields: # Check if list is not empty
            self.phi_sum = np.sum(self.phi_fields, axis=0) / max(1, len(self.phi_fields))
        else:
            self.phi_sum = np.zeros((self.N, self.N), dtype=np.float32)


    def _potential_deriv(self, field_k):
        # V'(phi) = -A*phi + B*phi^3 for potential V = -A/2 phi^2 + B/4 phi^4
        return -self.nonlinearity_A * field_k + self.nonlinearity_B * (field_k**3)

    def reset(self):
        self._initialize_fields_and_params()
        print("2D Simulation reset")

    def step(self):
        new_phi_list = []
        self.field_phases += self.base_frequencies * self.dt
        self.field_phases %= (2 * np.pi)
        
        current_sum_phi = np.sum(self.phi_fields, axis=0) # Sum once for all fields

        for k in range(self.num_fields):
            phi_k = self.phi_fields[k]
            phi_o_k = self.phi_o_fields[k]
            vel_k = phi_k - phi_o_k
            
            lap_k = convolve2d(phi_k, self.laplacian_kernel, mode='same', boundary='wrap')
            potential_deriv_k = self._potential_deriv(phi_k)
            
            coupling_denom = max(1, self.num_fields - 1)
            # Use the pre-calculated sum for efficiency
            other_fields_sum = current_sum_phi - phi_k 
            coupling_force = self.polyrhythm_coupling * other_fields_sum / coupling_denom
            
            driving_force_k = self.driving_amplitude * np.sin(self.field_phases[k])
            
            # FLSDAFL-like tension effect for each component field
            c2_factor = 1.0 / (1.0 + self.tension * phi_k**2 + 1e-6)
            
            acc = (c2_factor * self.diffusion_coeffs[k] * lap_k - 
                   potential_deriv_k + 
                   coupling_force + 
                   driving_force_k)
            
            new_phi_k = phi_k + (1 - self.damping_factor * self.dt) * vel_k + self.dt**2 * acc
            new_phi_list.append(new_phi_k)
        
        self.phi_o_fields = [p.copy() for p in self.phi_fields] # Save current as old for next step
        self.phi_fields = new_phi_list
        self._update_summed_fields()
        self.t_sim += self.dt
        self.step_count +=1

        if ENABLE_PATTERN_TRACKING and self.step_count % self.pattern_update_interval == 0:
            self._detect_patterns_2d()
            
    def _detect_patterns_2d(self):
        # Detect patterns in the summed field self.phi_sum
        binary_image = self.phi_sum > PATTERN_THRESHOLD
        labeled_array, num_features = label(binary_image)
        
        current_pids_found_this_cycle = set()

        for i in range(1, num_features + 1):
            mask = (labeled_array == i)
            area = np.sum(mask)

            if area < MIN_PATTERN_AREA:
                continue
            
            com_rc = center_of_mass(self.phi_sum, labels=labeled_array, index=i) # (row, col)
            com = (float(com_rc[0]), float(com_rc[1]))


            matched_pid = None
            min_dist_sq = (self.N / 5)**2 

            for pid, p_data in self.patterns.items():
                dist_sq = (p_data['pos'][0] - com[0])**2 + (p_data['pos'][1] - com[1])**2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    matched_pid = pid
            
            if matched_pid is not None:
                self.patterns[matched_pid]['pos'] = com
                self.patterns[matched_pid]['area'] = area
                self.patterns[matched_pid]['age'] += 1
                current_pids_found_this_cycle.add(matched_pid)
            else:
                new_id = self.next_pattern_id
                self.patterns[new_id] = {'pos': com, 'area': area, 'age': 0, 'id': new_id}
                current_pids_found_this_cycle.add(new_id)
                self.next_pattern_id += 1
        
        # Age out or remove patterns not found
        to_remove = []
        for pid, p_data in self.patterns.items():
            if pid not in current_pids_found_this_cycle:
                p_data['age'] -= 2 # Age faster if not seen
                if p_data['age'] < -10: # Remove if unseen for a while
                    to_remove.append(pid)
            else: # Reset age or increment if found
                 self.patterns[pid]['age'] = max(0, self.patterns[pid]['age'])


        for pid in to_remove:
            del self.patterns[pid]


    def apply_poke_2d(self, r, c, radius=5, amplitude=1.0):
        N = self.N
        # Ensure coordinates are valid
        r, c = int(r), int(c)
        if not (0 <= r < N and 0 <= c < N):
            return

        # Create coordinate grids for the poke area
        rr, cc = np.ogrid[:N, :N]
        dist_sq = (rr - r)**2 + (cc - c)**2
        
        # Gaussian pulse
        sigma_sq = (radius / 2.0)**2 # Sigma related to radius
        bump = amplitude * np.exp(-dist_sq / (2 * sigma_sq))
        
        # Apply bump to all fields (or a subset, e.g., lower frequency ones)
        for k in range(self.num_fields):
             # Weight poke more towards lower frequency fields
            weight = np.exp(-k / (self.num_fields / 5.0))
            self.phi_fields[k] += bump * weight
        print(f"Poked at ({r},{c})")


# --- Matplotlib Animation ---
sim_2d = PolyrhythmicSea2D(N=GRID_SIZE_2D, num_initial_fields=NUM_FIELDS)

fig, ax = plt.subplots(figsize=(8,8))
im = ax.imshow(sim_2d.phi_sum, cmap='viridis', vmin=-1.5, vmax=1.5, interpolation='bilinear') # Adjusted clim
pattern_markers, = ax.plot([], [], 'ro', markersize=10, alpha=0.5, label="Detected Patterns")
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', bbox=dict(facecolor='black', alpha=0.5))
pattern_count_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, color='white', bbox=dict(facecolor='black', alpha=0.5))

ax.set_title(f"2D Polyrhythmic Sea ({NUM_FIELDS} fields)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def init_animation():
    sim_2d.reset()
    # Initial poke to start dynamics
    sim_2d.apply_poke_2d(sim_2d.N // 2, sim_2d.N // 2, radius=GRID_SIZE_2D // 8, amplitude=2.0)
    im.set_data(sim_2d.phi_sum)
    pattern_markers.set_data([], [])
    time_text.set_text('')
    pattern_count_text.set_text('')
    return im, pattern_markers, time_text, pattern_count_text

def update_animation(frame):
    for _ in range(5): # Multiple sim steps per frame for speed
        sim_2d.step()
    
    im.set_data(sim_2d.phi_sum)
    vlim = np.max(np.abs(sim_2d.phi_sum)) * 1.1 + 1e-6 # Dynamic color limits
    im.set_clim(-vlim, vlim)
    
    # Update pattern markers
    if ENABLE_PATTERN_TRACKING and sim_2d.patterns:
        active_patterns = [p_data for p_id, p_data in sim_2d.patterns.items() if p_data['age'] >=0] # Only show non-negative age
        if active_patterns:
            pattern_x = [p['pos'][1] for p in active_patterns] # (row, col) -> (y,x) for plot
            pattern_y = [p['pos'][0] for p in active_patterns]
            pattern_markers.set_data(pattern_x, pattern_y)
        else:
            pattern_markers.set_data([], [])
    else:
        pattern_markers.set_data([], [])
        
    time_text.set_text(f"Time: {sim_2d.t_sim:.2f} | Step: {sim_2d.step_count}")
    pattern_count_text.set_text(f"Patterns: {len([p for p_id,p in sim_2d.patterns.items() if p['age']>=0])}")
    
    return im, pattern_markers, time_text, pattern_count_text

# --- Mouse Click for Poke ---
def onclick(event):
    if event.inaxes == ax:
        c, r = event.xdata, event.ydata # xdata is column, ydata is row
        if r is not None and c is not None:
            sim_2d.apply_poke_2d(int(r), int(c), radius=GRID_SIZE_2D // 10, amplitude=1.5)
            print(f"Clicked at grid ({int(r)}, {int(c)})")

fig.canvas.mpl_connect('button_press_event', onclick)

# Create and run animation
ani = animation.FuncAnimation(fig, update_animation, init_func=init_animation, 
                              frames=40000, interval=50, blit=True, repeat=False)

plt.show()

print("2D Simulation finished.")