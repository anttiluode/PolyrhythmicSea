
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from matplotlib.animation import FuncAnimation # Not used directly in this GUI update model
import matplotlib.gridspec as gridspec
# from matplotlib.colors import TwoSlopeNorm # Not used
from scipy.signal import convolve2d, hilbert, welch
from scipy.stats import linregress
from scipy.ndimage import label, center_of_mass
from scipy.spatial.transform import Rotation 
import tkinter as tk
from tkinter import ttk
import threading # Not strictly needed if GUI updates via root.after
import time
# import queue # Not used in this version

NUM_FIELDS = 200 # Now used by the class

# --- Core Parameters ---
GRID_SIZE = 128
INITIAL_NUM_FIELDS = 200 # Default number of fields for the simulator instance
DT_SIM = 0.05        

# Substrate Physics Parameters (Defaults for initialization)
BASE_FREQUENCIES_PARAM = np.linspace(0.5, 3.5, INITIAL_NUM_FIELDS) 
DIFFUSION_COEFFS_PARAM = np.linspace(0.08, 0.03, INITIAL_NUM_FIELDS)
NONLINEARITY_A_PARAM = 1.0
NONLINEARITY_B_PARAM = 1.5 
POLYRHYTHM_COUPLING_STRENGTH_PARAM = 0.25 
DAMPING_PARAM = 0.000 # Was 0.008, set to 0 as per discussion; can be tuned
TENSION_PARAM = 5.0 
DRIVING_AMPLITUDE_PARAM = 0.005

# Particle Detection & Tracking
PATTERN_THRESHOLD_FACTOR = 1.8 
MIN_LUMP_VOLUME = 5
PARTICLE_UPDATE_INTERVAL = 10 
PARTICLE_MAX_AGE_NO_DETECT = 60 

# Local CV Calculation for Particles
LOCAL_CV_PATCH_RADIUS = 5 
PARTICLE_LOCAL_ACTIVITY_HISTORY_SIZE = 128 
CV_WINDOW_SIZE_FOR_PARTICLE = 64 # Window for CV calc from particle's history

# Measurement Model
K_CV_TO_ANGLE_STD = 0.66
MEASUREMENT_TRIALS_PER_POINT = 100

# Analysis Windows for Global Substrate (Not plotted live in this GUI for speed)
PSD_WINDOW_SIZE_GLOBAL = 512
CV_WINDOW_SIZE_GLOBAL = 256


class ParticleLump2D:
    def __init__(self, pid, initial_pos_yx, initial_mask, t_creation):
        self.id = pid
        self.pos_yx = np.array(initial_pos_yx, dtype=float)
        self.mask = initial_mask.copy()
        self.area = np.sum(self.mask)
        self.creation_time = t_creation
        self.last_seen_time = t_creation
        self.is_active_this_cycle = True
        self.age_updates = 0

        self.orientation_vec_xy = np.array([1.0, 0.0]) 
        self.extract_orientation()

        self.local_substrate_activity_history = [] 
        self.local_CV = 0.1 
        self.update_local_substrate_CV(np.zeros((2*LOCAL_CV_PATCH_RADIUS+1, 2*LOCAL_CV_PATCH_RADIUS+1)))

    def extract_orientation(self):
        y_coords, x_coords = np.where(self.mask) 
        if len(x_coords) < 3: return

        x_mean, y_mean = np.mean(x_coords), np.mean(y_coords)
        points = np.vstack((x_coords - x_mean, y_coords - y_mean)).T
        
        if points.shape[0] >= 2:
            try:
                cov_matrix = np.cov(points, rowvar=False)
                if np.all(np.isfinite(cov_matrix)):
                    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                    major_axis_vec_2d = eigenvectors[:, np.argmax(eigenvalues)]
                    if major_axis_vec_2d[0] < 0: major_axis_vec_2d *= -1
                    self.orientation_vec_xy = major_axis_vec_2d
            except np.linalg.LinAlgError:
                pass 

    def update_local_substrate_CV(self, patch_sum_phi):
        self.local_substrate_activity_history.append(np.mean(patch_sum_phi))
        if len(self.local_substrate_activity_history) > PARTICLE_LOCAL_ACTIVITY_HISTORY_SIZE:
            self.local_substrate_activity_history.pop(0)

        if len(self.local_substrate_activity_history) >= CV_WINDOW_SIZE_FOR_PARTICLE : # Use dedicated CV window
            segment = np.array(self.local_substrate_activity_history[-CV_WINDOW_SIZE_FOR_PARTICLE:]) # Ensure correct window
            if np.std(segment) > 1e-7 : 
                try:
                    detrended_segment = segment - np.mean(segment) 
                    analytic_signal = hilbert(detrended_segment)
                    envelope = np.abs(analytic_signal)
                    if np.mean(envelope) > 1e-7:
                        cv = np.std(envelope) / np.mean(envelope)
                        self.local_CV = np.clip(cv, 0.01, 2.0)
                    else: self.local_CV = 0.01 
                except Exception: self.local_CV = 0.1 
            else: self.local_CV = 0.01
        # else: # Keep previous self.local_CV if not enough history

    def update_state(self, new_pos_yx, new_mask, current_time, substrate_patch_for_cv):
        self.pos_yx = np.array(new_pos_yx)
        self.mask = new_mask.copy()
        self.area = np.sum(self.mask)
        self.last_seen_time = current_time
        self.is_active_this_cycle = True
        self.age_updates +=1
        self.extract_orientation()
        self.update_local_substrate_CV(substrate_patch_for_cv)

    def check_if_aged_out(self, current_sim_time, sim_dt_param): # Pass DT
        if self.last_seen_time < current_sim_time - (PARTICLE_UPDATE_INTERVAL * sim_dt_param * (PARTICLE_MAX_AGE_NO_DETECT / PARTICLE_UPDATE_INTERVAL)):
            return True 
        return False
    
    def get_3d_orientation_for_jitter(self):
        return np.array([self.orientation_vec_xy[0], 0.0, self.orientation_vec_xy[1]])

    def __repr__(self):
        return (f"Lump(id={self.id}, pos=({self.pos_yx[1]:.1f},{self.pos_yx[0]:.1f}), "
                f"CV={self.local_CV:.3f}, age_upd={self.age_updates}, area={self.area}, "
                f"orient=({self.orientation_vec_xy[0]:.2f},{self.orientation_vec_xy[1]:.2f}))")

class QuantumSeaAndIslandsSimulator:
    def __init__(self, initialize_with_lumps=True, num_fields=INITIAL_NUM_FIELDS, dt=DT_SIM):
        self.grid_size = GRID_SIZE # Use global GRID_SIZE
        self.num_fields = num_fields
        self.dt = dt

        self.phi_fields = [(np.random.rand(self.grid_size, self.grid_size) - 0.5) * 0.2 for _ in range(self.num_fields)]
        if initialize_with_lumps:
            self._add_initial_lumps(num_lumps=1, strength_factor=2.0)

        self.phi_vel = [np.zeros((self.grid_size, self.grid_size)) for _ in range(self.num_fields)]
        self.t = 0.0
        self.step_count = 0
        
        self.sum_all_phi = np.zeros((self.grid_size, self.grid_size))
        self._update_sum_all_phi()

        self.laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        
        # Store parameters as instance attributes
        self.base_frequencies = np.linspace(0.5, 3.5, self.num_fields) 
        self.diffusion_coeffs = np.linspace(0.08, 0.03, self.num_fields)
        self.field_phases = np.random.uniform(0, 2 * np.pi, self.num_fields)
        
        self.nonlinearity_A = NONLINEARITY_A_PARAM
        self.nonlinearity_B = NONLINEARITY_B_PARAM
        self.polyrhythm_coupling = POLYRHYTHM_COUPLING_STRENGTH_PARAM
        self.damping = DAMPING_PARAM
        self.tension = TENSION_PARAM
        self.driving_amplitude = DRIVING_AMPLITUDE_PARAM
        
        self.tracked_particles = {} 
        self.next_particle_id = 0
        self.total_energy_history = []
        # Global substrate analysis (not used by GUI directly but can be added for debug)
        # self.local_activity_history_global = ... 

    def _add_initial_lumps(self, num_lumps=1, strength_factor=1.0):
        print(f"Adding {num_lumps} initial lumps...")
        y_coords_grid, x_coords_grid = np.ogrid[:self.grid_size, :self.grid_size]
        for _ in range(num_lumps):
            cy = self.grid_size / 2 + (np.random.rand() - 0.5) * (self.grid_size / 4)
            cx = self.grid_size / 2 + (np.random.rand() - 0.5) * (self.grid_size / 4)
            radius = self.grid_size / (10.0 + (np.random.rand()-0.5)*2) 
            lump_profile = strength_factor * 1.5 * np.exp(-((y_coords_grid - cy)**2 + (x_coords_grid - cx)**2) / (2 * radius**2))
            for i in range(self.num_fields):
                weight = np.exp(-i / (self.num_fields/4.0)) 
                self.phi_fields[i] += lump_profile * weight

    def _potential_deriv(self, field_k_component):
        return -self.nonlinearity_A * field_k_component + self.nonlinearity_B * (field_k_component**3)

    def _update_sum_all_phi(self):
        if self.phi_fields:
            self.sum_all_phi = np.sum(self.phi_fields, axis=0) / max(1, len(self.phi_fields))
        else:
            self.sum_all_phi = np.zeros((self.grid_size, self.grid_size))
            
    def _calculate_system_energy(self):
        kinetic_energy = 0.0
        potential_energy_self = 0.0
        potential_energy_gradient = 0.0 # For FLSDAFL part

        for i in range(self.num_fields):
            kinetic_energy += 0.5 * np.sum(self.phi_vel[i]**2) # Assuming phi_vel stores actual velocity now
            
            # Potential V(phi) = -A/2 * phi^2 + B/4 * phi^4
            potential_energy_self += np.sum(
                -self.nonlinearity_A * self.phi_fields[i]**2 / 2 +
                 self.nonlinearity_B * self.phi_fields[i]**4 / 4
            )
            
            # Gradient energy term (simplified, not full FLSDAFL energy)
            # lap_phi_i = convolve2d(self.phi_fields[i], self.laplacian_kernel, mode='same', boundary='wrap')
            # c2_factor_k = 1.0 / (1.0 + self.tension * self.phi_fields[i]**2 + 1e-9)
            # potential_energy_gradient += 0.5 * np.sum(c2_factor_k * self.diffusion_coeffs[i] * self.phi_fields[i] * (-lap_phi_i)) # E_grad = -1/2 * phi * D * nabla^2(phi)

        # More stable energy proxy for now
        # return np.sum(self.sum_all_phi**2) * self.dt # Multiply by dt to make it more like action
        return kinetic_energy + potential_energy_self # + potential_energy_gradient (can make it unstable if not careful)
        # Using sum of squares of phi_sum for simplicity in visualization
        # return np.sum(self.sum_all_phi**2)


    def detect_and_update_particles(self):
        current_sum_phi_np = self.sum_all_phi # Already updated in step()
        
        current_std = np.std(current_sum_phi_np)
        # Ensure threshold is positive and sensible even if field is mostly zero
        threshold = np.mean(current_sum_phi_np) + PATTERN_THRESHOLD_FACTOR * current_std \
            if current_std > 1e-6 else np.max(current_sum_phi_np) * 0.5 
        threshold = max(threshold, 0.05) 

        binary_image = current_sum_phi_np > threshold
        labeled_array, num_features = label(binary_image)
        
        current_pids_found_this_cycle = set()

        for i in range(1, num_features + 1):
            mask = (labeled_array == i)
            area = np.sum(mask)
            if area < MIN_LUMP_VOLUME: continue

            com_yx = center_of_mass(current_sum_phi_np, labels=labeled_array, index=i)
            if any(np.isnan(c) for c in com_yx): continue # Skip if CoM is NaN
            
            r_min = max(0, int(com_yx[0]) - LOCAL_CV_PATCH_RADIUS)
            r_max = min(self.grid_size, int(com_yx[0]) + LOCAL_CV_PATCH_RADIUS + 1)
            c_min = max(0, int(com_yx[1]) - LOCAL_CV_PATCH_RADIUS)
            c_max = min(self.grid_size, int(com_yx[1]) + LOCAL_CV_PATCH_RADIUS + 1)
            substrate_patch_for_cv = current_sum_phi_np[r_min:r_max, c_min:c_max]

            matched_pid = None
            min_dist_sq = (self.grid_size / 6)**2 
            for pid, p_obj in self.tracked_particles.items():
                dist_sq = (p_obj.pos_yx[0] - com_yx[0])**2 + (p_obj.pos_yx[1] - com_yx[1])**2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    matched_pid = pid
            
            if matched_pid is not None : 
                self.tracked_particles[matched_pid].update_state(com_yx, mask, self.t, substrate_patch_for_cv)
                current_pids_found_this_cycle.add(matched_pid)
            else: 
                new_id = self.next_particle_id
                self.tracked_particles[new_id] = ParticleLump2D(new_id, com_yx, mask, self.t)
                self.tracked_particles[new_id].update_local_substrate_CV(substrate_patch_for_cv)
                current_pids_found_this_cycle.add(new_id)
                self.next_particle_id += 1
        
        to_remove = []
        for pid, p_obj in list(self.tracked_particles.items()): 
            p_obj.is_active_this_cycle = pid in current_pids_found_this_cycle
            if not p_obj.is_active_this_cycle and p_obj.check_if_aged_out(self.t, self.dt): # Pass self.dt
                to_remove.append(pid)
        for pid in to_remove:
            del self.tracked_particles[pid]

    def step(self):
        new_phi_list = [np.zeros_like(p) for p in self.phi_fields]
        current_sum_all_phi_for_coupling = np.sum(self.phi_fields, axis=0)
        
        self.field_phases += self.base_frequencies * self.dt # Use self.dt
        self.field_phases %= (2 * np.pi)

        for i_field in range(self.num_fields):
            phi_k = self.phi_fields[i_field]
            # Corrected Verlet: phi_old is stored in self.phi_vel
            # vel_k = (phi_k - self.phi_vel[i_field]) / self.dt # More explicit velocity for standard Verlet
            # For the current leapfrog-like scheme:
            vel_k_approx = phi_k - self.phi_vel[i_field] # This is vel*dt if phi_vel is phi_old

            lap_phi_i = convolve2d(phi_k, self.laplacian_kernel, mode='same', boundary='wrap')
            force_potential = self.nonlinearity_A * phi_k - self.nonlinearity_B * (phi_k**3) # Corrected sign for V'
            
            coupling_denom = (self.num_fields - 1) if self.num_fields > 1 else 1
            other_fields_sum = current_sum_all_phi_for_coupling - phi_k 
            coupling_force = self.polyrhythm_coupling * other_fields_sum / coupling_denom
            
            driving_force_k = self.driving_amplitude * np.sin(self.field_phases[i_field])
            
            c2_factor_k = 1.0 / (1.0 + self.tension * phi_k**2 + 1e-9)

            acceleration = (c2_factor_k * self.diffusion_coeffs[i_field] * lap_phi_i + 
                            force_potential + 
                            coupling_force + 
                            driving_force_k) 
            
            # Verlet-like integration (current form in your script)
            # phi_new = phi_k + (1 - DAMPING*DT)*vel_k_approx + acc*DT^2
            # Store phi_k to become phi_old for next step
            phi_old_for_next_step = phi_k.copy()
            new_phi_k = phi_k + (1 - self.damping * self.dt) * vel_k_approx + acceleration * (self.dt**2)
            
            self.phi_vel[i_field] = phi_old_for_next_step # self.phi_vel now correctly stores phi_old
            new_phi_list[i_field] = new_phi_k
        
        self.phi_fields = new_phi_list
        self._update_sum_all_phi() 
        self.t += self.dt
        self.step_count += 1
        
        current_total_energy = self._calculate_system_energy()
        self.total_energy_history.append((self.t, current_total_energy)) 
        if len(self.total_energy_history) > 500: self.total_energy_history.pop(0)
        
        if self.step_count % PARTICLE_UPDATE_INTERVAL == 0:
            self.detect_and_update_particles()

# --- GUI Class (Simplified for this integrated demo) ---
# (GUI Class largely unchanged from your powerful.py, ensure it uses instance parameters correctly)
class IntegratedPreQMVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Integrated PreQM Substrate, Particle & Measurement Visualizer")
        # Pass specific dt and num_fields to the simulator instance
        self.sim = QuantumSeaAndIslandsSimulator(initialize_with_lumps=True, 
                                                 num_fields=INITIAL_NUM_FIELDS, 
                                                 dt=DT_SIM)
        self.measurement_model = PreQMMeasurementModel()
        
        self.running = False
        self.animation_job = None
        self.measurement_results_history = [] 
        self.selected_particle_id_for_measurement = None

        self._build_gui()
        self._setup_plots() 
        self.start_stop_sim() 

    def _build_gui(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.Frame(main_frame, padding=5)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        self.start_stop_button = ttk.Button(control_frame, text="Pause Sim", command=self.start_stop_sim)
        self.start_stop_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Reset Sim", command=self.reset_sim_gui).pack(side=tk.LEFT, padx=5)
        
        measure_frame = ttk.Frame(control_frame)
        measure_frame.pack(side=tk.LEFT, padx=20)
        ttk.Label(measure_frame, text="Prep. Angle α (°):").pack(side=tk.LEFT)
        self.alpha_slider = ttk.Scale(measure_frame, from_=0, to=180, orient='horizontal', length=150)
        self.alpha_slider.set(45); self.alpha_slider.pack(side=tk.LEFT, padx=5)
        ttk.Button(measure_frame, text=f"Run Measurements", command=self.run_born_measurements_gui).pack(side=tk.LEFT, padx=5) # Removed {MEASUREMENT_TRIALS}
        ttk.Button(measure_frame, text="Clear Born Plot", command=self.clear_born_plot_gui).pack(side=tk.LEFT, padx=5)

        plot_frame = tk.Frame(main_frame)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(15, 8)) 
        gs = self.fig.add_gridspec(2, 3, height_ratios=[1,1.2]) 
        self.ax_sum_phi = self.fig.add_subplot(gs[0, 0])
        self.ax_particle_cv = self.fig.add_subplot(gs[0, 1])
        self.ax_energy = self.fig.add_subplot(gs[0, 2])
        self.ax_born_rule = self.fig.add_subplot(gs[1, :]) 

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.status_var = tk.StringVar(value="Initializing...")
        ttk.Label(main_frame, textvariable=self.status_var).pack(side=tk.BOTTOM, fill=tk.X)

    def _setup_plots(self):
        self.im_sum_phi = self.ax_sum_phi.imshow(np.zeros((self.sim.grid_size,self.sim.grid_size)), cmap='viridis', origin='lower', vmin=-0.5, vmax=0.5, interpolation='bilinear') # Use self.sim.grid_size
        self.fig.colorbar(self.im_sum_phi, ax=self.ax_sum_phi, fraction=0.046, pad=0.04)
        self.particle_markers_plot, = self.ax_sum_phi.plot([],[], 'ro', markersize=6, alpha=0.8)
        self.orientation_arrows_plots = []
        self.ax_sum_phi.set_title(f"Total Field (Sum of {self.sim.num_fields})") # Use self.sim.num_fields
        self.ax_sum_phi.set_xticks([]); self.ax_sum_phi.set_yticks([])

        self.ax_particle_cv.set_title("Local CV of Tracked Particles")
        self.ax_particle_cv.set_xlabel("Particle Age (updates)")
        self.ax_particle_cv.set_ylabel("Local CV"); self.ax_particle_cv.set_ylim(0, 1.0)
        self.ax_particle_cv.axhline(0.5, color='r', linestyle='--', label="Target CV=0.5")
        self.ax_particle_cv.grid(True)
        
        self.energy_line_plot, = self.ax_energy.plot([],[])
        self.ax_energy.set_title("Total System Energy")
        self.ax_energy.set_xlabel("Time (s)")
        self.ax_energy.set_ylabel("Energy (arb. units)"); self.ax_energy.grid(True)

        alphas_ideal_deg = np.linspace(0, 180, 50)
        alphas_ideal_rad = np.deg2rad(alphas_ideal_deg)
        self.ax_born_rule.plot(alphas_ideal_deg, np.cos(alphas_ideal_rad/2)**2, 'k--', label="Ideal QM", lw=2)
        self.born_scatter_plot, = self.ax_born_rule.plot([], [], 'bo', alpha=0.7, label="Simulated P(0)")
        self.ax_born_rule.set_title("Born Rule Emergence P(0|α)")
        self.ax_born_rule.set_xlabel("Preparation Angle α (degrees from +Z)")
        self.ax_born_rule.set_ylabel("Prob P(Outcome +Z)")
        self.ax_born_rule.set_xlim(-5, 185); self.ax_born_rule.set_ylim(-0.05, 1.05)
        self.ax_born_rule.legend(fontsize='small'); self.ax_born_rule.grid(True)

        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.fig.suptitle("Integrated PreQM Substrate, Particle & Measurement Visualizer", fontsize=14)

    def update_gui_and_sim_step(self):
        if not self.running: return

        self.sim.step() 

        sum_phi_display = self.sim.sum_all_phi
        self.im_sum_phi.set_data(sum_phi_display)
        # More robust vlim:
        abs_max = np.max(np.abs(sum_phi_display))
        vlim = max(abs_max * 0.8, 1e-5) # Ensure vlim is positive and not too small
        self.im_sum_phi.set_clim(-vlim, vlim)
        
        active_p_x, active_p_y = [], []
        for arr in self.orientation_arrows_plots: arr.remove()
        self.orientation_arrows_plots = []
        
        status_particle_cv = 0.0
        active_particles_for_cv_plot = []

        for pid, p_obj in self.sim.tracked_particles.items():
            if p_obj.is_active_this_cycle:
                active_p_x.append(p_obj.pos_yx[1]) 
                active_p_y.append(p_obj.pos_yx[0])
                active_particles_for_cv_plot.append(p_obj)
                if status_particle_cv == 0.0 and p_obj.local_CV > 0.01 : # Take first non-trivial CV
                    status_particle_cv = p_obj.local_CV

                length = 3 
                dx = p_obj.orientation_vec_xy[0] * length 
                dy = -p_obj.orientation_vec_xy[1] * length 
                arrow = self.ax_sum_phi.arrow(p_obj.pos_yx[1], p_obj.pos_yx[0], dx, dy,
                                              head_width=1.0, head_length=1.5, fc='cyan', ec='cyan',
                                              length_includes_head=True, zorder=10)
                self.orientation_arrows_plots.append(arrow)
        self.particle_markers_plot.set_data(active_p_x, active_p_y)
        
        self.ax_particle_cv.clear() 
        self.ax_particle_cv.set_title("Local CV of Tracked Particles")
        self.ax_particle_cv.set_xlabel("Particle Age (updates since detection)")
        self.ax_particle_cv.set_ylabel("Local CV"); self.ax_particle_cv.set_ylim(0, 1.0)
        self.ax_particle_cv.axhline(0.5, color='r', linestyle='--', label="Target CV=0.5")
        self.ax_particle_cv.grid(True)
        plotted_legend_for_cv = False
        
        max_age_for_xlim = 0 # For dynamic xlim of CV plot
        for p_obj in active_particles_for_cv_plot:
            # Recompute CV history for plotting based on local_substrate_activity_history
            cv_plot_data = []
            current_activity_segment = []
            for activity_val in p_obj.local_substrate_activity_history:
                current_activity_segment.append(activity_val)
                if len(current_activity_segment) > PARTICLE_LOCAL_ACTIVITY_HISTORY_SIZE: # Keep window
                    current_activity_segment.pop(0)

                if len(current_activity_segment) >= CV_WINDOW_SIZE_FOR_PARTICLE//2: # Min data for CV
                    seg_arr = np.array(current_activity_segment)
                    if np.std(seg_arr) > 1e-7:
                        try:
                            analytic = hilbert(seg_arr - np.mean(seg_arr))
                            env = np.abs(analytic)
                            if np.mean(env) > 1e-7:
                                cv_plot_data.append(np.clip(np.std(env)/np.mean(env),0,2))
                            else: cv_plot_data.append(0.01)
                        except: cv_plot_data.append(0.1)
                    else: cv_plot_data.append(0.01)
                else: cv_plot_data.append(0.1) # Default for insufficient history

            if len(cv_plot_data) > 1:
                # X-axis for particle CV plot should be number of updates for that particle
                x_axis_cv = np.arange(max(0, p_obj.age_updates - len(cv_plot_data) + 1), p_obj.age_updates + 1)
                if len(x_axis_cv) == len(cv_plot_data): # Ensure lengths match
                    self.ax_particle_cv.plot(x_axis_cv, cv_plot_data, label=f"P{p_obj.id}", lw=1, alpha=0.9)
                    plotted_legend_for_cv = True
                    if x_axis_cv[-1] > max_age_for_xlim: max_age_for_xlim = x_axis_cv[-1]
                # else:
                #     print(f"Plotting CV mismatch for P{p_obj.id}: age_upd={p_obj.age_updates}, len_cv_data={len(cv_plot_data)}, len_x_axis={len(x_axis_cv)}")


        if plotted_legend_for_cv: self.ax_particle_cv.legend(fontsize='x-small', loc='upper right')
        if max_age_for_xlim > 0: self.ax_particle_cv.set_xlim(0, max_age_for_xlim + 1)
        else: self.ax_particle_cv.set_xlim(0, 10)


        if self.sim.total_energy_history:
            times, energies = zip(*self.sim.total_energy_history)
            self.energy_line_plot.set_data(times, energies)
            if times: self.ax_energy.set_xlim(max(0, times[0]-1), times[-1] + 1) 
            if energies:
                min_e, max_e = np.min(energies), np.max(energies)
                if max_e > min_e : self.ax_energy.set_ylim(min_e - 0.1*abs(min_e) if min_e!=0 else -0.1, 
                                                           max_e + 0.1*abs(max_e) if max_e!=0 else 0.1)
                else: self.ax_energy.set_ylim(min_e - 0.1, max_e + 0.1)

        self.canvas.draw_idle()
        num_active_particles = len([p for pid, p in self.sim.tracked_particles.items() if p.is_active_this_cycle])
        self.status_var.set(f"Time: {self.sim.t:.2f} | Step: {self.sim.step_count} | Particles: {num_active_particles} | Sel.P CV: {status_particle_cv:.3f}")

        if self.running:
            self.animation_job = self.root.after(30, self.update_gui_and_sim_step)

    # (start_stop_sim, reset_sim_gui, run_born_measurements_gui, _update_born_plot_data_gui, clear_born_plot_gui, on_close as before)
    def start_stop_sim(self):
        self.running = not self.running
        self.start_stop_button.config(text="Pause Sim" if self.running else "Resume Sim")
        if self.running:
            self.update_gui_and_sim_step()

    def reset_sim_gui(self):
        was_running = self.running
        self.running = False
        if self.animation_job: self.root.after_cancel(self.animation_job); self.animation_job = None
        
        self.sim = QuantumSeaAndIslandsSimulator(initialize_with_lumps=True, num_fields=INITIAL_NUM_FIELDS, dt=DT_SIM)
        self.measurement_results_history = []
        self.clear_born_plot_gui()
        
        self.ax_particle_cv.clear() 
        self.ax_particle_cv.set_title("Local CV of Tracked Particles"); self.ax_particle_cv.set_xlabel("Particle Age (updates)")
        self.ax_particle_cv.set_ylabel("Local CV"); self.ax_particle_cv.set_ylim(0, 1.0)
        self.ax_particle_cv.axhline(0.5, color='r', linestyle='--', label="Target CV=0.5"); 
        self.ax_particle_cv.grid(True)
        # Re-add legend after clear
        handles, labels = self.ax_particle_cv.get_legend_handles_labels()
        if "Target CV=0.5" not in labels : self.ax_particle_cv.axhline(0.5, color='r', linestyle='--', label="Target CV=0.5") # ensure it's there
        self.ax_particle_cv.legend(fontsize='small', loc='upper right')


        if hasattr(self, 'energy_line_plot'): self.energy_line_plot.set_data([],[])
        self.ax_energy.set_xlim(0,1); self.ax_energy.set_ylim(0,1) 

        print("Simulation Reset.")
        self.status_var.set("Status: Reset. Press Start/Resume.")
        self.start_stop_button.config(text="Start Sim" if not was_running else "Pause Sim") 
        # if was_running: self.start_stop_sim() # Optionally restart if it was running


    def run_born_measurements_gui(self): 
        active_particles = [p for p_id, p in self.sim.tracked_particles.items() if p.is_active_this_cycle]
        if not active_particles:
            print("No active particle found to perform measurements on."); return
        
        selected_particle = active_particles[0] 
        self.selected_particle_id_for_measurement = selected_particle.id

        alpha_deg = self.alpha_slider.get()
        prob_0_val = self.measurement_model.perform_measurement_trials(selected_particle, alpha_deg)

        if not np.isnan(prob_0_val):
            self.measurement_results_history.append((alpha_deg, prob_0_val))
            self.measurement_results_history.sort(key=lambda x: x[0]) 
            self._update_born_plot_data_gui() 
            print(f"  P(0|α={alpha_deg:.1f}°) = {prob_0_val:.3f} (using particle {selected_particle.id} CV={selected_particle.local_CV:.3f})")
        else:
            print(f"Measurement failed for particle {selected_particle.id} at α={alpha_deg:.1f}°")

    def _update_born_plot_data_gui(self): 
        if self.measurement_results_history:
            alphas, probs = zip(*self.measurement_results_history)
            self.born_scatter_plot.set_data(alphas, probs)
        else:
            self.born_scatter_plot.set_data([],[])
        self.canvas.draw_idle()

    def clear_born_plot_gui(self): 
        self.measurement_results_history = []
        self._update_born_plot_data_gui()

    def on_close(self):
        print("Closing GUI..."); self.running = False
        if self.animation_job: self.root.after_cancel(self.animation_job)
        self.root.destroy()

# --- Measurement Model ---
class PreQMMeasurementModel:
    def __init__(self):
        # Ensure this is the name used consistently
        self.apparatus_axis_A0_3d = np.array([0.0, 0.0, 1.0]) # Measure along lab +Z

    def _get_jittered_orientation_3d(self, ideal_orientation_vec_3d, angular_std_dev):
        # (Copied from previous script - robust version)
        if angular_std_dev < 1e-6: return ideal_orientation_vec_3d
        if np.allclose(ideal_orientation_vec_3d, [0,0,0]): return ideal_orientation_vec_3d

        if np.abs(ideal_orientation_vec_3d[0]) < 1e-6 and np.abs(ideal_orientation_vec_3d[1]) < 1e-6: 
            p_axis = np.array([1.0, 0.0, 0.0])
        else:
            p_axis = np.cross(ideal_orientation_vec_3d, np.array([0.0, 0.0, 1.0]))
        
        if np.linalg.norm(p_axis) < 1e-9:
            p_axis = np.array([1.0, 0.0, 0.0])
            if np.allclose(ideal_orientation_vec_3d, [1.0, 0.0, 0.0]): p_axis = np.array([0.0, 1.0, 0.0])

        if np.linalg.norm(p_axis) < 1e-9:
             return ideal_orientation_vec_3d 

        p_axis_unit = p_axis / np.linalg.norm(p_axis)
        j_angle = np.random.normal(0, angular_std_dev)
        rot = Rotation.from_rotvec(j_angle * p_axis_unit)
        return rot.apply(ideal_orientation_vec_3d)

    def perform_measurement_trials(self, particle_lump, prepared_alpha_deg, num_trials=MEASUREMENT_TRIALS_PER_POINT):
        if particle_lump is None: return np.nan
        
        alpha_rad = np.deg2rad(prepared_alpha_deg)
        particle_local_cv = particle_lump.local_CV
        angular_std_dev = K_CV_TO_ANGLE_STD * particle_local_cv

        ideal_L_x = np.sin(alpha_rad)
        ideal_L_z = np.cos(alpha_rad)
        vec_L_ideal_prepared_3d = np.array([ideal_L_x, 0.0, ideal_L_z])
        
        outcomes_0_count = 0 
        for _ in range(num_trials):
            vec_L_jittered_3d = self._get_jittered_orientation_3d(vec_L_ideal_prepared_3d, angular_std_dev)
            # Ensure this line uses the correctly initialized attribute name:
            projection_A0 = np.dot(vec_L_jittered_3d, self.apparatus_axis_A0_3d) # <<< CHANGED TO _3d
            prob_outcome_A0 = 0.5 * (1 + projection_A0)
            prob_outcome_A0 = np.clip(prob_outcome_A0, 0, 1)
            if np.random.rand() < prob_outcome_A0:
                outcomes_0_count += 1
        
        return outcomes_0_count / num_trials

# --- GUI Class ---
class QuantumIslandGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Quantum Island in Polyrhythmic Sea")
        self.sim = QuantumSeaAndIslandsSimulator(initialize_with_lumps=True)
        self.measurement_model = PreQMMeasurementModel()
        
        self.running = False
        self.animation_job = None
        self.measurement_results_history = [] # Store (alpha_deg, P0_val)
        self.selected_particle_id_for_measurement = None

        self._build_gui()
        self._setup_plots()
        self.start_stop_sim() # Auto-start

    def _build_gui(self):
        # (Similar to PreQMGaugeMeasurementGUI, simplified)
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.Frame(main_frame, padding=5)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        self.start_stop_button = ttk.Button(control_frame, text="Pause Sim", command=self.start_stop_sim)
        self.start_stop_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Reset Sim", command=self.reset_sim_gui).pack(side=tk.LEFT, padx=5)
        
        measure_frame = ttk.Frame(control_frame)
        measure_frame.pack(side=tk.LEFT, padx=20)
        ttk.Label(measure_frame, text="Prep. Angle α (°):").pack(side=tk.LEFT)
        self.alpha_slider = ttk.Scale(measure_frame, from_=0, to=180, orient='horizontal', length=150)
        self.alpha_slider.set(45); self.alpha_slider.pack(side=tk.LEFT, padx=5)
        ttk.Button(measure_frame, text=f"Run Measurements", command=self.run_born_measurements_gui).pack(side=tk.LEFT, padx=5)
        ttk.Button(measure_frame, text="Clear Born Plot", command=self.clear_born_plot_gui).pack(side=tk.LEFT, padx=5)

        plot_frame = tk.Frame(main_frame)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(15, 8))
        gs = self.fig.add_gridspec(2, 3, height_ratios=[1,1.2])
        self.ax_sum_phi = self.fig.add_subplot(gs[0, 0])
        self.ax_particle_cv = self.fig.add_subplot(gs[0, 1])
        self.ax_energy = self.fig.add_subplot(gs[0, 2])
        self.ax_born_rule = self.fig.add_subplot(gs[1, :])

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.status_var = tk.StringVar(value="Initializing...")
        ttk.Label(main_frame, textvariable=self.status_var).pack(side=tk.BOTTOM, fill=tk.X)

    def _setup_plots(self):
        # Plot 1: Sum Phi
        self.im_sum_phi = self.ax_sum_phi.imshow(np.zeros((GRID_SIZE,GRID_SIZE)), cmap='viridis', origin='lower', vmin=-1.0, vmax=1.0, interpolation='bilinear')
        self.fig.colorbar(self.im_sum_phi, ax=self.ax_sum_phi, fraction=0.046, pad=0.04)
        self.particle_markers_plot, = self.ax_sum_phi.plot([],[], 'ro', markersize=8, alpha=0.7)
        self.orientation_arrows_plots = []
        self.ax_sum_phi.set_title(f"Total Field (Sum of {NUM_FIELDS})")
        self.ax_sum_phi.set_xticks([]); self.ax_sum_phi.set_yticks([])

        # Plot 2: Particle Local CV History
        self.ax_particle_cv.set_title("Local CV of Tracked Particles")
        self.ax_particle_cv.set_xlabel("Particle Age (updates)")
        self.ax_particle_cv.set_ylabel("Local CV")
        self.ax_particle_cv.set_ylim(0, 1.0)
        self.ax_particle_cv.axhline(0.5, color='r', linestyle='--', label="Target CV=0.5")
        self.ax_particle_cv.grid(True)
        # Legend will be added dynamically

        # Plot 3: Total System Energy
        self.energy_line_plot, = self.ax_energy.plot([],[])
        self.ax_energy.set_title("Total System Energy")
        self.ax_energy.set_xlabel("Time (s)")
        self.ax_energy.set_ylabel("Energy (arb. units)")
        self.ax_energy.grid(True)

        # Plot 4: Born Rule Emergence
        alphas_ideal_deg = np.linspace(0, 180, 50)
        alphas_ideal_rad = np.deg2rad(alphas_ideal_deg)
        self.ax_born_rule.plot(alphas_ideal_deg, np.cos(alphas_ideal_rad/2)**2, 'k--', label="Ideal QM", lw=2)
        self.born_scatter_plot, = self.ax_born_rule.plot([], [], 'bo', alpha=0.7, label="Simulated P(0)")
        self.ax_born_rule.set_title("Born Rule Emergence P(0|α)")
        self.ax_born_rule.set_xlabel("Preparation Angle α (degrees from +Z)")
        self.ax_born_rule.set_ylabel("Prob P(Outcome +Z)")
        self.ax_born_rule.set_xlim(-5, 185); self.ax_born_rule.set_ylim(-0.05, 1.05)
        self.ax_born_rule.legend(fontsize='small'); self.ax_born_rule.grid(True)

        self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        self.fig.suptitle("Integrated PreQM Substrate, Particle & Measurement Visualizer", fontsize=14)


    def update_gui_and_sim_step(self):
        if not self.running:
            return

        self.sim.step() # Single step of the substrate simulation

        # Update Sum Phi Plot
        sum_phi_display = self.sim.sum_all_phi
        self.im_sum_phi.set_data(sum_phi_display)
        vlim = np.max(np.abs(sum_phi_display)) * 0.8 + 1e-5 # Slightly less than max for visibility
        self.im_sum_phi.set_clim(-vlim, vlim)

        # Update Particle Markers and Orientations
        active_p_x, active_p_y = [], []
        for arr in self.orientation_arrows_plots: arr.remove() # Clear old arrows
        self.orientation_arrows_plots = []
        
        # Determine which particle to use for status CV (e.g., oldest or first active)
        status_particle_cv = 0.0
        active_particles_for_cv_plot = []

        for pid, p_obj in self.sim.tracked_particles.items():
            if p_obj.is_active_this_cycle:
                active_p_x.append(p_obj.pos_yx[1]) 
                active_p_y.append(p_obj.pos_yx[0])
                active_particles_for_cv_plot.append(p_obj)
                if not status_particle_cv and p_obj.local_CV > 0: # Take first valid CV for status
                    status_particle_cv = p_obj.local_CV

                length = 4 # Arrow length in pixels
                # orientation_vec_xy is [x_comp, y_comp]
                dx = p_obj.orientation_vec_xy[0] * length 
                dy = -p_obj.orientation_vec_xy[1] * length # Invert dy for imshow 'lower' origin
                arrow = self.ax_sum_phi.arrow(p_obj.pos_yx[1], p_obj.pos_yx[0], dx, dy,
                                              head_width=1.0, head_length=1.5, fc='cyan', ec='cyan',
                                              length_includes_head=True, zorder=10)
                self.orientation_arrows_plots.append(arrow)
        self.particle_markers_plot.set_data(active_p_x, active_p_y)
        
        # Update Particle CV Plot
        self.ax_particle_cv.clear() # Redraw all lines each time
        self.ax_particle_cv.set_title("Local CV of Tracked Particles")
        self.ax_particle_cv.set_xlabel("Particle Age (updates since detection)")
        self.ax_particle_cv.set_ylabel("Local CV"); self.ax_particle_cv.set_ylim(0, 1.0)
        self.ax_particle_cv.axhline(0.5, color='r', linestyle='--', label="Target CV=0.5")
        self.ax_particle_cv.grid(True)
        plotted_legend_for_cv = False
        for p_obj in active_particles_for_cv_plot:
            if p_obj.local_substrate_activity_history and len(p_obj.local_substrate_activity_history)>1:
                # We need to re-calculate CV history for plotting from activity history
                cv_plot_history = []
                temp_hist = []
                for act_val in p_obj.local_substrate_activity_history:
                    temp_hist.append(act_val)
                    if len(temp_hist) >= 10: # Min window for CV for plot
                        seg = np.array(temp_hist)
                        if np.std(seg)>1e-7 and np.mean(seg)!=0:
                            try:
                                an_s = hilbert(seg-np.mean(seg)); env = np.abs(an_s)
                                if np.mean(env)>1e-7: cv_plot_history.append(np.clip(np.std(env)/np.mean(env),0,2))
                                else: cv_plot_history.append(0)
                            except: cv_plot_history.append(0.1)
                        else: cv_plot_history.append(0)
                    else: cv_plot_history.append(0.1)
                
                if len(cv_plot_history)>1:
                    self.ax_particle_cv.plot(range(p_obj.age_updates - len(cv_plot_history)+1, p_obj.age_updates+1), 
                                            cv_plot_history, label=f"P{p_obj.id}", lw=1, alpha=0.9)
                    plotted_legend_for_cv = True
        if plotted_legend_for_cv: self.ax_particle_cv.legend(fontsize='x-small', loc='upper right')


        # Update Energy Plot
        if self.sim.total_energy_history:
            times, energies = zip(*self.sim.total_energy_history)
            self.energy_line_plot.set_data(times, energies)
            if times: self.ax_energy.set_xlim(max(0, times[0]-1), times[-1] + 1) 
            if energies:
                min_e, max_e = np.min(energies), np.max(energies)
                if max_e > min_e : self.ax_energy.set_ylim(min_e - 0.1*abs(min_e) if min_e!=0 else -0.1, 
                                                           max_e + 0.1*abs(max_e) if max_e!=0 else 0.1)
                else: self.ax_energy.set_ylim(min_e - 0.1, max_e + 0.1)

        self.canvas.draw_idle()
        num_active_particles = len([p for pid, p in self.sim.tracked_particles.items() if p.is_active_this_cycle])
        self.status_var.set(f"Time: {self.sim.t:.2f} | Step: {self.sim.step_count} | Particles: {num_active_particles} | Sel.P CV: {status_particle_cv:.3f}")

        if self.running:
            self.animation_job = self.root.after(30, self.update_gui_and_sim_step) # GUI update interval

    def start_stop_sim(self): # Renamed
        self.running = not self.running
        self.start_stop_button.config(text="Pause Sim" if self.running else "Resume Sim")
        if self.running:
            self.update_gui_and_sim_step()

    def reset_sim_gui(self): # Renamed
        was_running = self.running
        self.running = False
        if self.animation_job: self.root.after_cancel(self.animation_job); self.animation_job = None
        
        self.sim = QuantumSeaAndIslandsSimulator(initialize_with_lumps=True)
        self.measurement_results_history = []
        self.clear_born_plot_gui()
        
        self.ax_particle_cv.clear()
        self.ax_particle_cv.set_title("Local CV of Tracked Particles"); self.ax_particle_cv.set_xlabel("Particle Age (updates)")
        self.ax_particle_cv.set_ylabel("Local CV"); self.ax_particle_cv.set_ylim(0, 1.0)
        self.ax_particle_cv.axhline(0.5, color='r', linestyle='--', label="Target CV=0.5"); self.ax_particle_cv.legend(fontsize='small')
        self.ax_particle_cv.grid(True)
        self.particle_cv_lines = {}

        if hasattr(self, 'energy_line_plot'): self.energy_line_plot.set_data([],[])
        self.ax_energy.set_xlim(0,1); self.ax_energy.set_ylim(0,1) # Reset energy plot limits

        print("Simulation Reset.")
        self.status_var.set("Status: Reset. Press Start/Resume.")
        self.start_stop_button.config(text="Start Sim" if not was_running else "Pause Sim") # Keep consistent
        if was_running: self.start_stop_sim() # Restart if it was running


    def run_born_measurements_gui(self): # Renamed
        active_particles = [p for p_id, p in self.sim.tracked_particles.items() if p.is_active_this_cycle]
        if not active_particles:
            print("No active particle found to perform measurements on."); return
        
        selected_particle = active_particles[0] # Use the first active one
        # Or, implement a way to select a particle, e.g., by ID or clicking
        self.selected_particle_id_for_measurement = selected_particle.id

        alpha_deg = self.alpha_slider.get()
        prob_0_val = self.measurement_model.perform_measurement_trials(selected_particle, alpha_deg)

        if not np.isnan(prob_0_val):
            self.measurement_results_history.append((alpha_deg, prob_0_val))
            self.measurement_results_history.sort(key=lambda x: x[0])
            self._update_born_plot_data_gui() # Renamed
            print(f"  P(0|α={alpha_deg:.1f}°) = {prob_0_val:.3f} (using particle {selected_particle.id} CV={selected_particle.local_CV:.3f})")
        else:
            print(f"Measurement failed for particle {selected_particle.id} at α={alpha_deg:.1f}°")


    def _update_born_plot_data_gui(self): # Renamed
        if self.measurement_results_history:
            alphas, probs = zip(*self.measurement_results_history)
            self.born_scatter_plot.set_data(alphas, probs)
        else:
            self.born_scatter_plot.set_data([],[])
        self.canvas.draw_idle()

    def clear_born_plot_gui(self): # Renamed
        self.measurement_results_history = []
        self._update_born_plot_data_gui()

    def on_close(self):
        print("Closing GUI..."); self.running = False
        if self.animation_job: self.root.after_cancel(self.animation_job)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = QuantumIslandGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
