import threading, time, queue
import numpy as np
from scipy.ndimage import convolve, label, binary_erosion, binary_dilation, gaussian_filter
from skimage.measure import marching_cubes
from collections import deque
import colorsys
import tkinter as tk # Keep for potential future dialogs, though Ursina has its own UI
from tkinter import ttk, messagebox # For potential dialogs
import matplotlib.pyplot as plt # Keep for potential debug plots, not main viz

from ursina import (
    Ursina, window, color, Entity, Mesh, EditorCamera, application,
    camera, Slider as UrsinaSlider, Text, Button as UrsinaButton, # Renamed Slider and Button
    destroy, mouse, Vec3, raycast, invoke
)
from ursina.shaders import lit_with_shadows_shader
from ursina.lights import DirectionalLight, AmbientLight

# --- FractalAgent class (enhanced with better coloring like best.py) ---
class FractalAgent:
    def __init__(self, id, mask, position, volume, age=0):
        self.id = id
        self.mask = mask.copy()
        self.position = position
        self.volume = volume
        self.age = age
        
        # Using the exact same coloring approach as best.py - direct from Ursina's random_color
        self.color = color.random_color()
        self.sub_sim = None

    def update(self, new_mask=None, new_position=None):
        if new_mask is not None: self.mask = new_mask.copy()
        if new_position is not None: self.position = new_position
        self.age += 1
        
    # Simplified versions of the sub-simulation methods for compatibility
    def promote_to_sub_simulation(self, parent_grid, local_N=32, min_volume=25):
        """Placeholder for sub-simulation promotion (not fully implemented yet)"""
        if self.age >= 20 and self.volume > min_volume and self.sub_sim is None:
            print(f"Promoting agent {self.id} to a sub-simulation!")
            # In a full implementation, we would create a local simulation here
            return True
        return False
        
    def step_sub_simulation(self, parent_grid, coupling=0.1):
        """Placeholder for sub-simulation stepping (not fully implemented yet)"""
        if self.sub_sim is not None:
            # In a full implementation, we would step the sub-simulation here
            pass

# ── Polyrhythmic Sea Solver (Modified MiniWoW) ───────────────────
class PolyrhythmicSea:
    def __init__(self, N=64, dt=0.05, # Reduced DT for stability
                 num_initial_fields=50, # Default number of phi fields
                 track_fractals=True):
        
        self.N = N
        self.dt = dt
        self.track_fractals = track_fractals
        self.lock = threading.Lock()

        # --- Core Substrate Parameters (Tunable) ---
        self.num_fields = num_initial_fields
        self.polyrhythm_coupling = 0.1
        self.nonlinearity_A = 1.0  # Same as pot_lin in best.py
        self.nonlinearity_B = 1.0  # Same as pot_cub in best.py  
        self.damping_factor = 0.005 # Same as damp in best.py
        self.tension = 5.0  # Adding tension param from best.py for consistency
        
        # Frequency range for oscillating fields
        self.base_frequencies_min = 0.5
        self.base_frequencies_max = 2.5
        self.diffusion_coeffs_min = 0.05
        self.diffusion_coeffs_max = 0.1

        # Initialize energy tracking
        self.energy_history = deque(maxlen=600)
        self.total_energy = 0.0

        self._initialize_fields_and_params()

        # Fractal tracking fields - matching best.py naming
        self.next_agent_id = 1
        self.agents = {}
        self.fractal_mask = np.zeros((N, N, N), dtype=bool)  # Renamed to match best.py
        self.last_detection_time = 0
        
        # Same 3D Laplacian kernel as in best.py
        self.kern = np.zeros((3,3,3), np.float32)
        self.kern[1,1,1] = -6
        for dx,dy,dz in [(1,1,0),(1,1,2),(1,0,1),(1,2,1),(0,1,1),(2,1,1)]:
            self.kern[dx,dy,dz] = 1

    def _initialize_fields_and_params(self):
        """Initializes or re-initializes fields based on self.num_fields etc."""
        N = self.N
        # Initialize lists for phi fields and their velocities
        self.phi_fields = [(np.random.rand(N, N, N).astype(np.float32) - 0.5) * 0.5  # Increased initial amplitude
                           for _ in range(self.num_fields)]
        self.phi_o_fields = [np.copy(phi) for phi in self.phi_fields]  # Previous timestep values
        
        # Field-specific parameters
        self.base_frequencies = np.linspace(self.base_frequencies_min, self.base_frequencies_max, self.num_fields)
        self.diffusion_coeffs = np.linspace(self.diffusion_coeffs_max, self.diffusion_coeffs_min, self.num_fields)
        self.field_phases = np.random.uniform(0, 2 * np.pi, self.num_fields)

        # Sum of all phi fields, for visualization and potentially for agent detection
        self.phi = np.zeros((N, N, N), dtype=np.float32)  # Main field (sum)
        self.phi_o = np.zeros((N, N, N), dtype=np.float32)  # Previous timestep
        self._update_summed_fields()

        # Reset tracking
        self.t_sim = 0.0  # Internal simulation time
        self.agents = {}
        self.next_agent_id = 1
        self.fractal_mask = np.zeros((N,N,N), dtype=bool)
        
        # Clear energy history and reset total energy
        if hasattr(self, 'energy_history'):
            self.energy_history.clear()
        else:
            self.energy_history = deque(maxlen=600)
        self.total_energy = 0.0

    def _update_summed_fields(self):
        """Update the main summed field from individual phi fields"""
        self.phi = np.sum(self.phi_fields, axis=0) / max(1, len(self.phi_fields))
        self.phi_o = np.sum(self.phi_o_fields, axis=0) / max(1, len(self.phi_o_fields))

    def field_energy(self):
        """Calculate the field energy distribution (similar to best.py)"""
        # Calculate the gradient components
        dx, dy, dz = np.gradient(self.phi)
        
        # Gradient energy term: 0.5 * |∇Ψ|²
        grad_energy = 0.5 * (dx**2 + dy**2 + dz**2)
        
        # Potential energy term: V(Ψ)
        # For the double-well potential: V(Ψ) = -pot_lin * Ψ + pot_cub * Ψ³
        potential_energy = -self.nonlinearity_A * self.phi + self.nonlinearity_B * self.phi**3
        potential_energy = potential_energy**2  # Ensure positive energy
        
        # Total energy = gradient energy + potential energy
        total_energy = grad_energy + potential_energy
        
        return total_energy

    def calculate_total_energy(self):
        """Calculate the total energy of the field (like in best.py)"""
        energy_density = self.field_energy()
        total = np.sum(energy_density)
        self.total_energy = total
        self.energy_history.append(total)
        return total

    def _potential_deriv(self, field_k):
        """Calculate the derivative of the potential function for a field"""
        # V'(phi) = -A*phi + B*phi^3
        return -self.nonlinearity_A * field_k + self.nonlinearity_B * (field_k**3)

    def resize_grid(self, new_N):
        """Resize the simulation grid"""
        with self.lock:
            self.N = new_N
            self._initialize_fields_and_params()  # Re-init all fields for new size
            self.fractal_mask = np.zeros((new_N, new_N, new_N), dtype=bool)
            print(f"Resized grid to {new_N}. Fields re-initialized.")

    def change_num_fields(self, new_num_fields):
        """Change the number of fields in the simulation"""
        with self.lock:
            self.num_fields = int(new_num_fields)
            self._initialize_fields_and_params()  # Re-init all fields
            print(f"Number of fields changed to {self.num_fields}. Fields re-initialized.")

    def reset(self):
        """Reset the simulation to initial state"""
        with self.lock:
            self._initialize_fields_and_params()
            print("Simulation reset")

    def step(self, n_steps=1):
        """Step the simulation forward by n_steps"""
        for _ in range(n_steps):
            with self.lock:  # Lock during the critical update section
                new_phi_list = []

                # Update phases for all fields
                self.field_phases += self.base_frequencies * self.dt
                self.field_phases %= (2 * np.pi)

                for k in range(self.num_fields):
                    phi_k = self.phi_fields[k]
                    phi_o_k = self.phi_o_fields[k]
                    
                    # Calculate velocity from current and previous positions
                    vel_k = phi_k - phi_o_k
                    
                    # Laplacian
                    lap_k = convolve(phi_k, self.kern, mode='wrap')
                    
                    # Non-linear potential term
                    potential_deriv_k = self._potential_deriv(phi_k)
                    
                    # Coupling between fields
                    coupling_denom = max(1, self.num_fields - 1)
                    other_fields_sum = (np.sum(self.phi_fields, axis=0) - phi_k)
                    coupling_force = self.polyrhythm_coupling * other_fields_sum / coupling_denom
                    
                    # Driving force from oscillator
                    driving_force_k = 0.005 * np.sin(self.field_phases[k])
                    
                    # Tension term similar to best.py
                    c2 = 1.0 / (1.0 + self.tension * phi_k**2 + 1e-6)
                    
                    # Combined acceleration
                    acc = (c2 * self.diffusion_coeffs[k] * lap_k - 
                          potential_deriv_k + 
                          coupling_force + 
                          driving_force_k)
                    
                    # Update using velocity Verlet integration (similar to best.py)
                    self.phi_o_fields[k] = phi_k.copy()
                    new_phi_k = phi_k + (1 - self.damping_factor * self.dt) * vel_k + self.dt**2 * acc
                    new_phi_list.append(new_phi_k)
                
                self.phi_fields = new_phi_list
                self._update_summed_fields()
                self.t_sim += self.dt

            # Pattern detection and energy calculation
            if self.track_fractals and int(self.t_sim / self.dt) % 10 == 0:  # Less frequent detection
                self.detect_stable_patterns()
                self.calculate_total_energy()

    def detect_stable_patterns(self, iso_threshold=1.0, min_volume=10, max_volume=None):
        """Identify stable patterns (isosurfaces) in the field - based on best.py"""
        if not self.track_fractals:
            self.agents = {}
            self.fractal_mask.fill(False)
            return
            
        # Only run detection periodically to save resources
        current_time = time.time()
        if current_time - self.last_detection_time < 0.5:  # Run detection every 0.5 seconds
            return
        self.last_detection_time = current_time
        
        # Use the summed field (self.phi) for pattern detection
        binary_mask = (self.phi > iso_threshold)
        
        # Clean up the mask - remove small holes and smooth edges
        binary_mask = binary_erosion(binary_mask, iterations=1)
        binary_mask = binary_dilation(binary_mask, iterations=1)
        
        # Label connected components
        labeled_mask, num_features = label(binary_mask)
        
        # Process each feature and track it
        active_agent_ids = set()
        for i in range(1, num_features + 1):
            component_mask = (labeled_mask == i)
            volume = np.sum(component_mask)
            
            # Skip components that are too small or too large
            if volume < min_volume:
                continue
            if max_volume is not None and volume > max_volume:
                continue
                
            # Get centroid (center of mass)
            coords = np.where(component_mask)
            position = (np.mean(coords[0]), np.mean(coords[1]), np.mean(coords[2]))
            
            # Try to match with existing agents based on position
            matched = False
            closest_agent_id = None
            min_distance = float('inf')
            
            for agent_id, agent in self.agents.items():
                dist = np.sqrt((agent.position[0] - position[0])**2 + 
                               (agent.position[1] - position[1])**2 + 
                               (agent.position[2] - position[2])**2)
                if dist < min_distance:
                    min_distance = dist
                    closest_agent_id = agent_id
            
            # If close enough to an existing agent, update it
            if closest_agent_id is not None and min_distance < 10:  # Adjust threshold as needed
                self.agents[closest_agent_id].update(component_mask, position)
                active_agent_ids.add(closest_agent_id)
                matched = True
            
            # Otherwise, create a new agent
            if not matched:
                new_id = self.next_agent_id
                self.next_agent_id += 1
                self.agents[new_id] = FractalAgent(new_id, component_mask, position, volume)
                active_agent_ids.add(new_id)
        
        # Remove agents that weren't matched in this frame
        to_remove = []
        for agent_id in self.agents:
            if agent_id not in active_agent_ids:
                to_remove.append(agent_id)
        
        for agent_id in to_remove:
            # Only remove if it's been missing for a few frames
            self.agents[agent_id].age -= 2  # Decrease age faster when not detected
            if self.agents[agent_id].age <= 0:
                del self.agents[agent_id]
            
        # Update the global fractal mask for visualization
        self.fractal_mask = np.zeros_like(binary_mask)
        for agent in self.agents.values():
            self.fractal_mask = np.logical_or(self.fractal_mask, agent.mask)
        
        # Promote stable agents to have their own simulations (placeholder)
        for agent in list(self.agents.values()):
            if agent.age > 20 and agent.volume > 25 and agent.sub_sim is None:
                agent.promote_to_sub_simulation(self.phi)

    def apply_poke(self, x, y, z, radius=3, amplitude=1.0, sigma=2.0):
        """Apply a gaussian 'poke' to the field at the specified position"""
        with self.lock:
            N = self.N
            # Ensure coordinates are valid
            x, y, z = int(x), int(y), int(z)
            if x < 0 or x >= N or y < 0 or y >= N or z < 0 or z >= N:
                return False
                
            # Define box bounds
            x_min, x_max = max(0, x-radius), min(N-1, x+radius)
            y_min, y_max = max(0, y-radius), min(N-1, y+radius)
            z_min, z_max = max(0, z-radius), min(N-1, z+radius)
            
            # Create coordinate grids for the box
            xx, yy, zz = np.meshgrid(
                np.arange(x_min, x_max+1),
                np.arange(y_min, y_max+1),
                np.arange(z_min, z_max+1),
                indexing='ij'
            )
            
            # Calculate squared distance from center
            dist_sq = (xx-x)**2 + (yy-y)**2 + (zz-z)**2
            
            # Apply gaussian pulse to all fields
            for k in range(self.num_fields):
                self.phi_fields[k][x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] += \
                    amplitude * np.exp(-dist_sq / (2 * sigma**2))
                
            return True


# --- Simulation thread and queue setup ---
paused = False
current_grid_size = 32  # Start with a medium grid for multi-field
enable_fractal_tracking = True 
iso_val = 1.0  # Initial isosurface threshold (same as best.py)
visualization_mode = 'field'  # 'field', 'agents', 'both' (matching best.py)

sim = PolyrhythmicSea(N=current_grid_size, track_fractals=enable_fractal_tracking)
field_q = queue.Queue(maxsize=2)
stop_evt = threading.Event()

def sim_worker(sim_instance, q_instance, stop_event_instance):
    """Worker thread to run simulation steps"""
    while not stop_event_instance.is_set():
        if not paused:
            sim_instance.step(1)
            with sim_instance.lock:
                if not q_instance.full():
                    # Send phi, agents, and fractal mask
                    q_instance.put((sim_instance.phi.copy(), 
                                   dict(sim_instance.agents),
                                   sim_instance.fractal_mask.copy()))
        time.sleep(0.01)  # Control loop speed

sim_thread = threading.Thread(target=sim_worker, args=(sim, field_q, stop_evt), daemon=True)
sim_thread.start()

# --- Ursina App Setup ---
app = Ursina(fullscreen=True, development_mode=False)
window.color = color.black  # Same as best.py
window.title = 'Enhanced Polyrhythmic Sea Explorer'
window.fps_counter.enabled = True

editor_cam = EditorCamera()
base_camera_speed = 5.0

# Lighting setup similar to best.py
sun = DirectionalLight(rotation=(45, -45, 45), color=color.white, shadows=True)
AmbientLight(color=color.rgba(0.5, 0.5, 0.7, 0.3))  # Increased ambient light

# Set up entities for visualization
container = Entity()
surface = Entity(parent=container, double_sided=False, shader=lit_with_shadows_shader)
agents_container = Entity(parent=container)
agent_entities = {}

# Add ground plane like in best.py
ground = Entity(model='plane', scale=100, y=-15,
                color=color.dark_gray, texture='white_cube', texture_scale=(100, 100))

# Click plane for mouse interactions
click_plane = Entity(
    model='plane',
    scale=(100, 100, 1),
    position=(0, 0, 0),
    rotation=(90, 0, 0),
    collider='box',
    visible=False
)

# Status text for fractal information
fractal_info_text = Text(text="No fractals detected yet", position=(0, 0.45), origin=(0, 0), scale=0.8)

# Energy display
energy_text = Text(text="Energy: 0.00", position=(-0.7, 0.45), origin=(-0.5, 0), scale=0.8)
energy_history_text = Text(text="", position=(-0.7, 0.4), origin=(-0.5, 0), scale=0.6, color=color.yellow)

# Poke indicator
poke_indicator = Entity(model='sphere', scale=0.1, color=color.yellow, visible=False, 
                        always_on_top=True, shader=lit_with_shadows_shader)

# Interaction parameters
time_val = 0.0
poke_radius = 3
poke_amplitude = 1.0

def update_color():
    """Get dynamic color based on time (exactly like best.py)"""
    hue = (time_val * 0.1) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
    # Use float values directly - don't multiply by 255!
    return color.rgb(r, g, b)

def update_mesh(phi_data):
    try:
        v, f, n, _ = marching_cubes(phi_data, level=iso_val)
        if v.size == 0:
            surface.visible = False
            return
            
        v = v - v.mean(0)  # Center mesh
        
        surface.model = Mesh(vertices=v.tolist(),
                            triangles=f.flatten().tolist(),
                            normals=n.tolist(),
                            mode='triangle')
                            
        # Use the same color updating as best.py
        surface.color = update_color()
        surface.visible = visualization_mode in ['field', 'both']
    except Exception as e:
        print('marching-cubes error:', e)
        surface.visible = False

def update_agent_visualizations(agents):
    """Update visualization of detected stable patterns"""
    global agent_entities, agents_container, fractal_info_text
    
    # Update info text
    if not agents:
        fractal_info_text.text = "No stable patterns detected"
    else:
        fractal_info_text.text = f"{len(agents)} stable patterns detected | {sum(1 for a in agents.values() if a.sub_sim)} with sub-simulations"
    
    # Remove entities for agents that no longer exist
    to_remove = []
    for agent_id in agent_entities:
        if agent_id not in agents:
            destroy(agent_entities[agent_id])
            to_remove.append(agent_id)
    
    for agent_id in to_remove:
        del agent_entities[agent_id]
    
    # Update or create entities for current agents
    for agent_id, agent in agents.items():
        # Skip agents that are too young to visualize
        if agent.age < 5:
            continue
            
        if agent_id in agent_entities:
            # Just update color/scale for existing entities
            entity = agent_entities[agent_id]
            scale_factor = min(1.0, agent.age / 20)  # Grow to full size over time
            entity.scale = 5 * scale_factor
            
            # Highlight agents with sub-simulations
            if agent.sub_sim is not None:
                entity.color = color.yellow
            else:
                entity.color = agent.color
        else:
            # Create new visualization for this agent
            entity = Entity(
                parent=agents_container,
                model='sphere',
                position=(-agent.position[1] + current_grid_size/2, 
                          -agent.position[2] + current_grid_size/2, 
                          -agent.position[0] + current_grid_size/2),  # Adjust for coordinate system
                scale=2,
                color=agent.color,
                shader=lit_with_shadows_shader
            )
            agent_entities[agent_id] = entity
    
    # Set visibility based on visualization mode
    agents_container.enabled = visualization_mode in ['agents', 'both']

def update_energy_display():
    """Update the energy display and energy history text"""
    # Update the energy text
    energy_text.text = f"Energy: {sim.total_energy:.2e}"
    
    # Update the energy history text
    if len(sim.energy_history) > 1:
        last_energy = sim.energy_history[-1]
        prev_energy = sim.energy_history[-2]
        if last_energy > prev_energy:
            trend = "↑"  # Rising energy
        elif last_energy < prev_energy:
            trend = "↓"  # Falling energy
        else:
            trend = "→"  # Stable energy
            
        # Calculate percentage change
        if prev_energy != 0:
            percent_change = (last_energy - prev_energy) / prev_energy * 100
            energy_history_text.text = f"Trend: {trend} {abs(percent_change):.2f}%"
        else:
            energy_history_text.text = f"Trend: {trend}"
    else:
        energy_history_text.text = "Monitoring energy..."

def adapt_scale_for_grid_size():
    """Adjust scale-dependent elements for the current grid size"""
    global current_grid_size, iso_val
    
    # Fix: Try to use move_speed if it exists, otherwise do nothing for the camera
    try:
        editor_cam.move_speed = base_camera_speed * (current_grid_size / 64)
    except AttributeError:
        # If move_speed doesn't exist, try other possible attribute names
        for attr_name in ['movement_speed', 'camera_speed', 'speed']:
            if hasattr(editor_cam, attr_name):
                setattr(editor_cam, attr_name, base_camera_speed * (current_grid_size / 64))
                break
        # If none of the above work, print a message
        else:
            print("Warning: Could not find camera speed attribute to adjust")
    
    # Adjust ground texture scale
    ground.texture_scale = (current_grid_size, current_grid_size)
    
    # Adjust isosurface threshold
    default_iso = 1.0
    iso_val = default_iso * (64 / current_grid_size)

def handle_poke():
    """Handle poking the simulation field"""
    if mouse.left:  # Left mouse button pressed
        # Perform raycast to get position
        hit_info = raycast(camera.world_position, camera.forward, distance=100)
        
        if hit_info.hit:
            # Convert world position to grid coordinates
            world_pos = hit_info.world_point
            N = sim.N
            
            # Convert from world to grid coordinates (taking into account coordinate system differences)
            x = int(N/2 - world_pos.z)
            y = int(N/2 - world_pos.x)
            z = int(N/2 - world_pos.y)
            
            # Check if coordinates are valid
            if 0 <= x < N and 0 <= y < N and 0 <= z < N:
                # Apply poke
                if sim.apply_poke(x, y, z, radius=poke_radius, amplitude=poke_amplitude):
                    # Show poke indicator
                    poke_indicator.position = hit_info.world_point
                    poke_indicator.visible = True
                    
                    # Hide indicator after a short time
                    invoke(setattr, poke_indicator, 'visible', False, delay=0.2)

# --- Main update loop ---
def update():
    global time_val
    
    if paused:
        return
        
    time_val += time.dt
    container.rotation_y += time.dt * 5  # Same rotation as best.py
    
    # Handle poking interaction
    handle_poke()

    # Process visualization data from simulation thread
    try:
        while True:
            data = field_q.get_nowait()
            if isinstance(data, tuple) and len(data) >= 2:
                phi_data, agents_data = data[0], data[1]
                update_mesh(phi_data)
                update_agent_visualizations(agents_data)
                update_energy_display()
            else:
                # Handle old format if needed
                update_mesh(data)
    except queue.Empty:
        pass

# --- Input handling ---
def input(key):
    global iso_val, paused, visualization_mode, enable_fractal_tracking
    
    if key == 'left arrow':
        iso_val = max(-2.0, iso_val - 0.05)
        print(f"Isosurface threshold: {iso_val:.2f}")
    elif key == 'right arrow':
        iso_val = min(2.0, iso_val + 0.05)
        print(f"Isosurface threshold: {iso_val:.2f}")
    elif key == 'p':
        paused = not paused
        print('Paused' if paused else 'Resumed')
    elif key == 'r':
        sim.reset()
        print('Simulation reset')
    elif key == 'v':  # Toggle visualization mode like in best.py
        if visualization_mode == 'field':
            visualization_mode = 'agents'
        elif visualization_mode == 'agents':
            visualization_mode = 'both'
        else:
            visualization_mode = 'field'
        print(f'Visualization mode: {visualization_mode}')
        surface.visible = visualization_mode in ['field', 'both']
        agents_container.enabled = visualization_mode in ['agents', 'both']
    elif key == 't':  # Toggle fractal tracking
        enable_fractal_tracking = not enable_fractal_tracking
        sim.track_fractals = enable_fractal_tracking
        print(f"Fractal tracking: {'enabled' if enable_fractal_tracking else 'disabled'}")
        if not enable_fractal_tracking:
            for ent in list(agent_entities.values()):
                destroy(ent)
            agent_entities.clear()
    elif key == 'escape':
        stop_evt.set()
        application.quit()
    elif key == 'f':
        window.fullscreen = not window.fullscreen

# --- UI elements ---
def add_slider(label, attr, rng, y):
    """Add a slider for adjusting simulation parameters (like best.py)"""
    txt = Text(text=f'{label}: {getattr(sim, attr):.3f}',
              x=-0.83, y=y + 0.04, parent=camera.ui, scale=0.75)
    sld = UrsinaSlider(min=rng[0], max=rng[1], default=getattr(sim, attr),
                step=(rng[1] - rng[0]) / 200, x=-0.85, y=y, scale=0.3,
                parent=camera.ui)

    def changed():
        val = sld.value
        setattr(sim, attr, val)
        txt.text = f'{label}: {val:.3f}'
    sld.on_value_changed = changed

# --- Topology and grid size selection ---
def add_topology_selector():
    """Add UI elements for selecting topology (like best.py)"""
    # Create a topology selection label
    Text(text='Topology:', x=0.65, y=0.4, parent=camera.ui, scale=0.75)
    
    topologies = ['box', 'sphere', 'torus', 'wave', 'random']
    buttons = []
    
    for i, topo in enumerate(topologies):
        btn = UrsinaButton(text=topo.capitalize(), scale=(0.15, 0.05), x=0.7, y=0.35 - i*0.06, 
                    parent=camera.ui, color=color.light_gray)
        buttons.append(btn)
        
        def make_on_click(topology):
            def on_click():
                # Reset all button colors
                for b in buttons:
                    b.color = color.light_gray
                # Highlight the selected button
                buttons[topologies.index(topology)].color = color.azure
                # Change the topology - reinitialize fields with a pattern
                sim.reset()  # This effectively changes the topology
                print(f"Changed topology to {topology}")
            return on_click
            
        btn.on_click = make_on_click(topo)
    
    # Add custom shape button
    custom_btn = UrsinaButton(text='Custom Shape', scale=(0.15, 0.05), x=0.7, y=0.35 - len(topologies)*0.06, 
                       parent=camera.ui, color=color.orange)
    
    # Add grid size selection
    Text(text='Grid Size:', x=0.65, y=0.0, parent=camera.ui, scale=0.75)
    Text(text='Warning: Large sizes may slow down\nyour system!', 
         x=0.65, y=-0.05, parent=camera.ui, scale=0.5, color=color.red)
    
    grid_sizes = [32, 64, 128, 256]
    grid_buttons = []
    
    for i, size in enumerate(grid_sizes):
        btn = UrsinaButton(text=str(size), scale=(0.15, 0.05), x=0.7, y=-0.1 - i*0.06, 
                    parent=camera.ui, color=color.light_gray)
        grid_buttons.append(btn)
        
        def make_on_click(grid_size):
            def on_click():
                global current_grid_size, paused
                # Don't do anything if already at this size
                if current_grid_size == grid_size:
                    return
                    
                # Reset all button colors
                for b in grid_buttons:
                    b.color = color.light_gray
                # Highlight the selected button
                grid_buttons[grid_sizes.index(grid_size)].color = color.azure
                
                # Change grid size - this will disrupt the simulation temporarily
                paused_state = paused
                if not paused:
                    # Pause simulation while changing grid
                    paused = True
                    time.sleep(0.1)  # Give time for thread to pause
                
                # Update the simulation grid size
                print(f"Changing grid size to {grid_size}...")
                current_grid_size = grid_size
                sim.resize_grid(grid_size)
                
                # Clear old agent entities when changing grid size
                for ent in list(agent_entities.values()):
                    destroy(ent)
                agent_entities.clear()
                
                # Adjust camera speed, ground texture scale based on new grid size
                adapt_scale_for_grid_size()
                
                # Resume if it was running
                if not paused_state:
                    paused = False
                
            return on_click
            
        btn.on_click = make_on_click(size)
    
    # Set the initial grid size button to be highlighted
    grid_buttons[grid_sizes.index(current_grid_size) if current_grid_size in grid_sizes else 0].color = color.azure

# --- Tracking toggle ---
def add_tracking_toggle():
    """Add button to enable/disable fractal tracking"""
    Text(text='Fractal Tracking:', x=0.65, y=-0.45, parent=camera.ui, scale=0.75)
    
    tracking_btn = UrsinaButton(
        text='Enabled' if enable_fractal_tracking else 'Disabled',
        scale=(0.15, 0.05), 
        x=0.7, y=-0.5,
        parent=camera.ui,
        color=color.green if enable_fractal_tracking else color.red
    )
    
    def toggle_tracking():
        global enable_fractal_tracking
        enable_fractal_tracking = not enable_fractal_tracking
        tracking_btn.text = 'Enabled' if enable_fractal_tracking else 'Disabled'
        tracking_btn.color = color.green if enable_fractal_tracking else color.red
        sim.track_fractals = enable_fractal_tracking
        print(f"Fractal tracking: {'enabled' if enable_fractal_tracking else 'disabled'}")
        
    tracking_btn.on_click = toggle_tracking

# --- Poke controls ---
def add_poke_controls():
    """Add controls for the poke tool"""
    Text(text='Poke Settings:', x=0.65, y=-0.6, parent=camera.ui, scale=0.75)
    
    # Radius slider
    txt_radius = Text(text=f'Radius: {poke_radius}', x=0.65, y=-0.65, parent=camera.ui, scale=0.6)
    sld_radius = UrsinaSlider(min=1, max=10, default=poke_radius,
                       x=0.7, y=-0.7, scale=0.15,
                       parent=camera.ui)
    
    def radius_changed():
        global poke_radius
        poke_radius = int(sld_radius.value)
        txt_radius.text = f'Radius: {poke_radius}'
    sld_radius.on_value_changed = radius_changed
    
    # Amplitude slider
    txt_amplitude = Text(text=f'Amplitude: {poke_amplitude:.1f}', x=0.65, y=-0.75, parent=camera.ui, scale=0.6)
    sld_amplitude = UrsinaSlider(min=0.1, max=5.0, default=poke_amplitude,
                         x=0.7, y=-0.8, scale=0.15,
                         parent=camera.ui)
    
    def amplitude_changed():
        global poke_amplitude
        poke_amplitude = sld_amplitude.value
        txt_amplitude.text = f'Amplitude: {poke_amplitude:.1f}'
    sld_amplitude.on_value_changed = amplitude_changed

# Add the UI elements
add_slider('dt', 'dt', (0.01, 0.2), 0.35)
add_slider('damping', 'damping_factor', (0.0, 0.05), 0.25)
add_slider('tension', 'tension', (0.0, 20.0), 0.15)
add_slider('nonlinearity_A', 'nonlinearity_A', (0.0, 2.0), 0.05)
add_slider('nonlinearity_B', 'nonlinearity_B', (0.0, 1.0), -0.05)
add_slider('coupling', 'polyrhythm_coupling', (0.0, 0.5), -0.15)
add_slider('num_fields', 'num_fields', (10, 200), -0.25)

add_topology_selector()
add_tracking_toggle()
add_poke_controls()

# Add instructions text
Text('WASD+RMB fly | ←/→ iso | P pause | R reset | V toggle view | ESC quit',
    y=-0.45, x=0, origin=(0, 0),
    background=True, background_color=color.rgba(0, 0, 0, 128),
    parent=camera.ui)

# Adjust scale-dependent elements for the initial grid size
adapt_scale_for_grid_size()

# Initial "poke" to get things started

sim.apply_poke(current_grid_size//2, current_grid_size//2, current_grid_size//2, radius=5, amplitude=2.0)

app.run()