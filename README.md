# Polyrhythmic Sea Simulators (PreQM Exploration)

This repository contains Python scripts that simulate a "Polyrhythmic Sea" – a concept related to a Pre-Quantum Mechanics (PreQM)
theory where quantum-like phenomena might emerge from an underlying classical, deterministic, but highly active and structured substrate.

The core idea is that a substrate composed of many coupled, non-linear fields can self-organize into complex patterns and exhibit
statistical properties (like 1/f noise and a specific Coefficient of Variation) that could be foundational to quantum measurement statistics.

## Simulators Included:

1.  **`polyrhythmic_sea_3d_ursina.py` (or `3d.py` as per your file)**
    *   A 3D simulation using the Ursina game engine for visualization.
    *   Models a substrate with multiple (`num_fields`) coupled 3D scalar fields (`phi_k`).
    *   Dynamics include diffusion, damping, non-linear self-interaction (e.g., `V'(phi) = -A*phi + B*phi^3`), inter-field polyrhythmic coupling,
      a driving force, and a field-dependent wave speed (FLSDAFL analogue).
    *   Visualizes the sum of these fields as an isosurface using Marching Cubes, often producing intricate, geometric, "crystal-like" patterns.
    *   Includes interactive controls for parameters, camera, and poking the simulation.
    *   Features "Fractal Tracking" to identify and visualize persistent localized structures (agents/particles).

2.  **`polyrhythmic_sea_2d_matplotlib.py` (or the 2D code you provided)**
    *   A 2D version of the simulator using Matplotlib for animation.
    *   Implements similar core physics with multiple coupled 2D scalar fields.
    *   Faster to run and iterate on parameters due to 2D nature.
    *   Visualizes the summed field and can detect simple 2D patterns.
    *   Allows for interactive "poking" of the field via mouse clicks.

## Theoretical Context (PreQM Ideas):

These simulators are tools to explore hypotheses such as:
*   **Emergence of Order:** How complex, geometric, and potentially fractal patterns arise from simpler non-linear rules.
*   **Substrate as a Dynamic Medium:** The "vacuum" is not empty but an active, polyrhythmic "sea."
*   **Particles as Excitations:** Stable "lumps" or "patterns" are excitations of this sea.
*   **1/f Noise and CV≈0.5:** The potential for such a substrate to endogenously generate statistical noise with a 1/f power
   spectrum and an amplitude envelope Coefficient of Variation (CV) around 0.5. This specific statistical environment is
   theorized to be crucial for the emergence of quantum measurement probabilities (Born rule).
*   **FLSDAFL:** The idea that "Field Lumps Slow Down Around Frequency Lumps" provides an analogue for gravity/time dilation.

## How to Run:

**Prerequisites:**
*   Python 3.x
*   See `requirements.txt` for necessary libraries.

**Running the 3D Ursina Simulator:**

pip install -r requirements_ursina.txt # (If you make a separate one for Ursina)

python 3d.py

Use code with caution.

Controls: Displayed on screen (WASD+RMB to fly, P to pause, R to reset, etc.). Sliders and buttons in the UI control simulation parameters.

Running the 2D Matplotlib Simulator:

pip install -r requirements_matplotlib.txt # (If you make a separate one for Matplotlib)

python 2d.py

Controls: The Matplotlib window will appear. You can often click on the plot to "poke" the simulation.

# Purpose of Exploration:

These simulators are for:

Visualizing emergent complexity from coupled non-linear fields.
Exploring parameter spaces to find regimes that produce interesting structures or statistical properties (like those hypothesized for PreQM).
Developing intuition about how particle-like entities and complex field dynamics might arise from a "simpler" underlying substrate.
Providing a computational laboratory for testing ideas related to a deterministic foundation for quantum phenomena.

Note: These are research tools and conceptual demonstrators. They are not intended to be fully optimized or bug-free production code but rather platforms for exploration and discovery.
