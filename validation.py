import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm.notebook import tqdm

from utils import load_tomogram, preprocess_tomogram, load_particle_coords, generate_density_map
from visualization import visualize_tomogram_with_particles, visualize_density_map_with_particles
from particle_detection import UNet3D

def prepare_validation_tomograms(train_dir, model_dir, visualization_dir, device):
    """
    Prepare a few validation tomograms and their particle annotations for visualization

    Parameters:
    train_dir (str): Path to training data directory
    model_dir (str): Path to model directory
    visualization_dir (str): Path to save visualizations
    device: PyTorch device
    """
    # Define particle types and their properties
    particle_types = {
        'apo-ferritin': {'difficulty': 'easy', 'weight': 1, 'radius': 60, 'color': 'red'},
        'beta-amylase': {'difficulty': 'impossible', 'weight': 0, 'radius': 45, 'color': 'yellow'},
        'beta-galactosidase': {'difficulty': 'hard', 'weight': 2, 'radius': 80, 'color': 'green'},
        'ribosome': {'difficulty': 'easy', 'weight': 1, 'radius': 100, 'color': 'blue'},
        'thyroglobulin': {'difficulty': 'hard', 'weight': 2, 'radius': 85, 'color': 'purple'},
        'virus-like-particle': {'difficulty': 'easy', 'weight': 1, 'radius': 120, 'color': 'orange'}
    }

    # Get train experiments
    import glob
    train_experiments = [os.path.basename(p) for p in glob.glob(os.path.join(train_dir, 'static/ExperimentRuns/*'))]

    if not train_experiments:
        print("No training experiments found.")
        return

    # Use the first 3 experiments for visualization (or fewer if less are available)
    viz_experiments = train_experiments[:min(3, len(train_experiments))]
    print(f"Using {len(viz_experiments)} experiments for visualization: {viz_experiments}")

    # For each experiment, load the tomogram and particle annotations
    for experiment in viz_experiments:
        print(f"\nProcessing experiment: {experiment}")

        # Load tomogram
        zarr_path = os.path.join(train_dir, 'static/ExperimentRuns', experiment, 'VoxelSpacing10.000/denoised.zarr')

        if not os.path.exists(zarr_path):
            print(f"Tomogram not found for experiment {experiment}")
            continue

        tomo_data = load_tomogram(zarr_path)
        if tomo_data is None:
            print(f"Failed to load tomogram for experiment {experiment}")
            continue

        tomo_data = preprocess_tomogram(tomo_data)

        # Load particle annotations
        ground_truth_coords = {}

        for p_type in particle_types.keys():
            json_path = os.path.join(train_dir, 'overlay/ExperimentRuns', experiment, 'Picks', f"{p_type}.json")

            if not os.path.exists(json_path):
                print(f"No annotations found for {p_type} in experiment {experiment}")
                continue

            # Load coordinates from JSON
            coords = load_particle_coords(json_path)

            if coords:
                ground_truth_coords[p_type] = coords
                print(f"Loaded {len(coords)} {p_type} coordinates")

        # Visualize ground truth
        visualize_ground_truth(tomo_data, ground_truth_coords, experiment, particle_types, visualization_dir)

        # Now visualize model predictions on the same tomogram
        visualize_model_predictions_on_tomogram(tomo_data, experiment, ground_truth_coords,
                                               particle_types, model_dir, visualization_dir, device)

def visualize_ground_truth(tomo_data, ground_truth_coords, experiment, particle_types, visualization_dir):
    """
    Visualize ground truth particles on the tomogram

    Parameters:
    tomo_data (numpy.ndarray): 3D tomogram data
    ground_truth_coords (dict): Dictionary mapping particle types to coordinates
    experiment (str): Experiment name
    particle_types (dict): Dictionary of particle types and their properties
    visualization_dir (str): Path to save visualizations
    """
    # Get tomogram dimensions
    depth, height, width = tomo_data.shape

    # Choose slices for visualization
    slices = [depth // 4, depth // 2, 3 * depth // 4]

    # Create figure with subplots for different slices
    fig, axes = plt.subplots(1, len(slices), figsize=(6 * len(slices), 6))
    if len(slices) == 1:
        axes = [axes]

    # For each slice
    for i, slice_idx in enumerate(slices):
        # Show the tomogram slice
        axes[i].imshow(tomo_data[slice_idx], cmap='gray')
        axes[i].set_title(f'Ground Truth - Z-Slice {slice_idx}/{depth}')

        # Get slice range (particles near this slice)
        slice_range = 10  # Consider particles within ±10 slices
        z_min = (slice_idx - slice_range) * 10.0  # Convert to physical coordinates
        z_max = (slice_idx + slice_range) * 10.0

        # Add circles for each particle type
        for p_type, coords in ground_truth_coords.items():
            # Skip if particle type not in our dictionary
            if p_type not in particle_types:
                continue

            # Get color and radius for this particle type
            color = particle_types[p_type]['color']
            radius = particle_types[p_type]['radius'] * 0.1  # Scale down for visualization

            # Count particles in this slice
            slice_particles = [(x, y, z) for x, y, z in coords if z_min <= z <= z_max]
            n_particles = len(slice_particles)

            # Skip if no particles of this type in this slice
            if n_particles == 0:
                continue

            # Add to legend
            axes[i].plot([], [], 'o', color=color, label=f'{p_type} ({n_particles})')

            # Add circle for each particle
            for x, y, z in slice_particles:
                # Convert physical coordinates to pixel coordinates
                y_px = y / 10.0
                x_px = x / 10.0

                # Add circle
                circle = plt.Circle((x_px, y_px), radius, color=color, fill=False, linewidth=1.5, alpha=0.7)
                axes[i].add_patch(circle)

        # Add legend
        axes[i].legend(loc='upper right', bbox_to_anchor=(1.1, 1))
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(visualization_dir, f'{experiment}_ground_truth.png'), dpi=200, bbox_inches='tight')
    plt.close()

def visualize_model_predictions_on_tomogram(tomo_data, experiment, ground_truth_coords,
                                          particle_types, model_dir, visualization_dir, device):
    """
    Visualize model predictions on a tomogram

    Parameters:
    tomo_data (numpy.ndarray): 3D tomogram data
    experiment (str): Experiment name
    ground_truth_coords (dict): Dictionary mapping particle types to ground truth coordinates
    particle_types (dict): Dictionary of particle types and their properties
    model_dir (str): Path to model directory
    visualization_dir (str): Path to save visualizations
    device: PyTorch device
    """
    # Load the trained model
    model_path = os.path.join(model_dir, 'model.pth')

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return

    # Initialize model
    model = UNet3D(in_channels=1, out_channels=1, init_features=16)

    # Load model weights
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # Get tomogram dimensions
    depth, height, width = tomo_data.shape

    # Create a simplified 3D density map
    print("Generating simplified density map for visualization...")
    density_map = generate_density_map(model, tomo_data, device, patch_size=64, stride=32, batch_size=8)

    # Create a combined visualization with side-by-side ground truth and predictions
    visualize_comparison(tomo_data, density_map, ground_truth_coords, experiment, particle_types, visualization_dir)

def visualize_comparison(tomo_data, density_map, ground_truth_coords, experiment, particle_types, visualization_dir):
    """
    Visualize comparison between ground truth and model predictions

    Parameters:
    tomo_data (numpy.ndarray): 3D tomogram data
    density_map (numpy.ndarray): 3D density map from model predictions
    ground_truth_coords (dict): Dictionary mapping particle types to ground truth coordinates
    experiment (str): Experiment name
    particle_types (dict): Dictionary of particle types and their properties
    visualization_dir (str): Path to save visualizations
    """
    from skimage.feature import peak_local_max

    # Get tomogram dimensions
    depth, height, width = tomo_data.shape

    # Choose slices for visualization
    slices = [depth // 4, depth // 2, 3 * depth // 4]

    # Find local maxima in density map
    print("Finding local maxima in density map...")
    from utils import find_local_maxima
    coords = find_local_maxima(density_map, min_distance=10, threshold_abs=0.3, threshold_rel=0.2)
    print(f"Found {len(coords)} local maxima")

    # Simplified particle type assignment for visualization
    # This doesn't replicate the full clustering logic but is sufficient for visualization
    predicted_coords = {}

    # Basic heuristic: stronger peaks are more likely to be larger/easier particles
    peak_values = np.array([density_map[z, y, x] for z, y, x in coords])
    peak_order = np.argsort(peak_values)[::-1]  # Sort in descending order

    # Assign top 20% to easier particles, next 30% to harder particles
    # Assign top 20% to easier particles, next 30% to harder particles
    n_peaks = len(coords)
    easy_threshold = int(n_peaks * 0.2)
    hard_threshold = int(n_peaks * 0.5)
    
    easy_particles = ['apo-ferritin', 'ribosome', 'virus-like-particle']
    hard_particles = ['beta-galactosidase', 'thyroglobulin']
    
    # Randomly assign among the categories
    np.random.seed(42)  # For reproducibility
    
    for i, idx in enumerate(peak_order):
        z, y, x = coords[idx]
        
        # Convert voxel coordinates to physical coordinates
        physical_x = x * 10.0
        physical_y = y * 10.0
        physical_z = z * 10.0
        
        if i < easy_threshold:
            # Assign to an easy particle type
            p_type = np.random.choice(easy_particles)
        elif i < hard_threshold:
            # Assign to a hard particle type
            p_type = np.random.choice(hard_particles)
        else:
            # Skip the rest
            continue
        
        if p_type not in predicted_coords:
            predicted_coords[p_type] = []
        
        predicted_coords[p_type].append((physical_x, physical_y, physical_z))
    
    # For each slice, create a side-by-side comparison
    for slice_idx in slices:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Ground truth visualization
        ax1.imshow(tomo_data[slice_idx], cmap='gray')
        ax1.set_title(f'Ground Truth - Z-Slice {slice_idx}/{depth}')
        
        # Add a semi-transparent overlay of the density map
        ax2.imshow(tomo_data[slice_idx], cmap='gray')
        density_overlay = ax2.imshow(density_map[slice_idx], cmap='hot', alpha=0.5)
        fig.colorbar(density_overlay, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_title(f'Model Predictions - Z-Slice {slice_idx}/{depth}')
        
        # Get slice range (particles near this slice)
        slice_range = 10  # Consider particles within ±10 slices
        z_min = (slice_idx - slice_range) * 10.0  # Convert to physical coordinates
        z_max = (slice_idx + slice_range) * 10.0
        
        # Add ground truth particles
        for p_type, coords in ground_truth_coords.items():
            # Skip if particle type not in our dictionary
            if p_type not in particle_types:
                continue
            
            # Get color and radius for this particle type
            color = particle_types[p_type]['color']
            radius = particle_types[p_type]['radius'] * 0.1  # Scale down for visualization
            
            # Count particles in this slice
            slice_particles = [(x, y, z) for x, y, z in coords if z_min <= z <= z_max]
            n_particles = len(slice_particles)
            
            # Skip if no particles of this type in this slice
            if n_particles == 0:
                continue
            
            # Add to legend
            ax1.plot([], [], 'o', color=color, label=f'{p_type} ({n_particles})')
            
            # Add circle for each particle
            for x, y, z in slice_particles:
                # Convert physical coordinates to pixel coordinates
                y_px = y / 10.0
                x_px = x / 10.0
                
                # Add circle
                circle = plt.Circle((x_px, y_px), radius, color=color, fill=False, linewidth=1.5, alpha=0.7)
                ax1.add_patch(circle)
        
        # Add predicted particles
        for p_type, coords in predicted_coords.items():
            # Skip if particle type not in our dictionary
            if p_type not in particle_types:
                continue
            
            # Get color and radius for this particle type
            color = particle_types[p_type]['color']
            radius = particle_types[p_type]['radius'] * 0.1  # Scale down for visualization
            
            # Count particles in this slice
            slice_particles = [(x, y, z) for x, y, z in coords if z_min <= z <= z_max]
            n_particles = len(slice_particles)
            
            # Skip if no particles of this type in this slice
            if n_particles == 0:
                continue
            
            # Add to legend
            ax2.plot([], [], 'o', color=color, label=f'{p_type} ({n_particles})')
            
            # Add circle for each particle
            for x, y, z in slice_particles:
                # Convert physical coordinates to pixel coordinates
                y_px = y / 10.0
                x_px = x / 10.0
                
                # Add circle
                circle = plt.Circle((x_px, y_px), radius, color=color, fill=False, linewidth=1.5, alpha=0.7)
                ax2.add_patch(circle)
        
        # Add legends
        ax1.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
        ax2.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
        
        # Remove axes
        ax1.axis('off')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(visualization_dir, f'{experiment}_slice_{slice_idx}_comparison.png'), dpi=200, bbox_inches='tight')
        plt.close()

def main():
    # Set paths
    base_dir = '/kaggle/input/czii-cryo-et-object-identification'
    train_dir = os.path.join(base_dir, 'train')
    model_dir = '/kaggle/working/models'
    visualization_dir = '/kaggle/working/val_visualizations'
    
    # Create visualization directory if it doesn't exist
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare validation tomograms and visualize
    prepare_validation_tomograms(train_dir, model_dir, visualization_dir, device)
    print("Validation tomogram visualization complete.")

if __name__ == "__main__":
    main()
