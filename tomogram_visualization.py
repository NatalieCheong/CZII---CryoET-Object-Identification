import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import zarr
import json
from tqdm.notebook import tqdm

def load_tomogram(zarr_path, resolution=0):
    """
    Load a tomogram from a zarr file

    Parameters:
    zarr_path (str): Path to the zarr directory
    resolution (int): Resolution level (0 = highest, 1 = medium, 2 = lowest)

    Returns:
    numpy.ndarray: The loaded tomogram data
    """
    try:
        # Open the zarr group
        z = zarr.open(zarr_path, mode='r')

        # Access the resolution level directly
        if str(resolution) in z:
            tomo_data = z[str(resolution)][:]
            print(f"Loaded tomogram with shape {tomo_data.shape}")
            return tomo_data
        else:
            print(f"Resolution level {resolution} not found in zarr file")
            return None
    except Exception as e:
        print(f"Error loading tomogram: {str(e)}")
        return None

def preprocess_tomogram(tomo_data):
    """
    Preprocess a tomogram for better visualization

    Parameters:
    tomo_data (numpy.ndarray): 3D tomogram data

    Returns:
    numpy.ndarray: Preprocessed tomogram data
    """
    # Make a copy to avoid modifying the original
    processed = tomo_data.copy()

    # Normalize to [0, 1] range
    min_val = processed.min()
    max_val = processed.max()
    if max_val > min_val:
        processed = (processed - min_val) / (max_val - min_val)

    # Enhance contrast
    processed = np.clip((processed - 0.1) * 1.25, 0, 1)

    return processed

def visualize_tomogram(tomo_data, title="Tomogram Slices", n_slices=3, figsize=(15, 5)):
    """
    Visualize slices of a 3D tomogram

    Parameters:
    tomo_data (numpy.ndarray): 3D tomogram data
    title (str): Plot title
    n_slices (int): Number of slices to visualize
    figsize (tuple): Figure size
    """
    if tomo_data is None:
        print("No tomogram data to visualize")
        return

    # Get tomogram dimensions
    depth, height, width = tomo_data.shape
    print(f"Tomogram dimensions: {depth} x {height} x {width}")

    # Choose slice indices at different depths
    slice_indices = np.linspace(depth // 4, 3 * depth // 4, n_slices).astype(int)

    # Create figure
    fig, axes = plt.subplots(1, n_slices, figsize=figsize)
    if n_slices == 1:
        axes = [axes]

    # Plot each slice
    for i, slice_idx in enumerate(slice_indices):
        im = axes[i].imshow(tomo_data[slice_idx], cmap='gray')
        axes[i].set_title(f'Z-Slice {slice_idx}/{depth}')
        axes[i].axis('off')
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

    # Show orthogonal views (XY, XZ, YZ planes)
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # XY plane (middle slice in Z)
    z_mid = depth // 2
    axes[0].imshow(tomo_data[z_mid], cmap='gray')
    axes[0].set_title(f'XY Plane (Z={z_mid})')
    axes[0].axis('off')

    # XZ plane (middle slice in Y)
    y_mid = height // 2
    axes[1].imshow(tomo_data[:, y_mid, :], cmap='gray')
    axes[1].set_title(f'XZ Plane (Y={y_mid})')
    axes[1].axis('off')

    # YZ plane (middle slice in X)
    x_mid = width // 2
    axes[2].imshow(tomo_data[:, :, x_mid], cmap='gray')
    axes[2].set_title(f'YZ Plane (X={x_mid})')
    axes[2].axis('off')

    plt.suptitle(f"{title} - Orthogonal Views")
    plt.tight_layout()
    plt.show()

def load_particle_coords(json_path):
    """
    Load particle coordinates from JSON file

    Parameters:
    json_path (str): Path to the JSON file

    Returns:
    list: List of (x, y, z) coordinate tuples
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract coordinates from points
        coords = []
        if 'points' in data:
            for point in data['points']:
                if 'location' in point:
                    loc = point['location']
                    coords.append((loc.get('x', 0), loc.get('y', 0), loc.get('z', 0)))

        return coords
    except Exception as e:
        print(f"Error loading {json_path}: {str(e)}")
        return []

def visualize_tomogram_with_particles(tomo_data, json_paths, title="Tomogram with Particles", slice_idx=None):
    """
    Visualize a tomogram slice with particle positions overlaid

    Parameters:
    tomo_data (numpy.ndarray): 3D tomogram data
    json_paths (list): List of paths to JSON files with particle coordinates
    title (str): Plot title
    slice_idx (int, optional): Specific slice to visualize. If None, middle slice is used.
    """
    if tomo_data is None:
        print("No tomogram data to visualize")
        return

    # Get tomogram dimensions
    depth, height, width = tomo_data.shape

    # Choose slice index if not specified
    if slice_idx is None:
        slice_idx = depth // 2

    # Load particle coordinates from all JSON files
    particles_by_type = {}

    for json_path in json_paths:
        particle_type = os.path.splitext(os.path.basename(json_path))[0]

        coords = load_particle_coords(json_path)
        if coords:
            particles_by_type[particle_type] = coords

    # Define colors for different particle types
    colors = {
        'apo-ferritin': 'red',
        'beta-amylase': 'gray',
        'beta-galactosidase': 'blue',
        'ribosome': 'green',
        'thyroglobulin': 'purple',
        'virus-like-particle': 'orange'
    }

    # Create figure
    plt.figure(figsize=(12, 10))

    # Show the tomogram slice
    plt.imshow(tomo_data[slice_idx], cmap='gray')

    # Overlay particle positions near the slice
    slice_range = 10  # Consider particles within ±10 slices

    for p_type, coords in particles_by_type.items():
        # Filter coordinates near the current slice
        slice_coords = []
        for x, y, z in coords:
            z_idx = int(z / 10)  # Convert physical z to index (assuming 10 Å voxel spacing)
            if abs(z_idx - slice_idx) <= slice_range:
                slice_coords.append((x, y))

        if slice_coords:
            x_coords = [x for x, _ in slice_coords]
            y_coords = [y for _, y in slice_coords]

            plt.scatter(x_coords, y_coords,
                       c=colors.get(p_type, 'white'),
                       label=f'{p_type} ({len(slice_coords)})',
                       alpha=0.7, s=30, edgecolors='white')

    plt.title(f'{title}\nSlice {slice_idx}/{depth}')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_3d_particle_distribution(particle_data, experiment):
    """
    Visualize 3D distribution of particles for a single experiment

    Parameters:
    particle_data (pd.DataFrame): DataFrame with particle information
    experiment (str): Experiment name to visualize
    """
    import pandas as pd  # Import here for clarity
    from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

    # Filter data for the specified experiment
    exp_data = particle_data[particle_data['experiment'] == experiment]

    if len(exp_data) == 0:
        print(f"No data found for experiment {experiment}")
        return

    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each particle type with a different color
    for p_type, group in exp_data.groupby('particle_type'):
        ax.scatter(group['x'], group['y'], group['z'],
                  label=p_type,
                  alpha=0.7, s=15)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Distribution of Particles in Experiment {experiment}')
    ax.legend()
    plt.tight_layout()
    plt.show()

def compare_tomogram_types(experiment, train_dir):
    """
    Compare different tomogram types (denoised, wbp, etc.) for the same experiment

    Parameters:
    experiment (str): Experiment name
    train_dir (str): Path to training data directory
    """
    # Define tomogram types to compare
    tomogram_types = ['denoised.zarr', 'ctfdeconvolved.zarr', 'isonetcorrected.zarr', 'wbp.zarr']

    # Load and visualize each tomogram type
    tomograms = {}

    for tomo_type in tomogram_types:
        tomo_path = os.path.join(train_dir, 'static/ExperimentRuns', experiment,
                              f'VoxelSpacing10.000/{tomo_type}')

        if os.path.exists(tomo_path):
            print(f"\nLoading {tomo_type} tomogram...")
            tomo_data = load_tomogram(tomo_path, resolution=0)

            if tomo_data is not None:
                tomograms[tomo_type] = tomo_data

    # Visualize comparison
    if tomograms:
        # Choose a middle slice
        slice_idx = tomograms[list(tomograms.keys())[0]].shape[0] // 2

        # Create figure with subplots
        fig, axes = plt.subplots(1, len(tomograms), figsize=(6 * len(tomograms), 6))
        if len(tomograms) == 1:
            axes = [axes]

        # Plot each tomogram type
        for i, (tomo_type, tomo_data) in enumerate(tomograms.items()):
            axes[i].imshow(tomo_data[slice_idx], cmap='gray')
            axes[i].set_title(f'{tomo_type} (Slice {slice_idx})')
            axes[i].axis('off')

        plt.suptitle(f'Tomogram Types Comparison - Experiment {experiment}')
        plt.tight_layout()
        plt.show()
    else:
        print("No tomograms loaded for comparison")

def main():
    # Set paths
    base_dir = '/kaggle/input/czii-cryo-et-object-identification'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    # Get list of training experiments
    train_experiments = [os.path.basename(p) for p in glob.glob(os.path.join(train_dir, 'static/ExperimentRuns/*'))]
    print(f"Found {len(train_experiments)} experiments in training data: {train_experiments}")

    # Choose one experiment for visualization
    if train_experiments:
        sample_experiment = train_experiments[0]
        print(f"\nVisualizing data for experiment: {sample_experiment}")

        # Path to the denoised tomogram
        denoised_path = os.path.join(train_dir, 'static/ExperimentRuns', sample_experiment, 'VoxelSpacing10.000/denoised.zarr')

        # Check if the path exists
        if os.path.exists(denoised_path):
            print(f"Loading denoised tomogram from: {denoised_path}")

            # Load and visualize the tomogram
            tomo_data = load_tomogram(denoised_path, resolution=0)

            if tomo_data is not None:
                # Preprocess for better visualization
                tomo_data = preprocess_tomogram(tomo_data)

                # Visualize the tomogram
                visualize_tomogram(tomo_data,
                                  title=f"Experiment {sample_experiment} - Denoised",
                                  n_slices=3)

                # Get particle annotation files for this experiment
                particle_jsons = glob.glob(os.path.join(train_dir, 'overlay/ExperimentRuns', sample_experiment, 'Picks/*.json'))

                if particle_jsons:
                    print(f"Found {len(particle_jsons)} particle annotation files")

                    # Visualize tomogram with particles
                    visualize_tomogram_with_particles(tomo_data, particle_jsons,
                                                    title=f"Experiment {sample_experiment} - With Particles")
                else:
                    print("No particle annotation files found for this experiment")

                # Compare different tomogram types
                compare_tomogram_types(sample_experiment, train_dir)

        else:
            print(f"Tomogram path does not exist: {denoised_path}")
    else:
        print("No experiment directories found in training data")

    # Also check a test experiment
    test_experiments = [os.path.basename(p) for p in glob.glob(os.path.join(test_dir, 'static/ExperimentRuns/*'))]
    if test_experiments:
        test_experiment = test_experiments[0]
        print(f"\nVisualizing data for test experiment: {test_experiment}")

        # Path to the denoised tomogram
        test_denoised_path = os.path.join(test_dir, 'static/ExperimentRuns', test_experiment, 'VoxelSpacing10.000/denoised.zarr')

        if os.path.exists(test_denoised_path):
            print(f"Loading test tomogram from: {test_denoised_path}")

            # Load and visualize test tomogram
            test_tomo = load_tomogram(test_denoised_path, resolution=0)

            if test_tomo is not None:
                # Preprocess for better visualization
                test_tomo = preprocess_tomogram(test_tomo)

                # Visualize the tomogram
                visualize_tomogram(test_tomo,
                                  title=f"Test Experiment {test_experiment} - Denoised",
                                  n_slices=3)
        else:
            print(f"Test tomogram path does not exist: {test_denoised_path}")

if __name__ == "__main__":
    main()
