import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zarr
import torch
import gc
from tqdm.notebook import tqdm
from scipy.ndimage import gaussian_filter
import glob

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
    Preprocess a tomogram
    
    Parameters:
    tomo_data (numpy.ndarray): 3D tomogram data
    
    Returns:
    numpy.ndarray: Preprocessed tomogram data
    """
    # Make a copy to avoid modifying the original
    processed = tomo_data.copy()
    
    # Apply Gaussian filtering to reduce noise
    processed = gaussian_filter(processed, sigma=1.0)
    
    # Normalize to [0, 1] range
    min_val = processed.min()
    max_val = processed.max()
    if max_val > min_val:
        processed = (processed - min_val) / (max_val - min_val)
    
    # Enhance contrast
    processed = np.clip((processed - 0.1) * 1.25, 0, 1)
    
    return processed

def load_particle_coords(json_path):
    """
    Load particle coordinates from JSON file
    
    Parameters:
    json_path (str): Path to the JSON file
    
    Returns:
    list: List of (x, y, z) coordinate tuples
    """
    import json
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

def create_density_map(shape, coords, radius=10):
    """
    Create a density map for particle locations
    
    Parameters:
    shape (tuple): Shape of the output density map (depth, height, width)
    coords (list): List of (x, y, z) coordinate tuples
    radius (float): Radius of particles in voxels
    
    Returns:
    numpy.ndarray: Density map with Gaussian-like peaks at particle locations
    """
    # Initialize empty map
    density_map = np.zeros(shape, dtype=np.float32)
    
    # Convert coordinates to indices
    # Assuming 10 Angstrom voxel spacing
    voxel_spacing = 10.0
    
    depth, height, width = shape
    
    # Add Gaussian peaks for each particle
    for x, y, z in coords:
        # Convert physical coordinates to voxel indices
        z_idx, y_idx, x_idx = int(z / voxel_spacing), int(y / voxel_spacing), int(x / voxel_spacing)
        
        # Skip if outside the volume
        if not (0 <= z_idx < depth and 0 <= y_idx < height and 0 <= x_idx < width):
            continue
        
        # Create a spherical mask around the particle
        z_min = max(0, z_idx - radius)
        z_max = min(depth, z_idx + radius + 1)
        y_min = max(0, y_idx - radius)
        y_max = min(height, y_idx + radius + 1)
        x_min = max(0, x_idx - radius)
        x_max = min(width, x_idx + radius + 1)
        
        # Create coordinate grids
        z_grid, y_grid, x_grid = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]
        
        # Calculate distance from center
        dist_from_center = np.sqrt(
            (z_grid - z_idx)**2 +
            (y_grid - y_idx)**2 +
            (x_grid - x_idx)**2
        )
        
        # Use a Gaussian-like function to create soft spheres
        mask = np.exp(-(dist_from_center**2) / (2 * (radius/2)**2))
        
        # Add to density map
        density_map[z_min:z_max, y_min:y_max, x_min:x_max] = np.maximum(
            density_map[z_min:z_max, y_min:y_max, x_min:x_max],
            mask
        )
    
    return density_map

def extract_subvolumes(tomo_data, coords, patch_size=64, target_radius=6, max_patches_per_tomo=1000):
    """
    Extract subvolumes (patches) from a tomogram
    
    Parameters:
    tomo_data (numpy.ndarray): 3D tomogram data
    coords (dict): Dictionary mapping particle types to coordinate lists
    patch_size (int): Size of cubic patches to extract
    target_radius (int): Radius of the target in the output density map (in voxels)
    max_patches_per_tomo (int): Maximum number of patches to extract per tomogram
    
    Returns:
    tuple: (patches, labels) where patches are subvolumes and labels are target density maps
    """
    depth, height, width = tomo_data.shape
    half_size = patch_size // 2
    
    patches = []
    labels = []
    metadata = []
    
    # Combine all coordinates
    all_coords = []
    for p_type, coords_list in coords.items():
        for coord in coords_list:
            all_coords.append((coord[0], coord[1], coord[2], p_type))
    
    # Shuffle to ensure random selection if we hit max_patches_per_tomo
    np.random.shuffle(all_coords)
    
    # Extract patches around particle centers
    patch_count = 0
    for x, y, z, p_type in all_coords:
        if patch_count >= max_patches_per_tomo:
            break
        
        # Convert physical coordinates to voxel indices
        z_idx, y_idx, x_idx = int(z / 10.0), int(y / 10.0), int(x / 10.0)
        
        # Check if the patch is fully within the tomogram
        if (z_idx - half_size < 0 or z_idx + half_size >= depth or
            y_idx - half_size < 0 or y_idx + half_size >= height or
            x_idx - half_size < 0 or x_idx + half_size >= width):
            continue
        
        # Extract the patch
        patch = tomo_data[
            z_idx - half_size:z_idx + half_size,
            y_idx - half_size:y_idx + half_size,
            x_idx - half_size:x_idx + half_size
        ]
        
        # Skip if patch is invalid
        if patch.shape != (patch_size, patch_size, patch_size):
            continue
        
        # Get list of scored particle types (weight > 0)
        particle_types = {
            'apo-ferritin': {'difficulty': 'easy', 'weight': 1},
            'beta-amylase': {'difficulty': 'impossible', 'weight': 0},
            'beta-galactosidase': {'difficulty': 'hard', 'weight': 2},
            'ribosome': {'difficulty': 'easy', 'weight': 1},
            'thyroglobulin': {'difficulty': 'hard', 'weight': 2},
            'virus-like-particle': {'difficulty': 'easy', 'weight': 1}
        }
        scored_particle_types = [p for p, props in particle_types.items() if props['weight'] > 0]
        
        # Create a label (target density map)
        if p_type in scored_particle_types:
            # Only create a target for scored particle types
            label = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)
            
            # Create a spherical mask
            z_grid, y_grid, x_grid = np.ogrid[
                :patch_size,
                :patch_size,
                :patch_size
            ]
            center = patch_size // 2
            
            dist_from_center = np.sqrt(
                (z_grid - center)**2 +
                (y_grid - center)**2 +
                (x_grid - center)**2
            )
            
            # Create Gaussian-like target
            label = np.exp(-(dist_from_center**2) / (2 * (target_radius/2)**2))
        else:
            # For non-scored particles, use empty labels
            label = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)
        
        patches.append(patch)
        labels.append(label)
        metadata.append({
            'particle_type': p_type,
            'x': x,
            'y': y,
            'z': z,
            'weight': particle_types[p_type]['weight']
        })
        
        patch_count += 1
    
    # Also add some random negative patches (background)
    num_negative = min(len(patches) // 4, max_patches_per_tomo - patch_count)
    
    for _ in range(num_negative):
        # Generate random coordinates away from particles
        while True:
            z_idx = np.random.randint(half_size, depth - half_size)
            y_idx = np.random.randint(half_size, height - half_size)
            x_idx = np.random.randint(half_size, width - half_size)
            
            # Check if this point is far from all particles
            physical_x = x_idx * 10.0
            physical_y = y_idx * 10.0
            physical_z = z_idx * 10.0
            
            # Check distance from all particles
            min_dist = float('inf')
            for x, y, z, p_type in all_coords:
                dist = np.sqrt((x - physical_x)**2 + (y - physical_y)**2 + (z - physical_z)**2)
                min_dist = min(min_dist, dist)
            
            # If far enough from particles, use this location
            if min_dist > 100:  # 100 Angstroms away from any particle
                break
        
        # Extract the patch
        patch = tomo_data[
            z_idx - half_size:z_idx + half_size,
            y_idx - half_size:y_idx + half_size,
            x_idx - half_size:x_idx + half_size
        ]
        
        # Skip if patch is invalid
        if patch.shape != (patch_size, patch_size, patch_size):
            continue
        
        # Use empty label for negative patches
        label = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)
        
        patches.append(patch)
        labels.append(label)
        metadata.append({
            'particle_type': 'background',
            'x': physical_x,
            'y': physical_y,
            'z': physical_z,
            'weight': 0
        })
    
    return np.array(patches), np.array(labels), metadata

def process_training_data(train_dir, output_dir, patch_size=64, split_ratio=0.2):
    """
    Process all training data, extract patches, and save them
    
    Parameters:
    train_dir (str): Path to training data directory
    output_dir (str): Path to output directory
    patch_size (int): Size of cubic patches to extract
    split_ratio (float): Ratio for validation split
    
    Returns:
    tuple: (train_patches, train_labels, val_patches, val_labels, metadata)
    """
    all_patches = []
    all_labels = []
    all_metadata = []
    
    # Get list of training experiments
    train_experiments = [os.path.basename(p) for p in glob.glob(os.path.join(train_dir, 'static/ExperimentRuns/*'))]
    print(f"Found {len(train_experiments)} training experiments: {train_experiments}")
    
    # Process each experiment
    for experiment in tqdm(train_experiments, desc="Processing experiments"):
        # Load tomogram
        zarr_path = os.path.join(train_dir, 'static/ExperimentRuns', experiment, 'VoxelSpacing10.000/denoised.zarr')
        
        if not os.path.exists(zarr_path):
            print(f"Tomogram not found: {zarr_path}")
            continue
        
        tomo_data = load_tomogram(zarr_path)
        
        if tomo_data is None:
            print(f"Failed to load tomogram for experiment {experiment}")
            continue
        
        # Preprocess tomogram
        tomo_data = preprocess_tomogram(tomo_data)
        
        # Load particle coordinates for each particle type
        coords = {}
        
        # Define particle types
        particle_types = {
            'apo-ferritin': {'difficulty': 'easy', 'weight': 1},
            'beta-amylase': {'difficulty': 'impossible', 'weight': 0},
            'beta-galactosidase': {'difficulty': 'hard', 'weight': 2},
            'ribosome': {'difficulty': 'easy', 'weight': 1},
            'thyroglobulin': {'difficulty': 'hard', 'weight': 2},
            'virus-like-particle': {'difficulty': 'easy', 'weight': 1}
        }
        
        for p_type in particle_types.keys():
            json_path = os.path.join(train_dir, 'overlay/ExperimentRuns', experiment, 'Picks', f"{p_type}.json")
            
            if os.path.exists(json_path):
                coords_list = load_particle_coords(json_path)
                if coords_list:
                    coords[p_type] = coords_list
        
        # Extract patches
        patches, labels, metadata = extract_subvolumes(tomo_data, coords, patch_size)
        
        # Add experiment to metadata
        for m in metadata:
            m['experiment'] = experiment
        
        # Collect patches, labels, and metadata
        all_patches.append(patches)
        all_labels.append(labels)
        all_metadata.extend(metadata)
        
        # Free memory
        del tomo_data, patches, labels
        gc.collect()
    
    # Combine all patches and labels
    all_patches = np.concatenate(all_patches, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Create metadata DataFrame
    metadata_df = pd.DataFrame(all_metadata)
    
    # Save metadata
    os.makedirs(os.path.join(output_dir, 'metadata'), exist_ok=True)
    metadata_df.to_csv(os.path.join(output_dir, 'metadata', 'training_metadata.csv'), index=False)
    
    # Split into training and validation sets
    from sklearn.model_selection import train_test_split
    train_patches, val_patches, train_labels, val_labels = train_test_split(
        all_patches, all_labels, test_size=split_ratio, random_state=42
    )
    
    # Save patches and labels
    os.makedirs(os.path.join(output_dir, 'training'), exist_ok=True)
    np.save(os.path.join(output_dir, 'training', 'train_patches.npy'), train_patches)
    np.save(os.path.join(output_dir, 'training', 'train_labels.npy'), train_labels)
    np.save(os.path.join(output_dir, 'training', 'val_patches.npy'), val_patches)
    np.save(os.path.join(output_dir, 'training', 'val_labels.npy'), val_labels)
    
    # Print statistics
    print(f"Processed {len(all_patches)} patches")
    print(f"Training set: {len(train_patches)} patches")
    print(f"Validation set: {len(val_patches)} patches")
    
    # Count number of patches per particle type
    particle_counts = metadata_df['particle_type'].value_counts()
    print("\nPatch distribution by particle type:")
    for p_type, count in particle_counts.items():
        print(f" - {p_type}: {count} patches")
    
    # Plot a few random patches
    plot_random_patches(train_patches, train_labels, metadata_df, n_samples=5)
    
    return train_patches, train_labels, val_patches, val_labels, metadata_df

def process_test_data(test_dir, output_dir, patch_size=64, stride=32):
    """
    Process test data by extracting overlapping patches
    
    Parameters:
    test_dir (str): Path to test data directory
    output_dir (str): Path to output directory
    patch_size (int): Size of cubic patches to extract
    stride (int): Stride between patches
    
    Returns:
    dict: Dictionary of test data by experiment
    """
    test_data = {}
    
    # Get list of test experiments
    test_experiments = [os.path.basename(p) for p in glob.glob(os.path.join(test_dir, 'static/ExperimentRuns/*'))]
    print(f"Found {len(test_experiments)} test experiments: {test_experiments}")
    
    # Process each experiment
    for experiment in tqdm(test_experiments, desc="Processing test experiments"):
        # Load tomogram
        zarr_path = os.path.join(test_dir, 'static/ExperimentRuns', experiment, 'VoxelSpacing10.000/denoised.zarr')
        
        if not os.path.exists(zarr_path):
            print(f"Test tomogram not found: {zarr_path}")
            continue
        
        tomo_data = load_tomogram(zarr_path)
        
        if tomo_data is None:
            print(f"Failed to load test tomogram for experiment {experiment}")
            continue
        
        # Preprocess tomogram
        tomo_data = preprocess_tomogram(tomo_data)
        
        # Extract overlapping patches
        depth, height, width = tomo_data.shape
        half_size = patch_size // 2
        
        # Initialize data structures
        patches = []
        coordinates = []
        
        # Extract patches with overlap (stride)
        for z in range(half_size, depth - half_size, stride):
            for y in range(half_size, height - half_size, stride):
                for x in range(half_size, width - half_size, stride):
                    # Extract the patch
                    # Extract the patch
                    patch = tomo_data[
                        z - half_size:z + half_size,
                        y - half_size:y + half_size,
                        x - half_size:x + half_size
                    ]
                    
                    # Skip if patch is invalid
                    if patch.shape != (patch_size, patch_size, patch_size):
                        continue
                    
                    patches.append(patch)
                    coordinates.append((x, y, z))
        
        # Convert to numpy arrays
        patches = np.array(patches)
        
        # Store test data
        test_data[experiment] = {
            'patches': patches,
            'coordinates': coordinates,
            'shape': tomo_data.shape
        }
        
        # Save the test data
        os.makedirs(os.path.join(output_dir, 'test', experiment), exist_ok=True)
        np.save(os.path.join(output_dir, 'test', experiment, 'patches.npy'), patches)
        np.save(os.path.join(output_dir, 'test', experiment, 'coordinates.npy'), coordinates)
        np.save(os.path.join(output_dir, 'test', experiment, 'shape.npy'), tomo_data.shape)
        
        # Print statistics
        print(f"Processed {len(patches)} patches for experiment {experiment}")
        
        # Free memory
        del tomo_data, patches
        gc.collect()
    
    return test_data

def plot_random_patches(patches, labels, metadata_df, n_samples=5):
    """
    Plot random patches and their labels
    
    Parameters:
    patches (numpy.ndarray): Array of patches
    labels (numpy.ndarray): Array of labels
    metadata_df (pandas.DataFrame): DataFrame with metadata
    n_samples (int): Number of samples to plot
    """
    # Get random indices
    indices = np.random.choice(len(patches), size=n_samples, replace=False)
    
    # Plot each sample
    for i, idx in enumerate(indices):
        patch = patches[idx]
        label = labels[idx]
        
        # Get metadata for this patch
        if i < len(metadata_df):
            p_type = metadata_df.iloc[idx]['particle_type']
            p_weight = metadata_df.iloc[idx]['weight']
        else:
            p_type = "Unknown"
            p_weight = "Unknown"
        
        # Plot the middle slice of the patch and label
        middle_slice = patch.shape[0] // 2
        
        plt.figure(figsize=(12, 6))
        
        # Plot patch
        plt.subplot(1, 2, 1)
        plt.imshow(patch[middle_slice], cmap='gray')
        plt.title(f"Patch {idx}: {p_type} (Weight: {p_weight})")
        plt.axis('off')
        
        # Plot label
        plt.subplot(1, 2, 2)
        plt.imshow(label[middle_slice], cmap='hot')
        plt.title(f"Label {idx}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    # Set paths
    base_dir = '/kaggle/input/czii-cryo-et-object-identification'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    output_dir = '/kaggle/working/preprocessed'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting data preprocessing...")
    
    # Set parameters
    patch_size = 64  # Size of cubic patches
    split_ratio = 0.2  # Ratio for validation split
    
    # Process training data
    train_patches, train_labels, val_patches, val_labels, metadata_df = process_training_data(
        train_dir, output_dir, patch_size, split_ratio
    )
    
    # Process test data
    test_data = process_test_data(test_dir, output_dir, patch_size, stride=32)
    
    print("Data preprocessing complete.")

if __name__ == "__main__":
    main()

       


  

   
