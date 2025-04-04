import os
import glob
import numpy as np
import pandas as pd
import zarr
import json
import torch
import gc
from tqdm.notebook import tqdm
from scipy.ndimage import gaussian_filter

# Function to load a tomogram
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

# Function to preprocess a tomogram
def preprocess_tomogram(tomo_data):
    """
    Preprocess a tomogram for better prediction

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

# Function to load particle coordinates from JSON
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

# Function to find local maxima in the density map
def find_local_maxima(density_map, min_distance=6, threshold_abs=0.05, threshold_rel=0.03):
    """
    Find local maxima in the density map

    Parameters:
    density_map (numpy.ndarray): Density map
    min_distance (int): Minimum distance between peaks
    threshold_abs (float): Minimum absolute threshold for peak
    threshold_rel (float): Minimum relative threshold for peak

    Returns:
    numpy.ndarray: Array of peak coordinates [z, y, x]
    """
    from skimage.feature import peak_local_max

    # Apply Gaussian smoothing to reduce noise
    smoothed_map = gaussian_filter(density_map, sigma=1.0)

    # Find local maxima
    coordinates = peak_local_max(
        smoothed_map,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        threshold_rel=threshold_rel,
        exclude_border=False
    )

    return coordinates

# Function to write predictions to JSON
def write_predictions_to_json(particles_by_type, output_path):
    """
    Write particle predictions to JSON

    Parameters:
    particles_by_type (dict): Dictionary mapping particle type to list of coordinates
    output_path (str): Path to save the JSON file
    """
    # Format the predictions according to the submission format
    prediction = {"points": []}

    for p_type, coords in particles_by_type.items():
        for x, y, z in coords:
            prediction["points"].append({
                "location": {"x": x, "y": y, "z": z},
                "type": p_type
            })

    # Write to file
    with open(output_path, 'w') as f:
        json.dump(prediction, f)

    print(f"Wrote {len(prediction['points'])} particles to {output_path}")

# Function to evaluate the prediction quality
def evaluate_prediction(particles_by_type, particle_types):
    """
    Evaluate prediction quality using heuristic metrics

    Parameters:
    particles_by_type (dict): Dictionary mapping particle type to list of coordinates
    particle_types (dict): Dictionary of particle types and their properties

    Returns:
    float: Heuristic quality score (0-1)
    """
    # Calculate score based on particle distribution
    scores = []

    # Expected particle counts based on observations
    expected_counts = {
        'apo-ferritin': 80,
        'beta-galactosidase': 60,
        'ribosome': 80,
        'thyroglobulin': 60,
        'virus-like-particle': 40
    }

    for p_type, coords in particles_by_type.items():
        if p_type not in expected_counts:
            continue

        # Count
        count = len(coords)
        expected = expected_counts[p_type]

        # Calculate normalized score (1.0 if count matches expected, less otherwise)
        # Use min-max normalization
        ratio = min(count / expected, 1.0) if expected > 0 else 0.0

        # Weight by particle importance
        weight = particle_types[p_type].get('weight', 1.0)

        # Add to scores
        scores.append(ratio * weight)

    # Calculate final score
    total_weight = sum(particle_types[p_type].get('weight', 1.0) for p_type in expected_counts if p_type in particle_types)

    if total_weight > 0:
        return sum(scores) / total_weight
    else:
        return 0.0

# Function to generate a density map for a full tomogram
def generate_density_map(model, tomo_data, device, patch_size=64, stride=32, batch_size=8):
    """
    Generate a density map for a full tomogram

    Parameters:
    model (nn.Module): Trained model
    tomo_data (numpy.ndarray): 3D tomogram data
    device: PyTorch device
    patch_size (int): Size of cubic patches
    stride (int): Stride between patches
    batch_size (int): Batch size for prediction

    Returns:
    numpy.ndarray: Density map
    """
    model.eval()

    # Get dimensions
    depth, height, width = tomo_data.shape

    # Initialize density map and count array
    density_map = np.zeros_like(tomo_data, dtype=np.float32)
    count = np.zeros_like(tomo_data, dtype=np.float32)

    # Half size for patch extraction
    half_size = patch_size // 2

    # Extract patches
    patches = []
    coordinates = []

    print("Extracting patches...")
    for z in tqdm(range(half_size, depth - half_size, stride)):
        for y in range(half_size, height - half_size, stride):
            for x in range(half_size, width - half_size, stride):
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
                coordinates.append((z, y, x))

                # Process in batches to avoid memory issues
                if len(patches) >= batch_size:
                    # Convert to tensor
                    batch_patches = torch.FloatTensor(np.array(patches)).unsqueeze(1).to(device)

                    # Predict
                    with torch.no_grad():
                        batch_outputs = model(batch_patches)

                    # Move to CPU and convert to numpy
                    batch_outputs = batch_outputs.detach().cpu().numpy()

                    # Add to density map
                    for i, (z, y, x) in enumerate(coordinates):
                        output = batch_outputs[i, 0]

                        # Add to density map
                        density_map[
                            z - half_size:z + half_size,
                            y - half_size:y + half_size,
                            x - half_size:x + half_size
                        ] += output

                        # Increment count
                        count[
                            z - half_size:z + half_size,
                            y - half_size:y + half_size,
                            x - half_size:x + half_size
                        ] += 1

                    # Clear lists
                    patches = []
                    coordinates = []

    # Process remaining patches
    if patches:
        # Convert to tensor
        batch_patches = torch.FloatTensor(np.array(patches)).unsqueeze(1).to(device)

        # Predict
        with torch.no_grad():
            batch_outputs = model(batch_patches)

        # Move to CPU and convert to numpy
        batch_outputs = batch_outputs.detach().cpu().numpy()

        # Add to density map
        for i, (z, y, x) in enumerate(coordinates):
            output = batch_outputs[i, 0]

            # Add to density map
            density_map[
                z - half_size:z + half_size,
                y - half_size:y + half_size,
                x - half_size:x + half_size
            ] += output

            # Increment count
            count[
                z - half_size:z + half_size,
                y - half_size:y + half_size,
                x - half_size:z + half_size
            ] += 1

    # Average overlapping regions
    density_map = np.divide(density_map, count, out=np.zeros_like(density_map), where=count > 0)

    return density_map
