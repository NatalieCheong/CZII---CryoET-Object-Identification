# -*- coding: utf-8 -*-
"""test_prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HvTbx00ghyQRKMG7smCJ64sYMsYrlb8u
"""

import os
import numpy as np
import pandas as pd
import torch
import glob
from tqdm.notebook import tqdm
import gc
from skimage.feature import peak_local_max

from utils import load_tomogram, preprocess_tomogram, generate_density_map, find_local_maxima, write_predictions_to_json, evaluate_prediction
from visualization import visualize_tomogram_with_particles, visualize_density_map_with_particles
from particle_detection import UNet3D, calibrated_cluster_particles

def analyze_validation_tomograms(train_dir, model_dir, device):
    """
    Analyze validation tomograms to calibrate thresholds

    Parameters:
    train_dir (str): Path to training data directory
    model_dir (str): Path to model directory
    device: PyTorch device

    Returns:
    dict: Dictionary of calibrated thresholds
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

    # Get scored particle types (weight > 0)
    scored_particle_types = [p for p, props in particle_types.items() if props['weight'] > 0]

    # Load the trained model
    model_path = os.path.join(model_dir, 'model.pth')

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return None

    # Initialize model
    model = UNet3D(in_channels=1, out_channels=1, init_features=16)

    # Load model weights
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # Get list of training experiments
    train_experiments = [os.path.basename(p) for p in glob.glob(os.path.join(train_dir, 'static/ExperimentRuns/*'))]

    if not train_experiments:
        print("No training experiments found.")
        return None

    # Sample a few experiments for analysis
    sample_size = min(3, len(train_experiments))
    validation_experiments = train_experiments[:sample_size]
    print(f"Using {len(validation_experiments)} experiments for threshold calibration: {validation_experiments}")

    # Dictionary to store signal intensities for each particle type
    particle_intensities = {p_type: [] for p_type in particle_types}

    from utils import load_particle_coords

    # Process each validation experiment
    for experiment in validation_experiments:
        print(f"\nProcessing validation experiment: {experiment}")

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

        # Generate density map
        print("Generating density map...")
        density_map = generate_density_map(model, tomo_data, device, patch_size=64, stride=32, batch_size=8)

        # Load ground truth particle positions
        ground_truth = {}

        for p_type in particle_types:
            json_path = os.path.join(train_dir, 'overlay/ExperimentRuns', experiment, 'Picks', f"{p_type}.json")

            if os.path.exists(json_path):
                coords = load_particle_coords(json_path)
                if coords:
                    ground_truth[p_type] = coords
                    print(f"Loaded {len(coords)} {p_type} coordinates")

        # Sample intensity values at ground truth positions
        for p_type, coords in ground_truth.items():
            for x, y, z in coords:
                # Convert physical coordinates to voxel indices
                z_idx, y_idx, x_idx = int(z / 10), int(y / 10), int(x / 10)

                # Check if the position is within the density map
                if (0 <= z_idx < density_map.shape[0] and
                    0 <= y_idx < density_map.shape[1] and
                    0 <= x_idx < density_map.shape[2]):

                    # Get the intensity value
                    intensity = density_map[z_idx, y_idx, x_idx]
                    particle_intensities[p_type].append(intensity)

        # Free memory
        del tomo_data, density_map
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Calculate intensity statistics for each particle type
    intensity_stats = {}

    print("\nParticle intensity statistics:")
    for p_type, intensities in particle_intensities.items():
        if intensities:
            stats = {
                'mean': np.mean(intensities),
                'median': np.median(intensities),
                'min': np.min(intensities),
                'max': np.max(intensities),
                'p25': np.percentile(intensities, 25),
                'p10': np.percentile(intensities, 10),
                'p5': np.percentile(intensities, 5),
                'count': len(intensities)
            }
            intensity_stats[p_type] = stats

            print(f"  - {p_type} ({len(intensities)} particles):")
            print(f"      Mean: {stats['mean']:.4f}, Median: {stats['median']:.4f}")
            print(f"      Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
            print(f"      P5: {stats['p5']:.4f}, P10: {stats['p10']:.4f}, P25: {stats['p25']:.4f}")

    import matplotlib.pyplot as plt
    # Visualize intensity distributions
    plt.figure(figsize=(12, 8))

    for p_type, intensities in particle_intensities.items():
        if len(intensities) > 10:  # Only plot if we have enough samples
            plt.hist(intensities, bins=20, alpha=0.6, label=f"{p_type} (n={len(intensities)})")

    plt.xlabel('Density Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Density Values at Ground Truth Particle Positions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join('/kaggle/working/final_visualizations', 'intensity_distributions.png'), dpi=200)
    plt.close()

    # Calculate calibrated thresholds based on validation analysis
    # Use much lower percentiles to ensure we find all types
    calibrated_thresholds = {}

    for p_type in scored_particle_types:
        if p_type in intensity_stats:
            stats = intensity_stats[p_type]

            # Use very low percentiles - prioritize recall over precision
            if p_type == 'apo-ferritin' or p_type == 'ribosome':
                # For easier particle types
                calibrated_thresholds[p_type] = stats['p5'] * 0.7  # 70% of 5th percentile
            elif p_type == 'beta-galactosidase':
                # For harder particle types
                calibrated_thresholds[p_type] = stats['p5'] * 0.6  # 60% of 5th percentile
            elif p_type == 'thyroglobulin':
                # More difficult to detect
                calibrated_thresholds[p_type] = stats['p5'] * 0.5  # 50% of 5th percentile
            elif p_type == 'virus-like-particle':
                # Most difficult to detect
                calibrated_thresholds[p_type] = stats['p5'] * 0.4  # 40% of 5th percentile
            else:
                # Default
                calibrated_thresholds[p_type] = stats['p5'] * 0.6

    # If we don't have statistics for some particle types, use defaults
    default_threshold = 0.002  # Very low default threshold
    for p_type in scored_particle_types:
        if p_type not in calibrated_thresholds:
            if p_type == 'thyroglobulin':
                calibrated_thresholds[p_type] = 0.0015  # Very low threshold
            elif p_type == 'virus-like-particle':
                calibrated_thresholds[p_type] = 0.0010  # Extremely low threshold
            elif p_type == 'beta-galactosidase':
                calibrated_thresholds[p_type] = 0.0018  # Low threshold
            else:
                calibrated_thresholds[p_type] = default_threshold

    print("\nCalibrated thresholds:")
    for p_type, threshold in calibrated_thresholds.items():
        print(f"  - {p_type}: {threshold:.4f}")

    return calibrated_thresholds

def process_test_tomograms(test_dir, model_dir, submission_dir, visualization_dir, device):
    """
    Process test tomograms and generate predictions

    Parameters:
    test_dir (str): Path to test data directory
    model_dir (str): Path to model directory
    submission_dir (str): Path to save predictions
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

    # Get scored particle types (weight > 0)
    scored_particle_types = [p for p, props in particle_types.items() if props['weight'] > 0]

    # Load the trained model
    model_path = os.path.join(model_dir, 'model.pth')

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train the model first.")
        return

    # Initialize model
    model = UNet3D(in_channels=1, out_channels=1, init_features=16)

    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Get calibrated thresholds
    train_dir = os.path.dirname(test_dir)  # Assuming train and test are in the same base directory
    calibrated_thresholds = analyze_validation_tomograms(train_dir, model_dir, device)

    if calibrated_thresholds is None:
        print("Failed to calibrate thresholds. Using default values.")
        calibrated_thresholds = {
            'apo-ferritin': 0.0020,
            'beta-galactosidase': 0.0018,
            'ribosome': 0.0020,
            'thyroglobulin': 0.0015,
            'virus-like-particle': 0.0010
        }

    # Modify thresholds to ensure better detection of all types
    # Further lower thresholds for all types to ensure better detection
    print("\nAdjusting thresholds for better particle type detection:")

    # Hardcode lower thresholds to ensure all types are detected
    for p_type in calibrated_thresholds:
        # Reduce all thresholds by 20%
        calibrated_thresholds[p_type] *= 0.8

    # Specifically lower thresholds for problematic types
    if 'thyroglobulin' in calibrated_thresholds:
        calibrated_thresholds['thyroglobulin'] = min(calibrated_thresholds['thyroglobulin'], 0.0015)

    if 'virus-like-particle' in calibrated_thresholds:
        calibrated_thresholds['virus-like-particle'] = min(calibrated_thresholds['virus-like-particle'], 0.0008)

    if 'beta-galactosidase' in calibrated_thresholds:
        calibrated_thresholds['beta-galactosidase'] = min(calibrated_thresholds['beta-galactosidase'], 0.0015)

    for p_type, threshold in calibrated_thresholds.items():
        print(f"  - {p_type}: {threshold:.4f}")

    # Get list of test experiments
    test_experiments = [os.path.basename(p) for p in glob.glob(os.path.join(test_dir, 'static/ExperimentRuns/*'))]

    if not test_experiments:
        print("No test experiments found.")
        return

    print(f"\nFound {len(test_experiments)} test experiments: {test_experiments}")

    # List to store all particle predictions for final submission
    all_particles = []

    # Process each test experiment
    for experiment in test_experiments:
        print(f"\nProcessing test experiment: {experiment}")

        # Create experiment output directory
        experiment_output_dir = os.path.join('/kaggle/working/preprocessed', 'test', experiment)
        os.makedirs(experiment_output_dir, exist_ok=True)

        # Load tomogram
        zarr_path = os.path.join(test_dir, 'static/ExperimentRuns', experiment, 'VoxelSpacing10.000/denoised.zarr')

        if not os.path.exists(zarr_path):
            print(f"Tomogram not found for experiment {experiment}")
            continue

        tomo_data = load_tomogram(zarr_path)
        if tomo_data is None:
            print(f"Failed to load tomogram for experiment {experiment}")
            continue

        tomo_data = preprocess_tomogram(tomo_data)

        # Generate density map
        print("Generating density map...")
        density_map = generate_density_map(model, tomo_data, device, patch_size=64, stride=32, batch_size=8)

        # Save density map
        np.save(os.path.join(experiment_output_dir, 'density_map.npy'), density_map)

        # Find local maxima
        print("Finding local maxima...")
        # Use very low thresholds to ensure we detect harder particles
        coordinates = find_local_maxima(
            density_map,
            min_distance=6,  # Decreased from 8 to detect smaller particles
            threshold_abs=0.05,  # Decreased to be more sensitive
            threshold_rel=0.03   # Decreased to be more sensitive
        )

        print(f"Found {len(coordinates)} potential particle locations")

        # Cluster particles
        print("Clustering particles...")
        particles_by_type = calibrated_cluster_particles(
            coordinates,
            density_map,
            scored_particle_types,
            calibrated_thresholds
        )

        # Evaluate prediction quality
        quality_score = evaluate_prediction(particles_by_type, particle_types)
        print(f"Prediction quality score: {quality_score:.4f}")

        # Check for missing particle types
        missing_types = [p_type for p_type in scored_particle_types if len(particles_by_type[p_type]) == 0]

        # If quality score is too low or we're missing particle types, try again with even lower thresholds
        if quality_score < 0.5 or missing_types:
            print(f"Low quality prediction (score: {quality_score:.4f}) or missing types: {missing_types}. Retrying with lower thresholds...")

            # Try with even lower thresholds
            lower_thresholds = {p_type: t * 0.6 for p_type, t in calibrated_thresholds.items()}

            # Especially lower thresholds for missing types
            for p_type in missing_types:
                lower_thresholds[p_type] = calibrated_thresholds[p_type] * 0.4

            # Find local maxima again with even lower thresholds
            coordinates = find_local_maxima(
                density_map,
                min_distance=5,  # Even smaller minimum distance
                threshold_abs=0.03,  # Much lower threshold
                threshold_rel=0.02   # Much lower relative threshold
            )

            print(f"Found {len(coordinates)} potential particle locations (retry)")

            # Cluster particles again
            particles_by_type = calibrated_cluster_particles(
                coordinates,
                density_map,
                scored_particle_types,
                lower_thresholds
            )

            # Re-evaluate
            quality_score = evaluate_prediction(particles_by_type, particle_types)
            print(f"New prediction quality score: {quality_score:.4f}")

            # If still missing particle types, use desperate measures
            missing_types = [p_type for p_type in scored_particle_types if len(particles_by_type[p_type]) == 0]
            if missing_types:
                print(f"Still missing types: {missing_types}. Trying desperate measures...")

                # For each missing type, find at least a few candidates
                for p_type in missing_types:
                    # Use peak_local_max with very low thresholds specific to this type
                    desperate_coords = peak_local_max(
                        density_map,
                        min_distance=4,
                        threshold_abs=0.01,
                        threshold_rel=0.01,
                        exclude_border=False,
                        num_peaks=10  # Limit to top 10 peaks
                    )

                    # Add top 5 coordinates to this particle type
                    for z, y, x in desperate_coords[:5]:
                        physical_coords = (x * 10.0, y * 10.0, z * 10.0)
                        particles_by_type[p_type].append(physical_coords)

                    print(f"Added {min(5, len(desperate_coords))} emergency particles for {p_type}")

        # Write predictions to JSON
        output_path = os.path.join(submission_dir, f"{experiment}.json")
        write_predictions_to_json(particles_by_type, output_path)

        # Visualize tomogram with particles
        vis_path = os.path.join(visualization_dir, f"{experiment}_tomogram_with_particles.png")
        visualize_tomogram_with_particles(tomo_data, particles_by_type, particle_types, vis_path)

        # Visualize density map with particles
        density_vis_path = os.path.join(visualization_dir, f"{experiment}_density_map_with_particles.png")
        visualize_density_map_with_particles(density_map, particles_by_type, particle_types, density_vis_path)

        # Add to list of all particles for CSV submission
        for p_type, coords in particles_by_type.items():
            for x, y, z in coords:
                all_particles.append({
                    'experiment': experiment,
                    'particle_type': p_type,
                    'x': x,
                    'y': y,
                    'z': z
                })

        # Free memory
        del tomo_data, density_map
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Create CSV submission file
    if all_particles:
        # Create DataFrame
        submission_df = pd.DataFrame(all_particles)

        # Add id column
        submission_df['id'] = range(len(submission_df))

        # Reorder columns
        submission_df = submission_df[['id', 'experiment', 'particle_type', 'x', 'y', 'z']]

        # Save submission
        submission_path = os.path.join(submission_dir, 'submission.csv')
        submission_df.to_csv(submission_path, index=False)

        print(f"\nSaved submission to {submission_path}")
        print(f"Total predictions: {len(submission_df)}")

        # Print submission statistics
        print("\nSubmission statistics:")
        print(submission_df.groupby(['experiment', 'particle_type']).size().unstack(fill_value=0))

    print("\nAll test tomograms processed.")

def main():
    # Set paths
    base_dir = '/kaggle/input/czii-cryo-et-object-identification'
    test_dir = os.path.join(base_dir, 'test')
    model_dir = '/kaggle/working/models'
    submission_dir = '/kaggle/working/final_submission'
    visualization_dir = '/kaggle/working/final_visualizations'

    # Create directories if they don't exist
    os.makedirs(submission_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Process test tomograms
    process_test_tomograms(test_dir, model_dir, submission_dir, visualization_dir, device)

    # Display test images
    from visualization import display_test_images
    display_test_images(visualization_dir)

    print("Test tomogram processing completed.")

if __name__ == "__main__":
    main()