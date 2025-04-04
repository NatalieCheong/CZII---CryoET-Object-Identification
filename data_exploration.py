import os
import glob
import zarr
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def explore_directory(directory, max_depth=3, current_depth=0, prefix=''):
    """
    Recursively explore a directory structure

    Parameters:
    directory (str): Path to directory
    max_depth (int): Maximum depth to explore
    current_depth (int): Current depth
    prefix (str): Prefix for indentation

    Returns:
    list: List of directory structure lines
    """
    if current_depth > max_depth:
        return [f"{prefix}..."]

    result = []
    try:
        items = os.listdir(directory)

        # Sort items to get a consistent order
        items.sort()

        for i, item in enumerate(items):
            item_path = os.path.join(directory, item)
            is_last = (i == len(items) - 1)

            # Skip certain directories or files
            if item.startswith('.') or item == '__pycache__':
                continue

            # Use different connectors for last item vs others
            connector = '└── ' if is_last else '├── '
            child_prefix = prefix + ('    ' if is_last else '│   ')

            if os.path.isdir(item_path):
                result.append(f"{prefix}{connector}{item}/")
                # Recursively explore subdirectories
                result.extend(explore_directory(item_path, max_depth, current_depth + 1, child_prefix))
            else:
                result.append(f"{prefix}{connector}{item}")
    except Exception as e:
        result.append(f"{prefix}Error: {str(e)}")

    return result

def visualize_data(data, title="Data Visualization", n_slices=3):
    """
    Visualize slices of a 3D or 4D array
    """
    if data is None:
        print("No data to visualize")
        return

    # Handle different data dimensions
    if len(data.shape) == 4:
        print(f"4D data of shape {data.shape}, taking first channel/time point")
        data = data[0]  # Take first channel/time point

    if len(data.shape) != 3:
        print(f"Cannot visualize data of shape {data.shape}")
        return

    # Get dimensions
    depth, height, width = data.shape
    print(f"Data dimensions: {depth} x {height} x {width}")

    # Choose slice indices
    slice_indices = np.linspace(depth // 4, 3 * depth // 4, n_slices).astype(int)

    # Create figure
    fig, axes = plt.subplots(1, n_slices, figsize=(5 * n_slices, 5))
    if n_slices == 1:
        axes = [axes]

    # Plot each slice
    for i, slice_idx in enumerate(slice_indices):
        im = axes[i].imshow(data[slice_idx], cmap='gray')
        axes[i].set_title(f'Slice {slice_idx}/{depth}')
        axes[i].axis('off')
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

    # Histogram of values
    plt.figure(figsize=(10, 6))
    plt.hist(data.flatten(), bins=100, alpha=0.7, color='blue')
    plt.title(f"Value Distribution - {title}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.show()

def explore_zarr(zarr_path, max_items=5):
    """
    Explore a zarr file's structure and metadata
    """
    print(f"Exploring zarr at: {zarr_path}")

    # First, explore the file system structure
    print("\nFile system structure:")
    structure = explore_directory(zarr_path, max_depth=4)
    for line in structure[:30]:  # Limit to first 30 lines
        print(line)
    if len(structure) > 30:
        print("... (truncated)")

    # Check for .zattrs file which contains metadata
    zattrs_path = os.path.join(zarr_path, '.zattrs')
    if os.path.exists(zattrs_path):
        print("\nFound .zattrs file. Contents:")
        try:
            with open(zattrs_path, 'r') as f:
                zattrs = json.load(f)
            print(json.dumps(zattrs, indent=2)[:1000] + "..." if len(json.dumps(zattrs)) > 1000 else "")

            # Check for multiscales metadata
            if 'multiscales' in zattrs:
                print("\nMultiscales metadata:")
                multiscales = zattrs['multiscales']
                print(json.dumps(multiscales, indent=2)[:1000] + "..." if len(json.dumps(multiscales)) > 1000 else "")

                # Extract dataset paths
                if multiscales and 'datasets' in multiscales[0]:
                    datasets = multiscales[0]['datasets']
                    print(f"\nDataset paths: {[d.get('path') for d in datasets]}")
        except Exception as e:
            print(f"Error reading .zattrs: {str(e)}")

    # Look for .zarray files which define arrays
    zarray_files = glob.glob(os.path.join(zarr_path, "**", ".zarray"), recursive=True)
    if zarray_files:
        print(f"\nFound {len(zarray_files)} .zarray files:")
        for i, zarray_file in enumerate(zarray_files[:max_items]):
            print(f" {i+1}. {os.path.relpath(zarray_file, zarr_path)}")
            try:
                with open(zarray_file, 'r') as f:
                    zarray = json.load(f)
                print(f" Shape: {zarray.get('shape')}")
                print(f" Chunks: {zarray.get('chunks')}")
                print(f" Data type: {zarray.get('dtype')}")
            except Exception as e:
                print(f" Error reading .zarray: {str(e)}")

    # Try to open the zarr file
    try:
        z = zarr.open(zarr_path, mode='r')

        # Check if it's a Group
        if isinstance(z, zarr.Group):
            print("\nZarr opened as a Group")
            print(f"Keys: {list(z.keys())}")

            # Explore each key
            for key in list(z.keys())[:max_items]:
                print(f"\nExploring group '{key}':")
                try:
                    item = z[key]
                    if isinstance(item, zarr.Array):
                        print(f" Array shape: {item.shape}")
                        print(f" Data type: {item.dtype}")
                        print(f" Chunks: {item.chunks}")
                    elif isinstance(item, zarr.Group):
                        print(f" Subgroup with keys: {list(item.keys())}")
                except Exception as e:
                    print(f" Error exploring group: {str(e)}")

        # Check if it's an Array
        elif isinstance(z, zarr.Array):
            print("\nZarr opened as an Array")
            print(f"Shape: {z.shape}")
            print(f"Data type: {z.dtype}")
            print(f"Chunks: {z.chunks}")

            # Print some sample data if small enough
            if np.prod(z.shape) < 100:
                print(f"Data sample: {z[:]}")
            else:
                print("Data too large to display sample")

        else:
            print(f"\nZarr opened as unknown type: {type(z)}")

    except Exception as e:
        print(f"\nError opening zarr: {str(e)}")

    return

def try_load_visualize(zarr_path):
    """
    Try different approaches to load and visualize data from a zarr file
    """
    print(f"Attempting to load and visualize data from: {zarr_path}")

    # Try approach 1: Direct array loading if zarr_path points to a zarr array
    try:
        print("\nApproach 1: Direct array loading")
        z1 = zarr.open(zarr_path, mode='r')
        if isinstance(z1, zarr.Array):
            print(f"Found array of shape: {z1.shape}")
            data1 = z1[:]
            visualize_data(data1, title="Approach 1: Direct array")
            return data1
    except Exception as e:
        print(f"Approach 1 failed: {str(e)}")

    # Try approach 2: Using multiscales metadata if available
    try:
        print("\nApproach 2: Using multiscales metadata")
        zattrs_path = os.path.join(zarr_path, '.zattrs')
        if os.path.exists(zattrs_path):
            with open(zattrs_path, 'r') as f:
                zattrs = json.load(f)

            if 'multiscales' in zattrs and 'datasets' in zattrs['multiscales'][0]:
                datasets = zattrs['multiscales'][0]['datasets']
                if datasets:
                    dataset_path = datasets[0].get('path')
                    if dataset_path:
                        full_path = os.path.join(zarr_path, dataset_path)
                        print(f"Loading dataset from: {full_path}")
                        z2 = zarr.open(full_path, mode='r')
                        if isinstance(z2, zarr.Array):
                            print(f"Found array of shape: {z2.shape}")
                            data2 = z2[:]
                            visualize_data(data2, title="Approach 2: Multiscales")
                            return data2
    except Exception as e:
        print(f"Approach 2 failed: {str(e)}")

    # Try approach 3: Find .zarray files and try to load data from their parent directories
    try:
        print("\nApproach 3: Find .zarray files and load from their parent directories")
        zarray_files = glob.glob(os.path.join(zarr_path, "**", ".zarray"), recursive=True)
        if zarray_files:
            for i, zarray_file in enumerate(zarray_files[:3]):  # Try first 3 .zarray files
                data_dir = os.path.dirname(zarray_file)
                print(f"Trying to load from: {data_dir}")
                try:
                    z3 = zarr.open(data_dir, mode='r')
                    if isinstance(z3, zarr.Array):
                        print(f"Found array of shape: {z3.shape}")
                        data3 = z3[:]
                        visualize_data(data3, title=f"Approach 3: Array {i+1}")
                        return data3
                except Exception as e:
                    print(f"Failed to load from {data_dir}: {str(e)}")
    except Exception as e:
        print(f"Approach 3 failed: {str(e)}")

    # Try approach 4: Traverse numeric subdirectories to find data
    try:
        print("\nApproach 4: Traverse numeric subdirectories")
        # Open zarr root
        z4 = zarr.open(zarr_path, mode='r')

        # Check if it's a Group
        if isinstance(z4, zarr.Group):
            # Look for numeric keys that might contain the data
            numeric_keys = [k for k in z4.keys() if k.isdigit()]
            if numeric_keys:
                print(f"Found numeric keys: {numeric_keys}")

                # Try to traverse the structure through a chain of numeric keys
                current_group = z4
                traversal_path = []

                # Try up to depth 5
                for _ in range(5):
                    numeric_keys = [k for k in current_group.keys() if k.isdigit()]
                    if not numeric_keys:
                        break

                    next_key = numeric_keys[0]  # Take first numeric key
                    traversal_path.append(next_key)

                    current_group = current_group[next_key]
                    if isinstance(current_group, zarr.Array):
                        print(f"Found array at path: {'/'.join(traversal_path)}")
                        print(f"Array shape: {current_group.shape}")
                        data4 = current_group[:]
                        visualize_data(data4, title="Approach 4: Numeric traversal")
                        return data4
    except Exception as e:
        print(f"Approach 4 failed: {str(e)}")

    print("\nAll approaches failed to load data")
    return None

def explore_json_files(json_path):
    """
    Explore the structure of a JSON file containing particle annotations
    """
    print(f"Exploring JSON file: {json_path}")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Print overall structure
        print("\nTop level keys:")
        for key in data.keys():
            if isinstance(data[key], list):
                print(f" {key}: list with {len(data[key])} items")
            elif isinstance(data[key], dict):
                print(f" {key}: dict with {len(data[key])} keys")
            else:
                print(f" {key}: {type(data[key]).__name__}")

        # Check for 'picks' which should contain particle coordinates
        if 'picks' in data:
            picks = data['picks']
            print(f"\nFound {len(picks)} picks")

            # Look at the first few picks
            if picks:
                print("\nSample pick entries:")
                for i, pick in enumerate(picks[:3]):
                    print(f" Pick {i+1}:")
                    print(json.dumps(pick, indent=4))

                # Extract a sample of the coordinates
                coords = []
                for pick in picks[:100]:  # Limit to first 100 picks
                    if 'position' in pick:
                        pos = pick['position']
                        if all(k in pos for k in ['x', 'y', 'z']):
                            coords.append((pos['x'], pos['y'], pos['z']))

                if coords:
                    print(f"\nExtracted {len(coords)} coordinates")

                    # Basic statistics
                    coords_array = np.array(coords)
                    print("\nCoordinate statistics:")
                    print(f" X: min={coords_array[:, 0].min():.2f}, max={coords_array[:, 0].max():.2f}, mean={coords_array[:, 0].mean():.2f}")
                    print(f" Y: min={coords_array[:, 1].min():.2f}, max={coords_array[:, 1].max():.2f}, mean={coords_array[:, 1].mean():.2f}")
                    print(f" Z: min={coords_array[:, 2].min():.2f}, max={coords_array[:, 2].max():.2f}, mean={coords_array[:, 2].mean():.2f}")

                    return True, coords
                else:
                    print("Could not extract valid coordinates from pick entries")
            else:
                print("No pick entries found")
        else:
            print("No 'picks' key found in JSON")

    except Exception as e:
        print(f"Error exploring JSON: {str(e)}")

    return False, []

def collect_particle_data(train_dir, particle_types):
    """
    Collect particle coordinates data from all JSON files

    Parameters:
    train_dir (str): Path to training data directory
    particle_types (dict): Dictionary of particle types and their properties

    Returns:
    list: List of particle data dictionaries
    """
    # Find all JSON files with particle annotations
    json_files = glob.glob(os.path.join(train_dir, 'overlay/ExperimentRuns/*/Picks/*.json'))
    print(f"Found {len(json_files)} JSON files")

    all_particle_data = []
    experiment_counts = {}
    particle_counts = {}
    experiments = []

    # Process all JSON files
    for json_path in json_files:
        # Extract experiment and particle type from path
        parts = json_path.split('/')
        experiment = parts[-3]
        particle_type = os.path.splitext(os.path.basename(json_path))[0]

        # Check if valid particle type
        if particle_type not in particle_types:
            continue

        # Load coordinates from JSON
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract coordinates from points
        coords = []
        if 'points' in data:
            for point in data['points']:
                if 'location' in point:
                    loc = point['location']
                    coords.append((loc.get('x', 0), loc.get('y', 0), loc.get('z', 0)))

        count = len(coords)

        # Update counts
        if experiment not in experiments:
            experiments.append(experiment)
            experiment_counts[experiment] = {}

        experiment_counts[experiment][particle_type] = count
        particle_counts[particle_type] = particle_counts.get(particle_type, 0) + count

        # Add coordinates to dataset
        for x, y, z in coords:
            all_particle_data.append({
                'experiment': experiment,
                'particle_type': particle_type,
                'x': x,
                'y': y,
                'z': z,
                'difficulty': particle_types[particle_type]['difficulty'],
                'weight': particle_types[particle_type]['weight']
            })

    # Print statistics
    print(f"Total number of experiments: {len(experiments)}")
    print(f"Total number of particles found: {len(all_particle_data)}")
    print("\nParticle distribution:")
    for particle in sorted(particle_counts.keys()):
        difficulty = particle_types.get(particle, {}).get('difficulty', 'unknown')
        weight = particle_types.get(particle, {}).get('weight', 'unknown')
        print(f" - {particle}: {particle_counts[particle]} particles (Difficulty: {difficulty}, Weight: {weight})")

    return all_particle_data, experiment_counts, experiments

def analyze_sample_submission(base_dir):
    """
    Analyze the sample submission file

    Parameters:
    base_dir (str): Path to the dataset base directory
    """
    # Load the sample submission file
    sample_submission_path = os.path.join(base_dir, 'sample_submission.csv')

    if os.path.exists(sample_submission_path):
        print(f"Loading sample submission from: {sample_submission_path}")
        sample_submission = pd.read_csv(sample_submission_path)

        # Print basic info
        print(f"\nSample submission shape: {sample_submission.shape}")
        print("\nSample submission columns:")
        for col in sample_submission.columns:
            print(f" - {col}")

        print("\nFirst 10 rows of the sample submission:")
        print(sample_submission.head(10))

        # Analyze the unique values in each column
        print("\nUnique values in each column:")
        for col in sample_submission.columns:
            unique_vals = sample_submission[col].nunique()
            print(f" - {col}: {unique_vals} unique values")

            # For columns with few unique values, print them
            if unique_vals < 10 and col != 'id':
                print(f" Values: {sorted(sample_submission[col].unique())}")

        # Check for missing values
        missing = sample_submission.isnull().sum()
        if missing.sum() > 0:
            print("\nMissing values in sample submission:")
            for col, count in missing.items():
                if count > 0:
                    print(f" - {col}: {count} missing values")
        else:
            print("\nNo missing values in the sample submission")

        # Analyze coordinates
        print("\nCoordinate statistics:")
        for col in ['x', 'y', 'z']:
            if col in sample_submission.columns:
                print(f" - {col}:")
                print(f" Min: {sample_submission[col].min()}")
                print(f" Max: {sample_submission[col].max()}")
                print(f" Mean: {sample_submission[col].mean()}")
                print(f" Std: {sample_submission[col].std()}")

        # Count rows per experiment and particle type
        if 'experiment' in sample_submission.columns and 'particle_type' in sample_submission.columns:
            exp_counts = sample_submission.groupby('experiment').size()
            print("\nSubmission entries per experiment:")
            for exp, count in exp_counts.items():
                print(f" - {exp}: {count} entries")

            type_counts = sample_submission.groupby('particle_type').size()
            print("\nSubmission entries per particle type:")
            for p_type, count in type_counts.items():
                print(f" - {p_type}: {count} entries")

            # Cross-tabulation of experiment and particle type
            cross_tab = pd.crosstab(sample_submission['experiment'], sample_submission['particle_type'])
            print("\nCross-tabulation of experiment and particle type:")
            print(cross_tab)

            # Visualize particle types in sample submission
            plt.figure(figsize=(10, 6))
            type_counts.plot(kind='bar')
            plt.title('Particle Types in Sample Submission')
            plt.xlabel('Particle Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
    else:
        print(f"Sample submission file not found at: {sample_submission_path}")

def main():
    # Set paths
    base_dir = '/kaggle/input/czii-cryo-et-object-identification'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

    # Particle types and their properties
    particle_types = {
        'apo-ferritin': {'difficulty': 'easy', 'weight': 1, 'color': 'red'},
        'beta-amylase': {'difficulty': 'impossible', 'weight': 0, 'color': 'yellow'},
        'beta-galactosidase': {'difficulty': 'hard', 'weight': 2, 'color': 'blue'},
        'ribosome': {'difficulty': 'easy', 'weight': 1, 'color': 'green'},
        'thyroglobulin': {'difficulty': 'hard', 'weight': 2, 'color': 'purple'},
        'virus-like-particle': {'difficulty': 'easy', 'weight': 1, 'color': 'orange'}
    }

    print("Exploring zarr file structure in the dataset")
    # List all denoised.zarr files in training data
    denoised_zarrs = glob.glob(os.path.join(train_dir, 'static/ExperimentRuns/*/VoxelSpacing10.000/denoised.zarr'))
    print(f"Found {len(denoised_zarrs)} denoised.zarr files in training data:")
    for i, zarr_path in enumerate(denoised_zarrs):
        parts = zarr_path.split('/')
        experiment = parts[-3]
        print(f" {i+1}. {experiment}")

    # Choose first experiment for exploration
    if denoised_zarrs:
        sample_zarr_path = denoised_zarrs[0]
        print(f"\nExploring sample zarr file: {sample_zarr_path}")

        # Explore zarr structure
        explore_zarr(sample_zarr_path)

        # Try to load and visualize data
        print("\nAttempting to load and visualize data...")
        try_load_visualize(sample_zarr_path)
    else:
        print("No denoised.zarr files found in training data")

    # Explore a test zarr file as well
    test_zarrs = glob.glob(os.path.join(test_dir, 'static/ExperimentRuns/*/VoxelSpacing10.000/denoised.zarr'))
    if test_zarrs:
        test_zarr_path = test_zarrs[0]
        print(f"\nExploring test zarr file: {test_zarr_path}")

        # Just do a basic exploration
        explore_zarr(test_zarr_path)
    else:
        print("\nNo denoised.zarr files found in test data")

    # Explore JSON particle annotations
    print("\nLooking for particle annotation JSON files...")
    particle_jsons = glob.glob(os.path.join(train_dir, 'overlay/ExperimentRuns/*/Picks/*.json'))
    print(f"Found {len(particle_jsons)} JSON files")

    # List the first few files
    for i, json_path in enumerate(particle_jsons[:10]):
        # Extract experiment and particle type from path
        parts = json_path.split('/')
        experiment = parts[-3]
        particle_type = os.path.splitext(os.path.basename(json_path))[0]
        print(f" {i+1}. {experiment} - {particle_type}")

    # Explore a JSON file
    if particle_jsons:
        json_path = particle_jsons[0]
        explore_json_files(json_path)

    # Collect particle data
    print("\nCollecting particle data...")
    all_particle_data, _, _ = collect_particle_data(train_dir, particle_types)

    # Convert to DataFrame
    if all_particle_data:
        particle_df = pd.DataFrame(all_particle_data)
        print("\nParticle coordinates DataFrame sample:")
        print(particle_df.head())

    # Analyze sample submission
    analyze_sample_submission(base_dir)

if __name__ == "__main__":
    main()
