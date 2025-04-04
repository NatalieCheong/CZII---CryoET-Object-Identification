import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import os

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

def visualize_tomogram_with_particles(tomo_data, particles_by_type, particle_types, output_path=None, slices=None):
    """
    Visualize tomogram slices with colored particle markers

    Parameters:
    tomo_data (numpy.ndarray): 3D tomogram data
    particles_by_type (dict): Dictionary mapping particle type to list of coordinates
    particle_types (dict): Dictionary mapping particle types to their properties
    output_path (str, optional): Path to save the visualization
    slices (list, optional): List of slice indices to visualize (default is middle slice)
    """
    # Get tomogram dimensions
    depth, height, width = tomo_data.shape

    # Choose slices if not provided
    if slices is None:
        slices = [depth // 4, depth // 2, 3 * depth // 4]

    # Create figure
    fig, axes = plt.subplots(1, len(slices), figsize=(6 * len(slices), 6))
    if len(slices) == 1:
        axes = [axes]

    # For each slice
    for i, slice_idx in enumerate(slices):
        # Show the tomogram slice
        axes[i].imshow(tomo_data[slice_idx], cmap='gray')
        axes[i].set_title(f'Z-Slice {slice_idx}/{depth}')

        # Get slice range (particles near this slice)
        slice_range = 10  # Consider particles within ±10 slices
        z_min = (slice_idx - slice_range) * 10.0  # Convert to physical coordinates
        z_max = (slice_idx + slice_range) * 10.0

        # Add circles for each particle type
        for p_type, coords in particles_by_type.items():
            # Skip if particle type not in our dictionary or no coordinates
            if p_type not in particle_types or not coords:
                continue

            # Get color and radius for this particle type
            color = particle_types[p_type]['color']
            radius = particle_types[p_type]['radius'] / 10.0  # Convert to voxel units

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

                # Calculate alpha based on distance from the slice
                z_px = z / 10.0
                alpha = 1.0 - abs(z_px - slice_idx) / slice_range

                # Add circle
                circle = plt.Circle((x_px, y_px), radius, color=color, fill=False, alpha=alpha, linewidth=1.5)
                axes[i].add_patch(circle)

        # Add legend
        axes[i].legend(loc='upper right', bbox_to_anchor=(1.1, 1))
        axes[i].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')

    plt.show()

def visualize_density_map_with_particles(density_map, particles_by_type, particle_types, output_path=None, slice_idx=None):
    """
    Visualize density map with particle locations

    Parameters:
    density_map (numpy.ndarray): 3D density map
    particles_by_type (dict): Dictionary mapping particle type to list of coordinates
    particle_types (dict): Dictionary mapping particle types to their properties
    output_path (str, optional): Path to save the visualization
    slice_idx (int, optional): Index of slice to visualize (default is middle slice)
    """
    # Get dimensions
    depth, height, width = density_map.shape

    # Choose slice if not provided
    if slice_idx is None:
        slice_idx = depth // 2

    # Create a custom colormap for density
    cmap_name = 'hot_alpha'
    colors = [(0, 0, 0, 0)]  # Start with transparent
    for i in range(1, 256):
        # Red-yellow colormap with increasing alpha
        alpha = i / 255.0
        if i < 128:
            # From transparent to red
            colors.append((i / 127.0, 0, 0, alpha * 0.7))
        else:
            # From red to yellow
            colors.append((1, (i - 128) / 127.0, 0, alpha * 0.7))

    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

    # Create figure
    plt.figure(figsize=(12, 10))

    # Show the density map slice
    plt.imshow(density_map[slice_idx], cmap=custom_cmap)

    # Get slice range (particles near this slice)
    slice_range = 10  # Consider particles within ±10 slices
    z_min = (slice_idx - slice_range) * 10.0  # Convert to physical coordinates
    z_max = (slice_idx + slice_range) * 10.0

    # Add markers for each particle type
    for p_type, coords in particles_by_type.items():
        # Skip if particle type not in our dictionary or no coordinates
        if p_type not in particle_types or not coords:
            continue

        # Get color for this particle type
        color = particle_types[p_type]['color']

        # Count particles in this slice
        slice_particles = [(x, y, z) for x, y, z in coords if z_min <= z <= z_max]
        n_particles = len(slice_particles)

        # Skip if no particles of this type in this slice
        if n_particles == 0:
            continue

        # Extract x, y coordinates for this slice
        x_coords = [x / 10.0 for x, y, z in slice_particles]
        y_coords = [y / 10.0 for x, y, z in slice_particles]

        # Plot particles
        plt.scatter(x_coords, y_coords, color=color, marker='o', s=50, facecolors='none',
                   label=f'{p_type} ({n_particles})', linewidth=1.5)

    plt.title(f'Density Map with Particles (Z-Slice {slice_idx}/{depth})', fontsize=14)
    plt.colorbar(label='Density Value')
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    plt.axis('off')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')

    plt.show()

def display_test_images(visualization_dir):
    """
    Display all test images generated in the visualization directory
    """
    import glob

    # Get all visualization files
    vis_files = glob.glob(os.path.join(visualization_dir, '*.png'))

    if not vis_files:
        print("No visualization files found in", visualization_dir)
        return

    print(f"Found {len(vis_files)} visualization files.")

    # Display each visualization file
    for vis_file in sorted(vis_files):
        filename = os.path.basename(vis_file)
        print(f"\nDisplaying: {filename}")

        # Load and display the image
        img = plt.imread(vis_file)
        plt.figure(figsize=(15, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title(filename, fontsize=14)
        plt.tight_layout()
        plt.show()

def visualize_model_predictions(model, val_loader, num_samples=5):
    """
    Visualize model predictions

    Parameters:
    model (nn.Module): Trained model
    val_loader (DataLoader): Validation data loader
    num_samples (int): Number of samples to visualize
    """
    import torch

    device = next(model.parameters()).device
    model.eval()

    # Get random samples
    samples = []
    with torch.no_grad():
        for patches, labels in val_loader:
            samples.append((patches, labels))
            if len(samples) >= num_samples:
                break

    # Visualize each sample
    for i, (patches, labels) in enumerate(samples):
        # Move data to device
        patches = patches.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(patches)

        # Move back to CPU and convert to numpy
        patches = patches.cpu().numpy()
        labels = labels.cpu().numpy()
        outputs = outputs.detach().cpu().numpy()

        # Plot middle slice of first batch item
        middle_slice = patches.shape[2] // 2

        plt.figure(figsize=(15, 5))

        # Plot patch
        plt.subplot(1, 3, 1)
        plt.imshow(patches[0, 0, middle_slice], cmap='gray')
        plt.title(f"Input Patch {i+1}")
        plt.axis('off')

        # Plot ground truth
        plt.subplot(1, 3, 2)
        plt.imshow(labels[0, 0, middle_slice], cmap='hot')
        plt.title("Ground Truth")
        plt.axis('off')

        # Plot prediction
        plt.subplot(1, 3, 3)
        plt.imshow(outputs[0, 0, middle_slice], cmap='hot')
        plt.title("Prediction")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
