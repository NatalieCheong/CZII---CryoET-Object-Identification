import torch
import torch.nn as nn
import torch.nn.functional as F

# 3D U-Net model for particle detection
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=16):
        super(UNet3D, self).__init__()

        features = init_features
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = self._block(features * 4, features * 8, name="bottleneck")

        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")

        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")

        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1")

        self.conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True)
        )

# Dataset class for 3D patches
class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, patches, labels):
        self.patches = patches
        self.labels = labels

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        # Add channel dimension and convert to torch tensors
        patch = torch.FloatTensor(self.patches[idx]).unsqueeze(0)
        label = torch.FloatTensor(self.labels[idx]).unsqueeze(0)

        return patch, label

# Dice loss for 3D segmentation
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        # Flatten the predictions and targets
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)

        return 1 - dice

# Focal loss for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# Combined loss function
class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()

    def forward(self, predictions, targets):
        dice = self.dice_loss(predictions, targets)
        focal = self.focal_loss(predictions, targets)

        return self.dice_weight * dice + self.focal_weight * focal

# Enhanced clustering function to detect all particle types
def calibrated_cluster_particles(coordinates, volume, particle_types, calibrated_thresholds):
    """
    Cluster particles using calibrated thresholds with a multi-pass approach to ensure all types are detected

    Parameters:
    coordinates (numpy.ndarray): Array of peak coordinates [z, y, x]
    volume (numpy.ndarray): Density map
    particle_types (list): List of particle types to consider
    calibrated_thresholds (dict): Calibrated thresholds for each particle type

    Returns:
    dict: Dictionary mapping particle type to list of coordinates
    """
    import numpy as np

    if len(coordinates) == 0:
        print("No coordinates found for clustering")
        return {p_type: [] for p_type in particle_types}

    # Sort peaks by intensity
    peak_values = np.array([volume[z, y, x] for z, y, x in coordinates])
    sorted_indices = np.argsort(peak_values)[::-1]  # Sort in descending order

    sorted_coordinates = coordinates[sorted_indices]
    sorted_values = peak_values[sorted_indices]

    # Dictionary to store particles by type
    particles_by_type = {p_type: [] for p_type in particle_types}

    # Assign peaks to particle types based on calibrated thresholds
    n_particles = {p_type: 0 for p_type in particle_types}

    # Set target and maximum number of particles per tomogram
    # Increased target numbers to ensure we find all types
    target_particles = {
        'apo-ferritin': 80,
        'beta-galactosidase': 60,
        'ribosome': 80,
        'thyroglobulin': 60,
        'virus-like-particle': 40
    }

    max_particles = {
        'apo-ferritin': 120,
        'beta-galactosidase': 100,
        'ribosome': 120,
        'thyroglobulin': 100,
        'virus-like-particle': 80
    }

    # Define size categories by radius for more balanced distribution
    size_categories = {
        'small': ['apo-ferritin'],
        'medium': ['beta-galactosidase', 'thyroglobulin'],
        'large': ['ribosome', 'virus-like-particle']
    }

    # Create reverse mapping of particle type to category
    particle_to_category = {}
    for category, p_types in size_categories.items():
        for p_type in p_types:
            particle_to_category[p_type] = category

    # Track assigned peaks
    already_assigned = set()

    # First pass: Assign top peaks to different size categories as evenly as possible
    size_counts = {cat: 0 for cat in size_categories}
    category_quotas = {
        'small': 0.30,  # 30% for small particles
        'medium': 0.40,  # 40% for medium particles
        'large': 0.30    # 30% for large particles
    }

    total_expected = min(len(sorted_coordinates), sum(target_particles.values()))
    category_limits = {
        cat: int(total_expected * quota) for cat, quota in category_quotas.items()
    }

    print("\nCategory distribution targets:")
    for cat, limit in category_limits.items():
        print(f"  - {cat}: {limit} particles")

    # First pass: Assign peaks to categories based on thresholds and type quotas
    for i, (z, y, x) in enumerate(sorted_coordinates):
        peak_value = sorted_values[i]

        # Skip if already assigned
        if i in already_assigned:
            continue

        # Skip if this peak is too close to an already assigned peak
        too_close = False
        for idx in already_assigned:
            z2, y2, x2 = sorted_coordinates[idx]
            dist = np.sqrt((z - z2)**2 + (y - y2)**2 + (x - x2)**2)
            if dist < 5:  # 5 voxels minimum distance (reduced from original 6)
                too_close = True
                break

        if too_close:
            continue

        # Try to assign to particle types that are under their limits
        assigned = False

        # Prioritize underrepresented categories
        categories_sorted = sorted(size_categories.keys(), key=lambda cat: size_counts[cat] / max(1, category_limits[cat]))

        for category in categories_sorted:
            if size_counts[category] >= category_limits[category] * 1.5:  # Allow going over by 50%
                continue  # Skip if category is well beyond limit

            # Try each particle type in this category
            p_types = size_categories[category]

            # Sort types by how far they are from their target
            p_types_sorted = sorted(p_types,
                                  key=lambda p: (n_particles[p] / max(1, target_particles[p])))

            for p_type in p_types_sorted:
                if p_type not in calibrated_thresholds:
                    continue  # Skip if we don't have a threshold

                if n_particles[p_type] >= max_particles[p_type]:
                    continue  # Skip if at limit

                # Check threshold - use a lower threshold as we go deeper into the sorted list
                threshold_factor = 1.0 - (i / len(sorted_coordinates)) * 0.2  # Gradually decrease threshold (reduced from 0.3)
                effective_threshold = calibrated_thresholds[p_type] * threshold_factor

                if peak_value >= effective_threshold:
                    # Convert voxel coordinates to physical coordinates
                    physical_coords = (x * 10.0, y * 10.0, z * 10.0)
                    particles_by_type[p_type].append(physical_coords)
                    n_particles[p_type] += 1
                    already_assigned.add(i)
                    size_counts[category] += 1
                    assigned = True
                    break

            if assigned:
                break

    # Print distribution after first pass
    print("\nParticle distribution after first pass:")
    for p_type in particle_types:
        print(f"  - {p_type}: {n_particles[p_type]} particles")

    # Check for missing or underrepresented types
    underrepresented = []
    for p_type in particle_types:
        # If we have less than 20% of target for this type, consider it underrepresented
        if n_particles[p_type] < target_particles.get(p_type, 50) * 0.2:
            underrepresented.append(p_type)

    print(f"Underrepresented types: {underrepresented}")

    # Second pass: Focus on underrepresented types with much lower thresholds
    if underrepresented:
        print("Running second pass for underrepresented types...")

        # Sort by intensity again - we'll use the remaining strong signals
        remaining_indices = [i for i in range(len(sorted_coordinates)) if i not in already_assigned]

        for i in remaining_indices:
            z, y, x = sorted_coordinates[i]
            peak_value = sorted_values[i]

            # Skip if too close to an already assigned peak
            too_close = False
            for idx in already_assigned:
                z2, y2, x2 = sorted_coordinates[idx]
                dist = np.sqrt((z - z2)**2 + (y - y2)**2 + (x - x2)**2)
                if dist < 5:  # 5 voxels minimum distance
                    too_close = True
                    break

            if too_close:
                continue

            # Try to assign to underrepresented types with very low thresholds
            for p_type in underrepresented:
                # Target at least 20% of the expected count for each type
                min_target = target_particles.get(p_type, 50) * 0.2

                if n_particles[p_type] >= min_target:
                    continue

                # Use a very low threshold - just to get some representation
                very_low_threshold = calibrated_thresholds[p_type] * 0.4  # 40% of calibrated threshold

                if peak_value >= very_low_threshold:
                    # Convert voxel coordinates to physical coordinates
                    physical_coords = (x * 10.0, y * 10.0, z * 10.0)
                    particles_by_type[p_type].append(physical_coords)
                    n_particles[p_type] += 1
                    already_assigned.add(i)

                    # Update category count
                    category = particle_to_category.get(p_type)
                    if category:
                        size_counts[category] += 1

                    break  # Assign at most one particle per coordinate

    # Print distribution after second pass
    print("\nParticle distribution after second pass:")
    for p_type in particle_types:
        print(f"  - {p_type}: {n_particles[p_type]} particles")

    # Check if we still have missing particle types
    missing_types = [p_type for p_type in particle_types if n_particles[p_type] == 0]

    # Third pass: Desperate measures for still-missing types
    if missing_types:
        print(f"Still missing types: {missing_types}. Using desperate measures.")

        # Use remaining peaks with extremely low thresholds
        remaining_indices = [i for i in range(len(sorted_coordinates)) if i not in already_assigned]

        # For each missing type, try to find at least a few candidates
        for p_type in missing_types:
            # Count needed for minimum representation
            min_count = 5  # At least 5 particles of each type

            # Find peaks for this type, sorted by intensity
            candidate_indices = []
            for i in remaining_indices:
                z, y, x = sorted_coordinates[i]
                # Use almost no threshold - we just need some representation
                candidate_indices.append((i, sorted_values[i], z, y, x))

            # Sort candidates by intensity
            candidate_indices.sort(key=lambda x: x[1], reverse=True)

            # Take top candidates that aren't too close to existing particles
            for i, value, z, y, x in candidate_indices:
                if n_particles[p_type] >= min_count:
                    break

                # Check if too close to already assigned particles
                too_close = False
                for idx in already_assigned:
                    z2, y2, x2 = sorted_coordinates[idx]
                    dist = np.sqrt((z - z2)**2 + (y - y2)**2 + (x - x2)**2)
                    if dist < 5:  # 5 voxels minimum distance
                        too_close = True
                        break

                if too_close:
                    continue

                # Convert voxel coordinates to physical coordinates
                physical_coords = (x * 10.0, y * 10.0, z * 10.0)
                particles_by_type[p_type].append(physical_coords)
                n_particles[p_type] += 1
                already_assigned.add(i)

                # Update category count
                category = particle_to_category.get(p_type)
                if category:
                    size_counts[category] += 1

            print(f"Added {n_particles[p_type]} emergency particles for {p_type}")

    # Fourth pass: Fill in with remaining high-value peaks to reach targets
    remaining_indices = [i for i in range(len(sorted_coordinates)) if i not in already_assigned]

    # For each particle type that's below target
    below_target_types = [(p, target_particles.get(p, 50) - n_particles[p])
                          for p in particle_types
                          if n_particles[p] < target_particles.get(p, 50)]

    # Sort by how far below target they are
    below_target_types.sort(key=lambda x: x[1], reverse=True)

    if below_target_types:
        print("\nRunning fourth pass to reach targets for types below target...")

        for p_type, deficit in below_target_types:
            if deficit <= 0:
                continue

            # Use lower thresholds for final pass
            final_threshold = calibrated_thresholds[p_type] * 0.3  # 30% of calibrated threshold

            # Count particles added for this type
            added = 0

            for i in remaining_indices[:]:  # Create a copy to modify during iteration
                if added >= deficit:
                    break

                z, y, x = sorted_coordinates[i]
                peak_value = sorted_values[i]

                # Skip if already assigned (could happen during this pass)
                if i in already_assigned:
                    continue

                # Skip if too close to an already assigned peak
                too_close = False
                for idx in already_assigned:
                    z2, y2, x2 = sorted_coordinates[idx]
                    dist = np.sqrt((z - z2)**2 + (y - y2)**2 + (x - x2)**2)
                    if dist < 5:  # 5 voxels minimum distance
                        too_close = True
                        break

                if too_close:
                    continue

                # Check if this peak meets our threshold
                if peak_value >= final_threshold:
                    # Convert voxel coordinates to physical coordinates
                    physical_coords = (x * 10.0, y * 10.0, z * 10.0)
                    particles_by_type[p_type].append(physical_coords)
                    n_particles[p_type] += 1
                    already_assigned.add(i)
                    added += 1

                    # Update category count
                    category = particle_to_category.get(p_type)
                    if category:
                        size_counts[category] += 1

            print(f"Added {added} additional particles for {p_type}")

    # Print final statistics
    print("\nFinal particle distribution:")
    for p_type in particle_types:
        print(f"  - {p_type}: {n_particles[p_type]} particles")

    print(f"Total particles found: {sum(n_particles.values())}")

    return particles_by_type
