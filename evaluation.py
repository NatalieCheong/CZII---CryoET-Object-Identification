import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist

def calculate_fbeta(precision, recall, beta=4):
    """
    Calculate F-beta score from precision and recall values.

    Parameters:
    precision (float): Precision value
    recall (float): Recall value
    beta (float): Beta value (defaults to 4 as per competition requirements)

    Returns:
    float: F-beta score
    """
    if precision == 0 and recall == 0:
        return 0

    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

def is_true_positive(pred_coords, true_coords, particle_type, particle_radii, radius_factor=0.5):
    """
    Determine if a predicted particle is a true positive.

    Parameters:
    pred_coords (tuple): Coordinates of the predicted particle (x, y, z)
    true_coords (list): List of coordinates of true particles
    particle_type (str): Type of particle
    particle_radii (dict): Dictionary mapping particle types to their radii
    radius_factor (float): Factor of particle radius for matching (0.5 means within half radius)

    Returns:
    bool: True if the predicted particle is a true positive, False otherwise
    int: Index of the matched true particle if found, -1 otherwise
    """
    if not true_coords:
        return False, -1

    # Get particle radius
    particle_radius = particle_radii.get(particle_type, 60)  # Default to 60 Angstroms if unknown

    # Calculate the distance threshold
    threshold = particle_radius * radius_factor

    # Convert to numpy arrays for vectorized operations
    pred_coords_array = np.array(pred_coords).reshape(1, 3)
    true_coords_array = np.array(true_coords)

    # Calculate distances to all true particles
    distances = cdist(pred_coords_array, true_coords_array)[0]

    # Find the minimum distance
    min_dist_idx = np.argmin(distances)
    min_dist = distances[min_dist_idx]

    # Check if the minimum distance is below the threshold
    if min_dist <= threshold:
        return True, min_dist_idx

    return False, -1

def evaluate_predictions(predictions, ground_truth, particle_types, beta=4):
    """
    Evaluate predictions against ground truth using the F-beta metric.

    Parameters:
    predictions (pd.DataFrame): DataFrame with columns 'experiment', 'particle_type', 'x', 'y', 'z'
    ground_truth (pd.DataFrame): DataFrame with columns 'experiment', 'particle_type', 'x', 'y', 'z'
    particle_types (dict): Dictionary of particle types and their properties
    beta (float): Beta value for F-beta calculation (default: 4)

    Returns:
    dict: Dictionary with evaluation results
    """
    results = {'overall': {}, 'by_type': {}, 'by_experiment': {}}

    # Get particle radii
    particle_radii = {p_type: props['radius'] for p_type, props in particle_types.items()}

    # Initialize counters
    total_tp = 0
    total_fp = 0
    total_fn = 0
    weighted_tp = 0
    weighted_fp = 0
    weighted_fn = 0

    # Initialize counters for each particle type
    type_stats = {}
    for p_type in particle_types:
        if particle_types[p_type]['weight'] > 0:  # Only consider scored particles
            type_stats[p_type] = {'tp': 0, 'fp': 0, 'fn': 0, 'weight': particle_types[p_type]['weight']}

    # Process each experiment
    experiments = ground_truth['experiment'].unique()
    for experiment in experiments:
        # Get predictions and ground truth for this experiment
        exp_pred = predictions[predictions['experiment'] == experiment]
        exp_true = ground_truth[ground_truth['experiment'] == experiment]

        # Process each particle type
        for p_type in particle_types:
            if particle_types[p_type]['weight'] == 0:
                continue  # Skip particles with weight 0 (beta-amylase)

            # Get predictions and ground truth for this type
            type_pred = exp_pred[exp_pred['particle_type'] == p_type]
            type_true = exp_true[exp_true['particle_type'] == p_type]

            # Extract coordinates
            pred_coords = type_pred[['x', 'y', 'z']].values.tolist()
            true_coords = type_true[['x', 'y', 'z']].values.tolist()

            # Count true positives, false positives, and false negatives
            tp = 0
            fp = len(pred_coords)  # Start assuming all predictions are false positives
            fn = len(true_coords)  # Start assuming all true particles are false negatives

            # Track which true particles have been matched
            matched_true = [False] * len(true_coords)

            # Check each prediction
            for pred_coord in pred_coords:
                is_tp, match_idx = is_true_positive(pred_coord, true_coords, p_type, particle_radii)

                if is_tp and not matched_true[match_idx]:
                    tp += 1
                    fp -= 1  # One less false positive
                    fn -= 1  # One less false negative
                    matched_true[match_idx] = True

            # Update type statistics
            type_stats[p_type]['tp'] += tp
            type_stats[p_type]['fp'] += fp
            type_stats[p_type]['fn'] += fn

            # Update total counters
            total_tp += tp
            total_fp += fp
            total_fn += fn

            # Update weighted counters
            weight = particle_types[p_type]['weight']
            weighted_tp += tp * weight
            weighted_fp += fp * weight
            weighted_fn += fn * weight

    # Calculate overall micro-averaged precision, recall, and F-beta
    if weighted_tp + weighted_fp > 0:
        weighted_precision = weighted_tp / (weighted_tp + weighted_fp)
    else:
        weighted_precision = 0

    if weighted_tp + weighted_fn > 0:
        weighted_recall = weighted_tp / (weighted_tp + weighted_fn)
    else:
        weighted_recall = 0

    weighted_fbeta = calculate_fbeta(weighted_precision, weighted_recall, beta)

    # Store overall results
    results['overall'] = {
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn,
        'weighted_true_positives': weighted_tp,
        'weighted_false_positives': weighted_fp,
        'weighted_false_negatives': weighted_fn,
        'precision': weighted_precision,
        'recall': weighted_recall,
        'f_beta': weighted_fbeta
    }

    # Calculate results for each particle type
    for p_type, stats in type_stats.items():
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']

        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0

        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0

        fbeta = calculate_fbeta(precision, recall, beta)

        results['by_type'][p_type] = {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f_beta': fbeta,
            'weight': stats['weight']
        }

    return results

def visualize_fbeta():
    """
    Create visualizations to understand the F-beta metric with beta=4.
    """
    # Create a grid of precision and recall values
    precision_values = np.linspace(0.01, 1.0, 100)
    recall_values = np.linspace(0.01, 1.0, 100)
    P, R = np.meshgrid(precision_values, recall_values)

    # Calculate F-beta for each precision-recall pair
    F_beta = np.zeros_like(P)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            F_beta[i, j] = calculate_fbeta(P[i, j], R[i, j], beta=4)

    # 3D surface plot of F-beta
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(P, R, F_beta, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    ax.set_zlabel('F-beta (beta=4)')
    ax.set_title('F-beta Metric (beta=4) for Different Precision and Recall Values')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()

    # Contour plot of F-beta
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(P, R, F_beta, 20, cmap='viridis')
    plt.colorbar(contour, label='F-beta (beta=4)')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Contour Plot of F-beta Metric (beta=4)')
    plt.grid(True, alpha=0.3)

    # Add some contour lines with labels
    contour_lines = plt.contour(P, R, F_beta, levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                              colors='white', linestyles='dashed')
    plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
    plt.tight_layout()
    plt.show()

    # Plot F-beta for fixed precision or recall values
    plt.figure(figsize=(12, 6))

    # For fixed precision values
    plt.subplot(1, 2, 1)
    for precision in [0.2, 0.4, 0.6, 0.8, 1.0]:
        fbeta_values = [calculate_fbeta(precision, r, beta=4) for r in recall_values]
        plt.plot(recall_values, fbeta_values, label=f'Precision = {precision:.1f}')

    plt.xlabel('Recall')
    plt.ylabel('F-beta (beta=4)')
    plt.title('F-beta for Fixed Precision Values')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # For fixed recall values
    plt.subplot(1, 2, 2)
    for recall in [0.2, 0.4, 0.6, 0.8, 1.0]:
        fbeta_values = [calculate_fbeta(p, recall, beta=4) for p in precision_values]
        plt.plot(precision_values, fbeta_values, label=f'Recall = {recall:.1f}')

    plt.xlabel('Precision')
    plt.ylabel('F-beta (beta=4)')
    plt.title('F-beta for Fixed Recall Values')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Compare different beta values
    plt.figure(figsize=(10, 6))

    # Fixed recall of 0.8
    recall = 0.8
    for beta in [0.5, 1, 2, 4, 8]:
        fbeta_values = [calculate_fbeta(p, recall, beta=beta) for p in precision_values]
        plt.plot(precision_values, fbeta_values, label=f'Beta = {beta}')

    plt.xlabel('Precision')
    plt.ylabel('F-beta')
    plt.title(f'F-beta Metrics for Different Beta Values (Recall = {recall})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def create_sample_data():
    """
    Create sample prediction and ground truth data for demonstration.
    """
    # Ground truth
    ground_truth_data = []

    # Experiment 1: 20 particles of each type (except beta-amylase)
    for _ in range(20):
        for p_type in ['apo-ferritin', 'beta-galactosidase', 'ribosome', 'thyroglobulin', 'virus-like-particle']:
            x = np.random.uniform(100, 1000)
            y = np.random.uniform(100, 1000)
            z = np.random.uniform(50, 150)
            ground_truth_data.append({'experiment': 'TS_5_4', 'particle_type': p_type, 'x': x, 'y': y, 'z': z})

    # Experiment 2: 15 particles of each type
    for _ in range(15):
        for p_type in ['apo-ferritin', 'beta-galactosidase', 'ribosome', 'thyroglobulin', 'virus-like-particle']:
            x = np.random.uniform(100, 1000)
            y = np.random.uniform(100, 1000)
            z = np.random.uniform(50, 150)
            ground_truth_data.append({'experiment': 'TS_6_4', 'particle_type': p_type, 'x': x, 'y': y, 'z': z})

    # Create ground truth DataFrame
    ground_truth_df = pd.DataFrame(ground_truth_data)

    # Create predictions with some noise and missing particles
    predictions_data = []

    # Copy 80% of ground truth with some noise
    for idx, row in ground_truth_df.iterrows():
        if np.random.random() < 0.8:  # 80% chance of detecting the particle
            noise_x = np.random.normal(0, 10)  # Add some noise
            noise_y = np.random.normal(0, 10)
            noise_z = np.random.normal(0, 5)
            predictions_data.append({
                'experiment': row['experiment'],
                'particle_type': row['particle_type'],
                'x': row['x'] + noise_x,
                'y': row['y'] + noise_y,
                'z': row['z'] + noise_z
            })

    # Add some false positives (10% of the total)
    num_false_positives = int(0.1 * len(ground_truth_df))
    for _ in range(num_false_positives):
        experiment = np.random.choice(['TS_5_4', 'TS_6_4'])
        p_type = np.random.choice(['apo-ferritin', 'beta-galactosidase', 'ribosome', 'thyroglobulin', 'virus-like-particle'])
        x = np.random.uniform(100, 1000)
        y = np.random.uniform(100, 1000)
        z = np.random.uniform(50, 150)
        predictions_data.append({'experiment': experiment, 'particle_type': p_type, 'x': x, 'y': y, 'z': z})

    # Create predictions DataFrame
    predictions_df = pd.DataFrame(predictions_data)

    return predictions_df, ground_truth_df

def visualize_evaluation_results(results):
    """
    Visualize evaluation results

    Parameters:
    results (dict): Dictionary with evaluation results from evaluate_predictions
    """
    # Extract results by particle type
    particle_types = list(results['by_type'].keys())
    precision_values = [results['by_type'][p]['precision'] for p in particle_types]
    recall_values = [results['by_type'][p]['recall'] for p in particle_types]
    fbeta_values = [results['by_type'][p]['f_beta'] for p in particle_types]

    # Bar chart of precision, recall, and F-beta by particle type
    plt.figure(figsize=(12, 6))
    x = np.arange(len(particle_types))
    width = 0.25

    plt.bar(x - width, precision_values, width, label='Precision')
    plt.bar(x, recall_values, width, label='Recall')
    plt.bar(x + width, fbeta_values, width, label='F-beta')

    plt.xlabel('Particle Type')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F-beta by Particle Type')
    plt.xticks(x, particle_types, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Print overall metrics
    print("\nOverall Evaluation Metrics:")
    print(f"Precision: {results['overall']['precision']:.4f}")
    print(f"Recall: {results['overall']['recall']:.4f}")
    print(f"F-beta (beta=4): {results['overall']['f_beta']:.4f}")

    # Print metrics by particle type
    print("\nMetrics by Particle Type:")
    for p_type, stats in results['by_type'].items():
        print(f"\n{p_type}:")
        print(f"  Precision: {stats['precision']:.4f}")
        print(f"  Recall: {stats['recall']:.4f}")
        print(f"  F-beta: {stats['f_beta']:.4f}")
        print(f"  TP: {stats['true_positives']}, FP: {stats['false_positives']}, FN: {stats['false_negatives']}")

def main():
    # Define particle types with properties for demonstration
    particle_types = {
        'apo-ferritin': {'difficulty': 'easy', 'weight': 1, 'radius': 60, 'color': 'red'},
        'beta-amylase': {'difficulty': 'impossible', 'weight': 0, 'radius': 45, 'color': 'yellow'},
        'beta-galactosidase': {'difficulty': 'hard', 'weight': 2, 'radius': 80, 'color': 'green'},
        'ribosome': {'difficulty': 'easy', 'weight': 1, 'radius': 100, 'color': 'blue'},
        'thyroglobulin': {'difficulty': 'hard', 'weight': 2, 'radius': 85, 'color': 'purple'},
        'virus-like-particle': {'difficulty': 'easy', 'weight': 1, 'radius': 120, 'color': 'orange'}
    }

    # Visualize the F-beta metric properties
    print("Visualizing F-beta metric properties...")
    visualize_fbeta()

    # Demonstrate the evaluation with sample data
    print("\nDemonstrating the evaluation metric with sample data...")
    predictions, ground_truth = create_sample_data()
    print(f"Ground truth shape: {ground_truth.shape}")
    print(f"Predictions shape: {predictions.shape}")

    # Evaluate the predictions
    results = evaluate_predictions(predictions, ground_truth, particle_types, beta=4)

    # Visualize the results
    visualize_evaluation_results(results)

if __name__ == "__main__":
    main()
