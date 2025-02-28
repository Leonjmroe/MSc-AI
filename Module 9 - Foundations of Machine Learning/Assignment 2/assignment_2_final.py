
# --------------------------------  
# ------- Imports / Globals ------
# --------------------------------  

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
import os

def load_mnist_data():

    # Load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # Normalise images to be in the range [-1, 1]
    X_train = X_train / 127.5 - 1
    X_test = X_test / 127.5 - 1

    # Convert each 28x28 image into a 784 dimensional vector
    features_count = np.prod(X_train.shape[1:])
    X_train_flatened = X_train.reshape(n_train, features_count)
    X_test_flatened = X_test.reshape(n_test, features_count)

    return X_train_flatened, X_test_flatened, y_train, y_test




# ----------------------  
# ------- Task 1 -------
# ----------------------  


def pca_plot(X_train_flatened, y_train):

    # Reduce the dimensionality of the data to 2 dimensions
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_flatened)

    # Create a scatter plot of the PCA data, colored by digit
    pca_fig = px.scatter(X_train_pca, x=0, y=1, color=y_train, title='PCA plot of the MNIST Dataset', width=1000, height=600)
    pca_fig.update_layout(xaxis_title='PC1', yaxis_title='PC2')

    # Create a DataFrame with the PCA data and digit labels
    df_pca = pd.DataFrame(X_train_pca, columns=['PC1', 'PC2'])
    df_pca['digit'] = y_train

    # Compute centroids for each class by taking the mean of PC1 and PC2
    centroids = df_pca.groupby('digit')[['PC1', 'PC2']].mean()

    # Plot the centroids of the pca components
    centroids_fig = px.scatter(centroids, x='PC1', y='PC2', color=centroids.index, title='PCA Centroids of the MNIST Dataset', width=1000, height=600)
    centroids_fig.update_traces(marker=dict(size=20))

    # Show plots
    pca_fig.show()
    centroids_fig.show()

    return X_train_pca

def define_digit_colors():
    return [
        'red',         # 0
        'orange',      # 1
        'yellow',      # 2
        'green',       # 3
        'cyan',        # 4
        'blue',        # 5
        'indigo',      # 6
        'violet',      # 7
        'magenta',     # 8
        'brown'        # 9
    ]

def prepare_pca_data(sample_size):
    # Load data using the consistent function
    X_train_flatened, X_test_flatened, y_train, y_test = load_mnist_data()

    # Use only a subset to reduce computation
    indices = np.random.choice(len(X_train_flatened), sample_size, replace=False)
    X_train_subset = X_train_flatened[indices]
    y_train_subset = y_train[indices]

    # Standardise and apply PCA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_subset)
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)

    return X_train_pca, y_train_subset

def plot_digit_centroids(x_train_pca, y_train_subset, digit_colors):

    plt.figure(figsize=(10, 8))

    # Calculate centroids for each digit
    for digit in range(10):
        points = x_train_pca[y_train_subset == digit]
        if len(points) > 0:
            centroid = np.mean(points, axis=0)
            plt.scatter(centroid[0], centroid[1],
                       s=100, color=digit_colors[digit], edgecolor='black', linewidth=1)
            plt.text(centroid[0], centroid[1], str(digit),
                    ha='center', va='center', color='black', fontweight='bold')

    plt.title("MNIST PCA - Digit Centroids")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def calculate_all_regressions(x_train_pca, y_train_subset):
    slopes = {}
    vectors = {}

    for digit in range(10):
        # Extract points for this digit
        points = x_train_pca[y_train_subset == digit]
        if len(points) > 0:
            # Fit regression
            reg = LinearRegression()
            X = points[:, 0].reshape(-1, 1)
            y = points[:, 1]
            reg.fit(X, y)

            # Store slope and vector
            slope = reg.coef_[0]
            intercept = reg.intercept_
            slopes[digit] = (slope, intercept)

            # Create and normalise direction vector
            vec = np.array([1, slope])
            vectors[digit] = vec / np.linalg.norm(vec)

    return slopes, vectors

def create_angle_matrix(vectors):
    angle_matrix = np.zeros((10, 10))

    for i in range(10):
        if i not in vectors:
            continue

        for j in range(10):
            if j not in vectors:
                continue

            # Cosine similarity between direction vectors
            cos_sim = np.dot(vectors[i], vectors[j])

            # Calculate angle in degrees (0-90 degrees range)
            angle = math.degrees(math.acos(min(max(cos_sim, -1.0), 1.0)))

            # For proper angle representation, we want the smallest angle between lines
            # If angle > 90 degrees, take 180-angle instead (the supplementary angle)
            if angle > 90:
                angle = 180 - angle

            angle_matrix[i, j] = angle

    # Create a dataframe for the angle matrix
    angle_df = pd.DataFrame(
        angle_matrix,
        index=[f'Digit {i}' for i in range(10)],
        columns=[f'Digit {i}' for i in range(10)]
    ).round(1)

    return angle_df

def plot_angle_heatmap(angle_df):
    plt.figure(figsize=(12, 10))
    sns.heatmap(angle_df, annot=True, cmap='YlOrRd', vmin=0, vmax=90,
               square=True, linewidths=.5, cbar_kws={"shrink": .8, "label": "Angle (degrees)"})
    plt.title('Angle Between Digit Regression Lines (degrees)', fontsize=16)
    plt.tight_layout()

    plt.show()

def plot_example_pairs(x_train_pca, y_train_subset, slopes, digit_colors, selected_pairs):
    for digit1, digit2 in selected_pairs:
        # Get points for each digit
        points1 = x_train_pca[y_train_subset == digit1]
        points2 = x_train_pca[y_train_subset == digit2]

        if len(points1) == 0 or len(points2) == 0:
            continue

        # Get regression parameters
        slope1, intercept1 = slopes[digit1]
        slope2, intercept2 = slopes[digit2]

        # Create the plot
        plt.figure(figsize=(10, 8))

        # Plot the points
        plt.scatter(points1[:, 0], points1[:, 1], color=digit_colors[digit1],
                   alpha=0.3, s=10, label=f'Digit {digit1}')
        plt.scatter(points2[:, 0], points2[:, 1], color=digit_colors[digit2],
                   alpha=0.3, s=10, label=f'Digit {digit2}')

        # Add regression lines
        x1_range = np.linspace(points1[:, 0].min(), points1[:, 0].max(), 100)
        y1_pred = slope1 * x1_range + intercept1
        plt.plot(x1_range, y1_pred, color='black', linewidth=2)

        x2_range = np.linspace(points2[:, 0].min(), points2[:, 0].max(), 100)
        y2_pred = slope2 * x2_range + intercept2
        plt.plot(x2_range, y2_pred, color='black', linewidth=2)

        # Calculate vector similarity
        vec1 = np.array([1, slope1])
        vec2 = np.array([1, slope2])

        norm_vec1 = vec1 / np.linalg.norm(vec1)
        norm_vec2 = vec2 / np.linalg.norm(vec2)

        cos_sim = np.dot(norm_vec1, norm_vec2)
        angle = math.degrees(math.acos(min(max(cos_sim, -1.0), 1.0)))

        # Take the smaller angle if > 90 degrees
        if angle > 90:
            angle = 180 - angle

        # Add annotation with angle info
        plt.annotate(f'Angle between lines: {angle:.1f}°',
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

        plt.title(f'Regression Comparison: Digit {digit1} vs Digit {digit2}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()

def run_task1(digit_pairs):

    # Load data for basic PCA visualisation if needed
    X_train_flatened, X_test_flatened, y_train, y_test = load_mnist_data()
    pca_plot(X_train_flatened, y_train)

    # Initialize colors for consistent visualisation
    digit_colors = define_digit_colors()

    # Prepare data for enhanced PCA analysis
    x_train_pca, y_train_subset = prepare_pca_data(sample_size=10000)

    # Plot digit centroids in PCA space
    plot_digit_centroids(x_train_pca, y_train_subset, digit_colors)

    # Calculate regression lines for all digits
    slopes, vectors = calculate_all_regressions(x_train_pca, y_train_subset)

    # Create and plot angle matrix
    angle_df = create_angle_matrix(vectors)
    plot_angle_heatmap(angle_df)

    # Plot example digit pairs with regression lines
    plot_example_pairs(x_train_pca, y_train_subset, slopes, digit_colors, digit_pairs)







# ----------------------  
# ------- Task 2 -------
# ----------------------  


def predict(x, w, b):
    z = np.dot(x, w) + b
    prediction = np.sign(z)
    prediction[prediction == 0] = 1
    return prediction

def optimise(x, y, w, b, max_iter, tol, learning_rate):
    n, m = x.shape

    # Initialise weights if none provided
    if w is None:
        w = np.random.randn(m) * 0.01
    if b is None:
        b = np.random.randn() * 0.01

    iter_count = 0
    error_history = []
    error = float('inf')

    while iter_count < max_iter and error > tol:
        y_pred = predict(x, w, b)
        error = np.mean(y_pred != y)
        error_history.append(error)

        # If error is low enough, break
        if error <= tol:
            break

        # Perceptron update for misclassified samples
        misclassified_indices = np.where(y_pred != y)[0]
        for i in misclassified_indices:
            w += learning_rate * y[i] * x[i]
            b += learning_rate * y[i]

        iter_count += 1

    return w, b, error_history

def evaluate_perceptron(x_train, y_train, x_test, y_test, max_iter, tol, lr):
    w, b, error_history = optimise(
        x_train, y_train,
        w=None,
        b=None,
        max_iter=max_iter,
        tol=tol,
        learning_rate=lr
    )

    # Evaluate on training set
    y_train_pred = predict(x_train, w, b)
    train_accuracy = np.mean(y_train_pred == y_train)

    # Evaluate on test set
    y_test_pred = predict(x_test, w, b)
    test_accuracy = np.mean(y_test_pred == y_test)

    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'weights': w,
        'bias': b,
        'error_history': error_history
    }

def visualise_weights(w, shape, digit_pair):
    plt.figure(figsize=(12, 5))
    weight_img = w.reshape(shape)

    # Plot raw weights heatmap
    plt.subplot(1, 2, 1)
    im = plt.imshow(weight_img, cmap='viridis')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f'Weights for {digit_pair[0]} vs {digit_pair[1]}')

    # Create a composite image:
    #   Red for positive weights (digit2),
    #   Blue for negative weights (digit1).
    composite = np.zeros((*shape, 3))
    pos_weights = np.copy(weight_img)
    neg_weights = np.copy(weight_img)

    # Zero out the opposite side so that
    #   pos_weights shows only >0
    #   neg_weights shows only <0
    pos_weights[pos_weights < 0] = 0
    neg_weights[neg_weights > 0] = 0
    neg_weights = np.abs(neg_weights)

    if pos_weights.max() > 0:
        composite[:, :, 0] = pos_weights / pos_weights.max()  # Red channel
    if neg_weights.max() > 0:
        composite[:, :, 2] = neg_weights / neg_weights.max()  # Blue channel

    plt.subplot(1, 2, 2)
    plt.imshow(composite)
    plt.title(f'Composite: (Blue={digit_pair[0]}, Red={digit_pair[1]})')

    plt.tight_layout()
    plt.show()

def prepare_binary_data(digit1, digit2, X_train, y_train, X_test, y_test):
    # Filter MNIST to keep only digit1 and digit2
    # Label them -1 (digit1) and +1 (digit2)

    # Training
    cond = (y_train == digit1) | (y_train == digit2)
    x_train_bin = X_train[cond]
    y_train_bin = y_train[cond].astype(float)
    y_train_bin[y_train_bin == digit1] = -1
    y_train_bin[y_train_bin == digit2] = +1

    # Test
    cond_test = (y_test == digit1) | (y_test == digit2)
    x_test_bin = X_test[cond_test]
    y_test_bin = y_test[cond_test].astype(float)
    y_test_bin[y_test_bin == digit1] = -1
    y_test_bin[y_test_bin == digit2] = +1

    return x_train_bin, y_train_bin, x_test_bin, y_test_bin

def run_digit_pair_experiments(digit_pairs, X_train, y_train, X_test, y_test, max_iter, tol, lr, sample_size):

    # For each (digit1, digit2) in digit_pairs, run the perceptron

    results = {}

    for (digit1, digit2) in digit_pairs:
        print(f"\n--- Perceptron for digit pair {digit1} vs {digit2} ---")

        # Prepare data
        x_train_bin, y_train_bin, x_test_bin, y_test_bin = prepare_binary_data(
            digit1, digit2, X_train, y_train, X_test, y_test
        )

        # If we want to sample a subset for faster training:
        if sample_size and len(x_train_bin) > sample_size:
            indices = np.random.choice(len(x_train_bin), sample_size, replace=False)
            x_train_bin = x_train_bin[indices]
            y_train_bin = y_train_bin[indices]

        # Evaluate
        result = evaluate_perceptron(
            x_train_bin, y_train_bin,
            x_test_bin, y_test_bin,
            max_iter=max_iter, tol=tol, lr=lr
        )
        results_key = f"{digit1}_vs_{digit2}"
        results[results_key] = result

        # Print accuracies
        print(f"  Training Accuracy: {result['train_accuracy']:.4f}")
        print(f"  Test Accuracy: {result['test_accuracy']:.4f}")

        # Plot error curve
        plt.figure(figsize=(8, 5))
        plt.plot(result['error_history'], marker='o')
        plt.title(f'Error Curve: {digit1} vs {digit2}')
        plt.xlabel('Iteration')
        plt.ylabel('Misclassification Error')
        plt.grid(True)
        plt.show()

        # Visualise weights
        visualise_weights(result['weights'], shape=(28, 28), digit_pair=(digit1, digit2))

    return results

def visualise_experiment_results(results):
    pairs = []
    train_accs = []
    test_accs = []
    iterations = []

    for pair_key, res in results.items():
        pairs.append(pair_key)
        train_accs.append(res['train_accuracy'])
        test_accs.append(res['test_accuracy'])
        iterations.append(len(res['error_history']))

    df_results = pd.DataFrame({
        'Digit Pair': pairs,
        'Train Accuracy': train_accs,
        'Test Accuracy': test_accs,
        'Iterations': iterations
    })

    print("\n----- Perceptron Results -----")
    print(df_results)

    # Bar chart
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(pairs))

    plt.bar(index, train_accs, bar_width, label='Train Acc', color='skyblue')
    plt.bar(index + bar_width, test_accs, bar_width, label='Test Acc', color='orange')

    plt.xlabel('Digit Pairs')
    plt.ylabel('Accuracy')
    plt.title('Perceptron Performance')
    plt.xticks(index + bar_width / 2, pairs, rotation=45)
    plt.ylim(0.5, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df_results

def run_task2(digit_pairs, max_iter=1000, tol=1e-3, learning_rate=0.01):
    
      X_train, X_test, y_train, y_test = load_mnist_data()
      print(f"Loaded MNIST: X_train={X_train.shape}, X_test={X_test.shape}")

      # Run experiments on each digit pair
      results = run_digit_pair_experiments(
          digit_pairs,
          X_train,
          y_train,
          X_test,
          y_test,
          max_iter=max_iter,
          tol=tol,
          lr=learning_rate,
          sample_size=10000
      )

      # Summarise
      df_results = visualise_experiment_results(results)

      return df_results





# ----------------------  
# ------- Task 3 -------
# ----------------------  


def create_mlp(input_shape, hidden_units, output_units):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_shape,)))

    for units in hidden_units:
        model.add(layers.Dense(units, activation='relu'))

    model.add(layers.Dense(output_units, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_and_evaluate_mlp(model, x_train, y_train, x_test, y_test, batch_size, epochs, verbose):
    """
    Train and evaluate an MLP model.
    
    Parameters:
    - verbose: Controls the verbosity of the output during training
               0 = silent, 1 = progress bar, 2 = one line per epoch
    """
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        verbose=verbose
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=0)

    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'history': history
    }

def plot_training_curves(history, model_name):
    # Plot the training and validation accuracy curves
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def compare_mlp_architectures(architectures, x_train, y_train, x_test, y_test, epochs, batch_size, verbose):

    results = {}

    for name, hidden_layers in architectures.items():
        print(f"Training {name} with architecture {hidden_layers}...")

        # Build the model
        model = create_mlp(input_shape=x_train.shape[1],
                           hidden_units=hidden_layers,
                           output_units=10)
        # Train
        history_info = train_and_evaluate_mlp(
            model, x_train, y_train, x_test, y_test,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose
        )

        trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])

        results[name] = {
            'train_accuracy': history_info['train_accuracy'],
            'test_accuracy': history_info['test_accuracy'],
            'parameters': trainable_params,
            'model': model,
            'history': history_info['history']
        }

        print(f"  Train accuracy: {results[name]['train_accuracy']:.4f}")
        print(f"  Test accuracy: {results[name]['test_accuracy']:.4f}")
        print(f"  Parameters: {results[name]['parameters']:,}\n")
        
        # Plot individual training curves for each architecture
        plot_training_curves(history_info['history'], model_name=name)

    return results

def plot_mlp_comparison(results, architectures):

    # Build a table
    data = {
        'MLP': [],
        'Hidden Layers': [],
        'Parameters': [],
        'Train Accuracy': [],
        'Test Accuracy': []
    }

    for name, res in results.items():
        data['MLP'].append(name)
        data['Hidden Layers'].append(len(architectures[name]))
        data['Parameters'].append(res['parameters'])
        data['Train Accuracy'].append(res['train_accuracy'])
        data['Test Accuracy'].append(res['test_accuracy'])

    # Plot 1 - Accuracy vs #Layers
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(data['Hidden Layers'], data['Train Accuracy'], 'o-', label='Train Acc')
    plt.plot(data['Hidden Layers'], data['Test Accuracy'], 's-', label='Test Acc')
    plt.xlabel('Number of Hidden Layers')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of Hidden Layers')
    plt.xticks(data['Hidden Layers'])
    plt.grid(True)
    plt.legend()

    # Plot 2 - Accuracy vs #Parameters (log)
    plt.subplot(1, 2, 2)
    plt.plot(data['Parameters'], data['Train Accuracy'], 'o-', label='Train Acc')
    plt.plot(data['Parameters'], data['Test Accuracy'], 's-', label='Test Acc')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of Parameters')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    # Plot all training histories on a single plot
    plt.figure(figsize=(12, 6))
    for name, res in results.items():
        plt.plot(res['history'].history['val_accuracy'], label=f"{name} Test")
    plt.title('Test Accuracy Across All Architectures')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Create and display a detailed table
    df_results = pd.DataFrame(data)
   
    # Create a more detailed table with all metrics
    detailed_data = {
        'MLP': [],
        'Architecture': [],
        'Hidden Layers': [],
        'Parameters': [],
        'Train Accuracy': [],
        'Test Accuracy': [],
        'Train Loss': [],
        'Test Loss': []
    }
    
    for name, res in results.items():
        detailed_data['MLP'].append(name)
        detailed_data['Architecture'].append(str(architectures[name]))
        detailed_data['Hidden Layers'].append(len(architectures[name]))
        detailed_data['Parameters'].append(f"{res['parameters']:,}")
        detailed_data['Train Accuracy'].append(f"{res['train_accuracy']:.4f}")
        detailed_data['Test Accuracy'].append(f"{res['test_accuracy']:.4f}")
        detailed_data['Train Loss'].append(f"{res['history'].history['loss'][-1]:.4f}")
        detailed_data['Test Loss'].append(f"{res['history'].history['val_loss'][-1]:.4f}")
    
    detailed_df = pd.DataFrame(detailed_data)
  
    return df_results, detailed_df

def run_task3(mlp_architectures, epochs, batch_size=50):

    X_train, X_test, y_train, y_test = load_mnist_data()
    print(f"MNIST loaded: X_train={X_train.shape}, X_test={X_test.shape}")

    # Convert labels to one-hot
    y_train_one_hot = keras.utils.to_categorical(y_train, 10)
    y_test_one_hot = keras.utils.to_categorical(y_test, 10)

    indices = np.random.choice(len(X_train), 10000, replace=False)
    X_train = X_train[indices]
    y_train_one_hot = y_train_one_hot[indices]

    # Train first model: 2-layer MLP with 1000 units each
    print("\n--- Training MLP [1000,1000] ---")
    model1 = create_mlp(input_shape=X_train.shape[1], hidden_units=[1000, 1000], output_units=10)
    result1 = train_and_evaluate_mlp(
        model1, X_train, y_train_one_hot, X_test, y_test_one_hot,
        batch_size=batch_size, epochs=epochs, verbose=1
    )
    print(f"Final Training Accuracy: {result1['train_accuracy']:.4f}")
    print(f"Final Test Accuracy:     {result1['test_accuracy']:.4f}")
    plot_training_curves(result1['history'], model_name="MLP [1000,1000]")

    # Train second model: 5-layer MLP with 500 units each
    print("\n--- Training MLP [500 x 5] ---")
    model2 = create_mlp(input_shape=X_train.shape[1], hidden_units=[500]*5, output_units=10)
    result2 = train_and_evaluate_mlp(
        model2, X_train, y_train_one_hot, X_test, y_test_one_hot,
        batch_size=batch_size, epochs=epochs, verbose=1
    )
    print(f"Final Training Accuracy: {result2['train_accuracy']:.4f}")
    print(f"Final Test Accuracy:     {result2['test_accuracy']:.4f}")
    plot_training_curves(result2['history'], model_name="MLP [500 x 5]")

    # Compare all architectures
    print("\n--- Comparing All MLP Architectures ---")
    comparison_results = compare_mlp_architectures(
        mlp_architectures,
        X_train,
        y_train_one_hot,
        X_test,
        y_test_one_hot,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Plot comparison and display tables
    summary_df, detailed_df = plot_mlp_comparison(comparison_results, mlp_architectures)
    
    # Plot all test accuracies for final comparison
    plt.figure(figsize=(14, 7))
    
    # Add the first two models
    plt.plot(result1['history'].history['val_accuracy'], label="MLP [1000,1000]")
    plt.plot(result2['history'].history['val_accuracy'], label="MLP [500 x 5]")
    
    # Add all other architectures
    for name, res in comparison_results.items():
        plt.plot(res['history'].history['val_accuracy'], label=f"{name}")
    
    # Display the detailed table
    print("\nDetailed Comparison of All MLP Architectures:")
    display(detailed_df)




# ----------------------  
# ------- Task 4 -------
# ----------------------  

def prepare_cnn_data():

    # Load data
    X_train_flattened, X_test_flattened, y_train, y_test = load_mnist_data()

    # Determine sizes
    n_train = len(y_train)
    n_test = len(y_test)

    # Reshape from (28*28,) back to (28,28,1)
    # Convert from [-1,1] to [0,1]
    x_train = ((X_train_flattened.reshape((n_train, 28, 28, 1))) + 1) / 2
    x_test = ((X_test_flattened.reshape((n_test, 28, 28, 1))) + 1) / 2

    # One-hot encode labels
    y_train_oh = tf.keras.utils.to_categorical(y_train, 10)
    y_test_oh = tf.keras.utils.to_categorical(y_test, 10)

    return x_train, y_train_oh, x_test, y_test_oh

def create_cnn(input_shape, filters, output_units):
    model = Sequential()

    # First convolutional layer (stride=1)
    model.add(Conv2D(filters[0], kernel_size=(4, 4), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape))

    # Subsequent convolutional layers (stride=2)
    for f in filters[1:]:
        model.add(Conv2D(f, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu'))

    model.add(Flatten())
    model.add(Dense(output_units, activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_cnn_model(model, x_train, y_train, x_test, y_test, batch_size, epochs, verbose):

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        verbose=verbose
    )

    # Evaluate final train/test accuracy
    train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=0)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    return {
        'model': model,
        'history': history,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
    }

def plot_cnn_results(history):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def create_and_train_cnn(name, filters, x_train, y_train, x_test, y_test, batch_size, epochs):

    # Clear previous models from memory/session:
    tf.keras.backend.clear_session()

    model = create_cnn(input_shape=x_train.shape[1:], filters=filters, output_units=10)
    model._name = name
    result = train_cnn_model(model, x_train, y_train, x_test, y_test, batch_size=batch_size, epochs=epochs, verbose=1)

    # Count trainable parameters
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])

    result.update({
        'name': name,
        'parameters': trainable_params
    })
    return result

def compare_architectures(architectures, x_train, y_train, x_test, y_test, batch_size, epochs):

    results = {}
    for name, filters in architectures.items():
        print(f"\nTraining {name} with filters {filters} ...")
        results[name] = create_and_train_cnn(
            name, filters, x_train, y_train, x_test, y_test, batch_size, epochs
        )
        print(f"  -> Train accuracy: {results[name]['train_accuracy']:.4f}")
        print(f"  -> Test accuracy:  {results[name]['test_accuracy']:.4f}")
        print(f"  -> Parameters:    {results[name]['parameters']:,}\n")
        
        # Plot each CNN's results
        plot_cnn_results(results[name]['history'])
        
    return results

def plot_comparison(results, architectures):

    results_table = {
        'CNN': [],
        'Layers': [],
        'Parameters': [],
        'Train Accuracy': [],
        'Test Accuracy': []
    }

    for name, result in results.items():
        results_table['CNN'].append(name)
        results_table['Layers'].append(len(architectures[name]))
        results_table['Parameters'].append(result['parameters'])
        results_table['Train Accuracy'].append(result['train_accuracy'])
        results_table['Test Accuracy'].append(result['test_accuracy'])

    # Plot - Accuracy vs #Layers
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(results_table['Layers'], results_table['Test Accuracy'], 's-', label='Test Accuracy')
    plt.xlabel('Number of Convolutional Layers')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of Convolutional Layers')
    plt.legend()
    plt.grid(True)
    plt.xticks(results_table['Layers'])

    # Plot - Accuracy vs #Parameters (log)
    plt.subplot(1, 2, 2)
    plt.plot(results_table['Parameters'], results_table['Test Accuracy'], 's-', label='Test Accuracy')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of Parameters')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
    # Create a combined plot with all CNNs
    plt.figure(figsize=(12, 8))
    
    # Use different markers and colors for each CNN
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_table['CNN'])))
    
    for i, cnn_name in enumerate(results_table['CNN']):
        plt.scatter(results_table['Parameters'][i], results_table['Test Accuracy'][i], 
                   marker=markers[i % len(markers)], color=colors[i], s=100, 
                   label=f"{cnn_name}")
    
    plt.xscale('log')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Test Accuracy')
    plt.title('CNN Performance Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # Create a line chart comparing all CNNs
    plt.figure(figsize=(14, 7))
    
    # Sort CNNs by parameter count for better visualization
    sorted_indices = np.argsort(results_table['Parameters'])
    sorted_names = [results_table['CNN'][i] for i in sorted_indices]
    sorted_params = [results_table['Parameters'][i] for i in sorted_indices]
    sorted_test_acc = [results_table['Test Accuracy'][i] for i in sorted_indices]
    
    plt.plot(sorted_names, sorted_test_acc, 's-', linewidth=2, label='Test Accuracy')
    
    plt.xlabel('CNN Architecture')
    plt.ylabel('Accuracy')
    plt.title('CNN Performance Comparison (Ordered by Parameter Count)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Plot all training histories on a single plot
    plt.figure(figsize=(12, 6))
    for name, res in results.items():
        plt.plot(res['history'].history['val_accuracy'], label=f"{name}")
    plt.title('Test Accuracy Across All Architectures')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Display summary table
    summary_df = pd.DataFrame({
        'CNN': results_table['CNN'],
        'Layers': results_table['Layers'],
        'Parameters': results_table['Parameters'],
        'Train Accuracy': results_table['Train Accuracy'],
        'Test Accuracy': results_table['Test Accuracy']
    })
    
    print("\nCNN Architecture Comparison Summary:")
    display(summary_df)

def run_task4(cnn_architectures, epochs, batch_size=50):

    # Prepare data
    x_train, y_train, x_test, y_test = prepare_cnn_data()

    # Build base CNN (3 conv layers by default)
    print("\n--- Base CNN Model ---")
    base_cnn = create_cnn(input_shape=(28,28,1), filters=[32, 64, 128], output_units=10)
    base_cnn.summary()

    # Train the base CNN model
    print("\n--- Training Base CNN Model ---")
    base_result = train_cnn_model(base_cnn, x_train, y_train, x_test, y_test, 
                                  batch_size=batch_size, epochs=epochs, verbose=1)

    # Print final base CNN accuracy
    print(f"\nBase CNN -> Training accuracy: {base_result['train_accuracy']:.4f}")
    print(f"Base CNN -> Test accuracy:     {base_result['test_accuracy']:.4f}")

    # Plot learning curves
    plot_cnn_results(base_result['history'])

    # Compare multiple architectures
    print("\n--- Comparing CNN Architectures ---")

    results = compare_architectures(cnn_architectures,
                                   x_train, y_train,
                                   x_test, y_test,
                                   batch_size=batch_size, epochs=epochs)

    # Plot comparison and display summary table
    plot_comparison(results, cnn_architectures)





# ----------------------  
# ------- Task 5 -------
# ----------------------  


def plot_filters(model, layer_idx, cols):
    layer = model.layers[layer_idx]

    if not isinstance(layer, layers.Conv2D):
        print(f"Layer {layer_idx} is not a convolutional layer.")
        return

    filters, biases = layer.get_weights()
    if len(filters) == 0:
        print(f"Layer {layer.name} has not been initialised with weights yet.")
        return

    n_filters, height, width, channels = filters.shape
    rows = int(np.ceil(n_filters / cols))

    plt.figure(figsize=(cols * 2, rows * 2))

    for i in range(n_filters):
        plt.subplot(rows, cols, i + 1)
        filter_img = np.mean(filters[i, :, :, :], axis=2)
        filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-7)
        plt.imshow(filter_img, cmap='viridis')
        plt.axis('off')
        plt.title(f'Filter {i+1}')

    plt.suptitle(f'Filters from layer {layer.name}')
    plt.tight_layout()
    plt.show()

def plot_activation_maps(model, image, layer_indices, digit_class, cols):
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)

    # Display input image
    plt.figure(figsize=(5, 5))
    plt.imshow(np.squeeze(image), cmap='gray')
    plt.title(f'Input Image: Digit {digit_class}')
    plt.axis('off')
    plt.show()

    # Make sure the model has been built
    if not model.built:
        model.build(image.shape)

    # Create models for each layer
    for layer_idx in layer_indices:
        layer = model.layers[layer_idx]

        # Create a temporary input layer if needed
        if not hasattr(model, '_is_graph_network') or not model._is_graph_network:
            # For Sequential models that haven't been called
            temp_input = layers.Input(shape=image.shape[1:])

            # Create a new model that goes from input to the target layer
            x = temp_input
            for i in range(layer_idx + 1):
                x = model.layers[i](x)

            activation_model = models.Model(inputs=temp_input, outputs=x)
        else:
            # For models with defined inputs/outputs
            activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer.output)

        # Get activations
        activations = activation_model.predict(image)
        n_filters = activations.shape[-1]
        rows = int(np.ceil(n_filters / cols))

        plt.figure(figsize=(cols * 2, rows * 2))

        for i in range(n_filters):
            if i < activations.shape[-1]:
                plt.subplot(rows, cols, i + 1)
                activation = activations[0, :, :, i]
                activation = (activation - activation.min()) / (activation.max() - activation.min() + 1e-7)
                plt.imshow(activation, cmap='viridis')
                plt.axis('off')
                plt.title(f'Filter {i+1}')

        plt.suptitle(f'Activation maps from layer {layer.name} for Digit {digit_class}')
        plt.tight_layout()
        plt.show()

def generate_deep_dream(model, class_idx, iterations, step_size, octave_scale, num_octaves):

    img = np.random.normal(size=(28, 28, 1)) * 0.1

    # Define loss function
    @tf.function
    def calc_loss(image, class_idx):
        image = tf.cast(image, tf.float32)
        pred = model(image)
        return pred[:, class_idx]

    # Define gradient ascent step
    @tf.function
    def gradient_ascent_step(image, class_idx, step_size):
        with tf.GradientTape() as tape:
            tape.watch(image)
            loss = calc_loss(image, class_idx)
        gradient = tape.gradient(loss, image)
        gradient = tf.math.l2_normalize(gradient)
        image = image + gradient * step_size
        return image

    # Process octaves
    original_shape = img.shape[:-1]
    img = np.expand_dims(img, axis=0)

    for octave in range(num_octaves):
        octave_shape = tuple(np.array(original_shape) * octave_scale**(num_octaves - octave - 1))
        octave_shape = tuple(map(int, octave_shape)) + (1,)

        resized_img = tf.image.resize(img, octave_shape[:-1])

        for i in range(iterations):
            resized_img = gradient_ascent_step(resized_img, class_idx, step_size)

        img = tf.image.resize(resized_img, original_shape)

    dream_img = img[0].numpy()
    dream_img = (dream_img - dream_img.min()) / (dream_img.max() - dream_img.min())

    return dream_img

def visualise_deep_dream_simpler(model, class_indices, input_shape):
   
    if not model.built:
        model.build((None,) + input_shape)

    plt.figure(figsize=(len(class_indices) * 5, 5))

    for i, class_idx in enumerate(class_indices):
        try:
            img = tf.random.normal((1,) + input_shape) * 0.1
            img = tf.Variable(img)

            learning_rate = 0.1
            steps = 100

            for step in range(steps):
                with tf.GradientTape() as tape:
                    pred = model(img)
                    loss = -tf.math.log(pred[0, class_idx] + 1e-7)

                grads = tape.gradient(loss, img)
                img.assign_sub(grads * learning_rate)

                if step % 10 == 0:
                    img_np = img.numpy()
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-7)
                    img.assign(tf.convert_to_tensor(img_np, dtype=tf.float32))

            plt.subplot(1, len(class_indices), i + 1)
            dream_img = np.squeeze(img.numpy())
            plt.imshow(dream_img, cmap='viridis')
            plt.title(f"Deep Dream: Digit '{class_idx}'")
            plt.axis('off')
        except Exception as e:
            plt.subplot(1, len(class_indices), i + 1)
            plt.text(0.5, 0.5, f"Deep Dream generation failed:\n{str(e)}",
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')

    plt.tight_layout()
    plt.show()

def visualise_cnn_outcomes(model, x_test, y_test, cols, class_indices, input_shape):
 
    if not model.built:
        model.build((None, 28, 28, 1))

    model.summary()

    print("\nVisualising Filters:")
    conv_layer_indices = [i for i, layer in enumerate(model.layers)
                          if isinstance(layer, layers.Conv2D)]

    for layer_idx in conv_layer_indices:
        print(f"Layer {layer_idx}: {model.layers[layer_idx].name}")
        plot_filters(model, layer_idx, cols)

    # Find examples of digits '2' and '9'
    digit_2_idx = np.where(np.argmax(y_test, axis=1) == 2)[0][0]
    digit_9_idx = np.where(np.argmax(y_test, axis=1) == 9)[0][0]

    digit_2_img = x_test[digit_2_idx]
    digit_9_img = x_test[digit_9_idx]

    print("\nVisualising Activation Maps for Digit '2':")
    plot_activation_maps(model, digit_2_img, conv_layer_indices, 2, cols)

    print("\nVisualising Activation Maps for Digit '9':")
    plot_activation_maps(model, digit_9_img, conv_layer_indices, 9, cols)

    print("\nGenerating Deep Dream Images:")
    visualise_deep_dream_simpler(model, class_indices, input_shape)

    print("\nDeep Dream Analysis:")
    print("The deep dream images show patterns the model is sensitive to for each digit class.")

def run_task5(digit_pairs, epochs, cols=8, batch_size=50):

    # Load data using the consistent function
    X_train_flatened, X_test_flatened, y_train, y_test_original = load_mnist_data()

    # Prepare data for CNN (converting from [-1, 1] to [0, 1])
    n_train = len(y_train)
    n_test = len(y_test_original)

    x_train = (X_train_flatened.reshape((n_train, 28, 28, 1)) + 1) / 2
    x_test = (X_test_flatened.reshape((n_test, 28, 28, 1)) + 1) / 2

    # Convert labels to one-hot encoding
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test_original, 10)

    print("Creating and training a simple CNN model...")
    model = create_cnn()

    # Build the model explicitly with the input shape
    model.build((None, 28, 28, 1))

    # Train with a small subset for demonstration
    model.fit(x_train[:5000], y_train[:5000],validation_data=(x_test[:1000], y_test[:1000]), epochs=epochs, batch_size=batch_size, verbose=1)

    visualise_cnn_outcomes(model, x_test, y_test, cols, digit_pairs, (28, 28, 1))




# ----------------------  
# ------- Task 6 -------
# ----------------------  


def load_fashion_mnist_data():
    (train_X, train_y_1), (test_X, test_y_1) = keras.datasets.fashion_mnist.load_data()
    train_X = np.expand_dims(train_X / 255.0, axis=-1)  # Normalize to [0,1] and add channel
    test_X = np.expand_dims(test_X / 255.0, axis=-1)

    def create_group_label(y):
        group_labels = np.zeros_like(y)
        group_labels[np.isin(y, [5, 7, 9])] = 0  # Shoes
        group_labels[np.isin(y, [3, 6, 8])] = 1  # Gendered
        group_labels[np.isin(y, [0, 1, 2, 4])] = 2  # Uni-Sex
        return group_labels

    train_y_2 = create_group_label(train_y_1)
    test_y_2 = create_group_label(test_y_1)

    train_y_1 = keras.utils.to_categorical(train_y_1, 10)
    test_y_1 = keras.utils.to_categorical(test_y_1, 10)
    train_y_2 = keras.utils.to_categorical(train_y_2, 3)
    test_y_2 = keras.utils.to_categorical(test_y_2, 3)

    return train_X, train_y_1, train_y_2, test_X, test_y_1, test_y_2

def create_single_task_cnn(input_shape, num_classes, task_name):
    model = models.Sequential(name=f"Single_{task_name}")
    model.add(layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=2, strides=2))
    model.add(layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2, strides=2))
    model.add(layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(3136, activation='relu'))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_multitask_model(input_shape, lambda_value):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    x = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    shared_dense = layers.Dense(3136, activation='relu')(x)

    # Item classification
    task1 = layers.Dense(1024, activation='relu')(shared_dense)
    task1 = layers.Dense(100, activation='relu')(task1)
    task1_output = layers.Dense(10, activation='softmax', name='task1_output')(task1)

    # Group classification
    task2 = layers.Dense(1024, activation='relu')(shared_dense)
    task2 = layers.Dense(100, activation='relu')(task2)
    task2_output = layers.Dense(3, activation='softmax', name='task2_output')(task2)

    model = keras.Model(inputs=inputs, outputs=[task1_output, task2_output])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss={'task1_output': 'categorical_crossentropy', 'task2_output': 'categorical_crossentropy'},
                  loss_weights={'task1_output': float(lambda_value), 'task2_output': 1.0 - float(lambda_value)},
                  metrics={'task1_output': 'accuracy', 'task2_output': 'accuracy'})
    return model

def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    else:
        plt.plot(history.history['task1_output_accuracy'], label='Task1 Train Acc')
        plt.plot(history.history['task2_output_accuracy'], label='Task2 Train Acc')
        plt.plot(history.history['val_task1_output_accuracy'], label='Task1 Val Acc')
        plt.plot(history.history['val_task2_output_accuracy'], label='Task2 Val Acc')
    
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    if 'loss' in history.history:
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
    else:
        plt.plot(history.history['task1_output_loss'], label='Task1 Train Loss')
        plt.plot(history.history['task2_output_loss'], label='Task2 Train Loss')
        plt.plot(history.history['val_task1_output_loss'], label='Task1 Val Loss')
        plt.plot(history.history['val_task2_output_loss'], label='Task2 Val Loss')
    
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def train_single_task_models(train_X, train_y_1, train_y_2, test_X, test_y_1, test_y_2, batch_size, epochs):
    model_task1 = create_single_task_cnn(train_X.shape[1:], 10, "Task1_Item")
    model_task2 = create_single_task_cnn(train_X.shape[1:], 3, "Task2_Group")

    print("Item Classification - Model Summary:")
    model_task1.summary()
    print("\nGroup Classification - Model Summary:")
    model_task2.summary()

    print("\nTraining Item Classification Model...")
    start_time = time.time()
    history_task1 = model_task1.fit(train_X, train_y_1, batch_size=batch_size, epochs=epochs, validation_data=(test_X, test_y_1), verbose=1)
    task1_train_time = time.time() - start_time
    task1_test_loss, task1_test_acc = model_task1.evaluate(test_X, test_y_1, verbose=0)
    print(f"Item Classification - Test Accuracy: {task1_test_acc:.4f}")
    
    # Plot training history for Task 1
    plot_training_history(history_task1, "Single Task - Item Classification")

    print("\nTraining Group Classification Model...")
    start_time = time.time()
    history_task2 = model_task2.fit(train_X, train_y_2, batch_size=batch_size, epochs=epochs,
                                    validation_data=(test_X, test_y_2), verbose=1)
    task2_train_time = time.time() - start_time
    task2_test_loss, task2_test_acc = model_task2.evaluate(test_X, test_y_2, verbose=0)
    print(f"Group Classification - Test Accuracy: {task2_test_acc:.4f}")
    
    # Plot training history for Task 2
    plot_training_history(history_task2, "Single Task - Group Classification")

    return {
        'task1': {
            'model': model_task1, 
            'accuracy': task1_test_acc, 
            'params': model_task1.count_params(), 
            'train_time': task1_train_time,
            'history': history_task1
        },
        'task2': {
            'model': model_task2, 
            'accuracy': task2_test_acc, 
            'params': model_task2.count_params(), 
            'train_time': task2_train_time,
            'history': history_task2
        }
    }

def train_multitask_models(train_X, train_y_1, train_y_2, test_X, test_y_1, test_y_2, lambda_values, batch_size, epochs):
    mtl_results = {}
    for lam in lambda_values:
        print(f"\nTraining MTL Model with λ = {lam}")
        model = create_multitask_model(train_X.shape[1:], lam)
        if lam == lambda_values[0]:
            model.summary()

        start_time = time.time()
        history = model.fit(
            train_X, 
            {'task1_output': train_y_1, 'task2_output': train_y_2},
            batch_size=batch_size, 
            epochs=epochs,
            validation_data=(test_X, {'task1_output': test_y_1, 'task2_output': test_y_2}), 
            verbose=1
        )
        train_time = time.time() - start_time

        test_results = model.evaluate(
            test_X, 
            {'task1_output': test_y_1, 'task2_output': test_y_2},
            verbose=0, 
            return_dict=True
        )
        task1_acc = test_results.get('task1_output_accuracy', 0.0)
        task2_acc = test_results.get('task2_output_accuracy', 0.0)
        print(f"λ = {lam}: Task 1 Acc = {task1_acc:.4f}, Task 2 Acc = {task2_acc:.4f}")
        
        # Plot training history for this MTL model
        plot_training_history(history, f"Multi-Task Learning (λ={lam})")

        mtl_results[lam] = {
            'model': model, 
            'task1_accuracy': task1_acc, 
            'task2_accuracy': task2_acc,
            'params': model.count_params(), 
            'train_time': train_time,
            'history': history
        }
    return mtl_results

def analyse_results(single_results, mtl_results, lambda_values):
    task1_mtl_accs = [mtl_results[lam]['task1_accuracy'] for lam in lambda_values]
    task2_mtl_accs = [mtl_results[lam]['task2_accuracy'] for lam in lambda_values]
    single_task1_acc = single_results['task1']['accuracy']
    single_task2_acc = single_results['task2']['accuracy']

    # Plot accuracies vs lambda
    plt.figure(figsize=(12, 8))
    plt.plot(lambda_values, task1_mtl_accs, 'o-', linewidth=2, label='MTL Task 1 (Item)')
    plt.plot(lambda_values, task2_mtl_accs, 's-', linewidth=2, label='MTL Task 2 (Group)')
    plt.axhline(single_task1_acc, color='r', linestyle='--', linewidth=2, label=f'Single Task 1: {single_task1_acc:.4f}')
    plt.axhline(single_task2_acc, color='g', linestyle='--', linewidth=2, label=f'Single Task 2: {single_task2_acc:.4f}')
    plt.xlabel('λ (Task 1 Loss Weight)', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('MTL Performance vs. λ', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(lambda_values)
    plt.tight_layout()
    plt.show()
    
    # Model comparison bar chart
    plt.figure(figsize=(14, 8))
    
    # Prepare data for bar chart
    models = ['Single Task 1', 'Single Task 2'] + [f'MTL λ={lam}' for lam in lambda_values]
    task1_accs = [single_task1_acc, 0] + task1_mtl_accs
    task2_accs = [0, single_task2_acc] + task2_mtl_accs
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width/2, task1_accs, width, label='Task 1 (Item) Accuracy', color='skyblue')
    rects2 = ax.bar(x + width/2, task2_accs, width, label='Task 2 (Group) Accuracy', color='lightgreen')
    
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add accuracy values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.4f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.show()

    # Results table
    print("\n### Performance Comparison ###")
    print(f"{'Model':<15} | {'Task 1 Acc':<10} | {'Task 2 Acc':<10} | {'Params':<12} | {'Train Time (s)':<12}")
    print("-" * 65)
    print(f"{'Single Task 1':<15} | {single_task1_acc:<10.4f} | {'-':<10} | {single_results['task1']['params']:<12,} | {single_results['task1']['train_time']:<12.2f}")
    print(f"{'Single Task 2':<15} | {'-':<10} | {single_task2_acc:<10.4f} | {single_results['task2']['params']:<12,} | {single_results['task2']['train_time']:<12.2f}")
    total_single_params = single_results['task1']['params'] + single_results['task2']['params']
    total_single_time = single_results['task1']['train_time'] + single_results['task2']['train_time']
    print(f"{'Single Total':<15} | {'-':<10} | {'-':<10} | {total_single_params:<12,} | {total_single_time:<12.2f}")
    print("-" * 65)
    for lam in lambda_values:
        print(f"{'MTL λ='+str(lam):<15} | {mtl_results[lam]['task1_accuracy']:<10.4f} | {mtl_results[lam]['task2_accuracy']:<10.4f} | {mtl_results[lam]['params']:<12,} | {mtl_results[lam]['train_time']:<12.2f}")
    print("-" * 65)

    # Create a pandas DataFrame for better display
    data = {
        'Model': ['Single Task 1', 'Single Task 2', 'Single Total'] + [f'MTL λ={lam}' for lam in lambda_values],
        'Task 1 Accuracy': [single_task1_acc, float('nan'), float('nan')] + [mtl_results[lam]['task1_accuracy'] for lam in lambda_values],
        'Task 2 Accuracy': [float('nan'), single_task2_acc, float('nan')] + [mtl_results[lam]['task2_accuracy'] for lam in lambda_values],
        'Parameters': [single_results['task1']['params'], single_results['task2']['params'], total_single_params] + [mtl_results[lam]['params'] for lam in lambda_values],
        'Training Time (s)': [single_results['task1']['train_time'], single_results['task2']['train_time'], total_single_time] + [mtl_results[lam]['train_time'] for lam in lambda_values]
    }
    
    summary_df = pd.DataFrame(data)
    display(summary_df)

    # Additional analysis
    param_savings = total_single_params - mtl_results[0.5]['params']
    print(f"Parameter Savings with MTL: {param_savings:,} ({param_savings/total_single_params*100:.2f}%)")

    best_lambda = max(lambda_values, key=lambda lam: (mtl_results[lam]['task1_accuracy'] + mtl_results[lam]['task2_accuracy']) / 2)
    best_avg_acc = (mtl_results[best_lambda]['task1_accuracy'] + mtl_results[best_lambda]['task2_accuracy']) / 2
    single_avg_acc = (single_task1_acc + single_task2_acc) / 2
    print(f"Best MTL Avg Accuracy (λ={best_lambda}): {best_avg_acc:.4f} vs. Single Avg: {single_avg_acc:.4f}")

    improves_both = any(mtl_results[lam]['task1_accuracy'] > single_task1_acc and
                        mtl_results[lam]['task2_accuracy'] > single_task2_acc for lam in lambda_values)
    print(f"MTL Improves Both Tasks Simultaneously: {'Yes' if improves_both else 'No'}")

def run_task6(lambda_values, epochs, batch_size=50):
    
    print(f"Parameters: batch_size={batch_size}, epochs={epochs}, lambda_values={lambda_values}")
    train_X, train_y_1, train_y_2, test_X, test_y_1, test_y_2 = load_fashion_mnist_data()
    print(f"Data loaded: train_X shape={train_X.shape}, test_X shape={test_X.shape}")

    single_results = train_single_task_models(
        train_X, train_y_1, train_y_2, 
        test_X, test_y_1, test_y_2, 
        batch_size=batch_size, 
        epochs=epochs
    )

    mtl_results = train_multitask_models(
        train_X, train_y_1, train_y_2, 
        test_X, test_y_1, test_y_2,
        lambda_values=lambda_values, 
        batch_size=batch_size, 
        epochs=epochs
    )

    analyse_results(single_results, mtl_results, lambda_values)

