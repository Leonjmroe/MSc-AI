# Imports

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA
import os



# ----------------------------------------
# --------------- Global -----------------
# ----------------------------------------

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




# ----------------------------------------
# --------------- Task 1 -----------------
# ----------------------------------------

def pca_plot(X_train_flatened, y_train, output_prefix='pca_visualization'):

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
    
    # Standardize and apply PCA
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

            # Create and normalize direction vector
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

def run_task1(sample_size=10000, selected_pairs=[(0,1), (3,8), (4,9)]):

    # Load data for basic PCA visualization if needed
    X_train_flatened, X_test_flatened, y_train, y_test = load_mnist_data()
    pca_plot(X_train_flatened, y_train)

    # Initialize colors for consistent visualization
    digit_colors = define_digit_colors()
    
    # Prepare data for enhanced PCA analysis
    x_train_pca, y_train_subset = prepare_pca_data(sample_size=sample_size)
    
    # Plot digit centroids in PCA space
    plot_digit_centroids(x_train_pca, y_train_subset, digit_colors)
    
    # Calculate regression lines for all digits
    slopes, vectors = calculate_all_regressions(x_train_pca, y_train_subset)
    
    # Create and plot angle matrix
    angle_df = create_angle_matrix(vectors)
    plot_angle_heatmap(angle_df)

    # Plot example digit pairs with regression lines
    plot_example_pairs(x_train_pca, y_train_subset, slopes, digit_colors, selected_pairs)



# ----------------------------------------
# --------------- Task 2 -----------------
# ----------------------------------------



def prepare_data_for_binary_classification(X_train, X_test, y_train, y_test, digit_1, digit_2):

    # Convert each 28x28 image into a 784 dimensional vector
    features_count = np.prod(X_train.shape[1:])
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    X_train_flatened = X_train.reshape(n_train, features_count)
    X_test_flatened = X_test.reshape(n_test, features_count)

    # Filter out for digit_1 and digit_2 for binary classification
    cond = (y_train == digit_1) + (y_train == digit_2)
    binary_x_train = X_train_flatened[cond, :]
    binary_y_train = y_train[cond] * 1.0

    # Normalise training labels
    binary_y_train[binary_y_train == digit_1] = -1
    binary_y_train[binary_y_train == digit_2] = 1

    # Filter out for digit_1 and digit_2 for binary classification
    cond_test = (y_test == digit_1) + (y_test == digit_2)
    binary_x_test = X_test_flatened[cond_test, :]
    binary_y_test = y_test[cond_test] * 1.0

    # Normalise test labels
    binary_y_test[binary_y_test == digit_1] = -1
    binary_y_test[binary_y_test == digit_2] = 1

    return binary_x_train, binary_y_train, binary_x_test, binary_y_test

def predict(x, w, b):

    # Compute the linear combination for each sample
    z = np.dot(x, w) + b

    # If z >= 0, predict 1, otherwise predict -1
    prediction = np.where(z >= 0, 1, -1)

    return prediction

def run_epoch_perceptron(binary_x_train, binary_y_train, binary_x_test, binary_y_test, num_epochs=100, learning_rate=0.01):

    def train_perceptron(x_train, y_train, num_epochs, learning_rate):

        # Get the number of samples and features
        n_samples, n_features = x_train.shape

        # Initialise weights and bias to zero
        w = np.zeros(n_features)
        b = 0.0

        # Lists to store accuracy values
        train_accuracies = []
        epochs = []

        # Batch of stochastic gradient descent
        for epoch in range(num_epochs):
            for i in range(n_samples):

                # Check if the sample is misclassified
                if y_train[i] * (np.dot(x_train[i], w) + b) <= 0:

                    # Update weights and bias using the perceptron rule
                    w += learning_rate * y_train[i] * x_train[i]
                    b += learning_rate * y_train[i]

            # Evaluate training progress at each epoch
            predictions = predict(x_train, w, b)
            accuracy = np.mean(predictions == y_train)
            train_accuracies.append(accuracy)
            epochs.append(epoch + 1)

        # Plot accuracy vs epochs
        fig = px.line(x=epochs, y=train_accuracies, title='Training Accuracy vs Epochs', labels={'x': 'Epoch', 'y': 'Accuracy'}, width=1000, height=500)
        fig.show()

        return w, b


    # Train the perceptron using the binary training data
    w, b = train_perceptron(binary_x_train, binary_y_train, num_epochs, learning_rate)

    # Predict on the training data
    train_predictions = predict(binary_x_train, w, b)
    train_accuracy = np.mean(train_predictions == binary_y_train)
    print('Final Training Accuracy:', train_accuracy)

    # Predict on the test data
    test_predictions = predict(binary_x_test, w, b)
    test_accuracy = np.mean(test_predictions == binary_y_test)
    print('Test Accuracy:', test_accuracy)

    return test_accuracy

def run_optimisation_perceptron(binary_x_train, binary_y_train, binary_x_test, binary_y_test, max_iters=1000, learning_rate=0.01, tolerance=1e-3):

    def optimise_perceptron(x, y, max_iters, learning_rate, tolerance):

        # Initialise variables
        iter = 0
        error = np.inf
        error_list = []
        n,m = x.shape
        rng = np.random.default_rng()
        w = rng.random(m)
        b = rng.random()

        # While the iteration is less than the maximum number of iterations and the error is greater than the tolerance
        while (iter <= max_iters) & (error > tolerance):

            # Predict all samples
            predictions = predict(x, w, b)

            # Identify misclassified samples
            misclassified_indices = np.where(predictions != y)[0]

            # Compute current error (fraction of misclassified samples)
            error = len(misclassified_indices) / n
            error_list.append(error)

            # If no misclassifications, we can stop early
            if len(misclassified_indices) == 0:
                break

            # Update w, b for each misclassified sample
            for i in misclassified_indices:
                w += learning_rate * y[i] * x[i]
                b += learning_rate * y[i]

            iter += 1

        return w, b, error_list


    # Optimise on the training set
    w_opt, b_opt, error_list = optimise_perceptron(binary_x_train, binary_y_train, max_iters, learning_rate, tolerance)

    # Evaluate on training
    train_pred = predict(binary_x_train, w_opt, b_opt)
    train_accuracy = np.mean(train_pred == binary_y_train)
    print('Final Training Accuracy:', train_accuracy)

    # Evaluate on test
    test_pred = predict(binary_x_test, w_opt, b_opt)
    test_accuracy = np.mean(test_pred == binary_y_test)
    print('Test Accuracy:', test_accuracy)

    # Error Curve
    df_error = pd.DataFrame({'Iteration': list(range(1, len(error_list) + 1)), 'Misclassification Error': error_list})
    fig_error = px.line(df_error, x='Iteration', y='Misclassification Error', title='Perceptron Training Error', markers=True, width=1000, height=500)
    fig_error.show()

    # Visualise the learned weights as an image
    w_image = w_opt.reshape(28, 28)
    fig_weights = px.imshow(w_image, color_continuous_scale='RdBu', title='Learned Weight Image', width=1000, height=500)
    fig_weights.show()

    return test_accuracy

def run_perceptron(X_train, X_test, y_train, y_test):

    digits = {'sample_1': (1, 0), 'sample_2': (8, 3), 'sample_3': (4, 9), 'sample_4': (8, 7), 'sample_5': (2, 9)}

    results = {}

    for run, (digit_1, digit_2) in enumerate(digits.values()):
        print(f'\n\nRun: {run + 1 }: -- Training for digits {digit_1} and {digit_2} --\n\n')
        print(' -- Epoch Perceptron Training --\n')
        binary_x_train, binary_y_train, binary_x_test, binary_y_test = prepare_data_for_binary_classification(X_train, X_test, y_train, y_test, digit_1, digit_2)
        epoch_test_accuracy = run_epoch_perceptron(binary_x_train, binary_y_train, binary_x_test, binary_y_test)
        print('\n -- Optimisation Perceptron Training --\n')
        optimisation_test_accuracy = run_optimisation_perceptron(binary_x_train, binary_y_train, binary_x_test, binary_y_test)
        results[f'run_{run + 1}'] = {'digit_1': round(digit_1, 0), 'digit_2': round(digit_2, 0), 'epoch_test_accuracy': round(epoch_test_accuracy, 2), 'optimisation_test_accuracy': round(optimisation_test_accuracy, 2)}

    df = pd.DataFrame(results)

    return df

def run_task2():
    try:
        print("Starting Task 2...")
        # Load and prepare data
        X_train_flatened, X_test_flatened, y_train, y_test = load_mnist_data()
        
        # Task 1: PCA visualization
        print("\n\n--- Task 1: PCA Visualization ---\n")
        pca_plot(X_train_flatened, y_train)
        
        # Task 2: Perceptron
        print("\n\n--- Task 2: Perceptron Binary Classification ---\n")
        results_df = run_perceptron(X_train_flatened, X_test_flatened, y_train, y_test)
        print("\nSummary of Perceptron Results:")
        print(results_df)
        
        # Task 3: MLP
        print("\n\n--- Task 3: Multi-Layer Perceptron ---\n")
        run_mlp(X_train_flatened, X_test_flatened, y_train, y_test)
        
        print("Task 2 completed successfully!")
        return results_df
    except Exception as e:
        print(f"Error in Task 2: {e}")
        import traceback
        traceback.print_exc()
        return None




# ----------------------------------------
# --------------- Task 3 -----------------
# ----------------------------------------


def run_mlp(X_train, X_test, y_train, y_test):
    # Prepare data
    # Convert labels to one-hot encoding
    y_train_one_hot = to_categorical(y_train, 10)
    y_test_one_hot = to_categorical(y_test, 10)
    
    # Reshape data for MLP (already flattened)
    X_train_normalized = X_train.copy()
    X_test_normalized = X_test.copy()
    
    # First MLP Model
    print("\nTraining first MLP model (2 hidden layers with 1000 neurons each)")
    model1 = Sequential([
        Dense(1000, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(1000, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model1.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    history1 = model1.fit(
        X_train_normalized, y_train_one_hot,
        batch_size=50,
        epochs=10,
        validation_data=(X_test_normalized, y_test_one_hot)
    )

    print("\nFirst MLP Model Results:")
    plot(history1)

    # Second MLP Model
    print("\nTraining second MLP model (5 hidden layers with 500 neurons each)")
    model2 = Sequential([
        Dense(500, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(500, activation='relu'),
        Dense(500, activation='relu'),
        Dense(500, activation='relu'),
        Dense(500, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model2.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history2 = model2.fit(
        X_train_normalized, y_train_one_hot,
        batch_size=50,
        epochs=10,
        validation_data=(X_test_normalized, y_test_one_hot)
    )

    print("\nSecond MLP Model Results:")
    plot(history2)

def plot(history):
    train_acc = history.history['accuracy'][-1] * 100
    test_acc = history.history['val_accuracy'][-1] * 100
    print(f"Training accuracy: {train_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.title('Training vs. Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def run_task3():
    try:
        print("Starting Task 3...")
        # Load and prepare data
        X_train_flatened, X_test_flatened, y_train, y_test = load_mnist_data()
        
        # Run MLP models
        print("\n--- Running Multi-Layer Perceptron Models ---")
        results = run_mlp(X_train_flatened, X_test_flatened, y_train, y_test)
        
        print("Task 3 completed successfully!")
        return results
    except Exception as e:
        print(f"Error in Task 3: {e}")
        import traceback
        traceback.print_exc()
        return None




# ----------------------------------------
# --------------- Task 4 -----------------
# ----------------------------------------  


def plot_cnn_results(history):
    train_acc = history.history['accuracy'][-1] * 100
    test_acc = history.history['val_accuracy'][-1] * 100
    print(f'Training accuracy: {train_acc:.2f}%')
    print(f'Test accuracy: {test_acc:.2f}%')

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.title('Training vs. Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def prepare_cnn_data():
    """Load and prepare MNIST data for CNN models."""
    # Load data using the consistent function
    X_train_flatened, X_test_flatened, y_train, y_test = load_mnist_data()
    
    # Get original dimensions
    n_train = len(y_train)
    n_test = len(y_test)
    
    # Reshape the flattened data back to images for CNN
    # Note: load_mnist_data normalizes to [-1, 1], but we need [0, 1] for CNN models
    # So we transform from [-1, 1] to [0, 1]
    x_train = (X_train_flatened.reshape((n_train, 28, 28, 1)) + 1) / 2
    x_test = (X_test_flatened.reshape((n_test, 28, 28, 1)) + 1) / 2

    # One-hot encode the labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return x_train, y_train, x_test, y_test

def build_base_cnn_model():
    # Build the model
    model = Sequential([
        Conv2D(32, kernel_size=(4, 4), strides=(1, 1), activation='relu', input_shape=(28, 28, 1)),
        Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu'),
        Conv2D(128, kernel_size=(4, 4), strides=(2, 2), activation='relu'),
        Flatten(),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model

def train_cnn_model(model, x_train, y_train, x_test, y_test, batch_size=50, epochs=10):
    # Train the model
    history = model.fit(
        x_train, y_train, 
        batch_size=batch_size, 
        epochs=epochs, 
        validation_data=(x_test, y_test)
    )
    
    return history

def run_task4():
    try:
        print("Starting Task 4...")
        print("\n--- Running Convolutional Neural Network Model ---")
        
        # Prepare data
        x_train, y_train, x_test, y_test = prepare_cnn_data()
        
        # Build and train base CNN model
        print("\nBase CNN Model:")
        model = build_base_cnn_model()
        history = train_cnn_model(model, x_train, y_train, x_test, y_test)
        
        # Plot results
        plot_cnn_results(history)
        
        print("Task 4 completed successfully!")
        return history
    except Exception as e:
        print(f"Error in Task 4: {e}")
        import traceback
        traceback.print_exc()
        return None





# ----------------------------------------
# ------------ NAJIB Task 2 --------------
# ----------------------------------------  

# Task 2.1: Implement the predict function for the perceptron
def predict(x, w, b):
    """Predict class labels for samples in x using the perceptron model."""
    z = np.dot(x, w) + b
    prediction = np.sign(z)
    prediction[prediction == 0] = 1
    return prediction

# Task 2.2: Implement the optimize function for training the perceptron
def optimize(x, y, w=None, b=None, max_iter=1000, tol=1e-3, learning_rate=0.01):
    """Train a perceptron model using the perceptron learning algorithm."""
    n, m = x.shape
    
    # Initialize weights and bias if not provided
    if w is None:
        w = np.random.randn(m) * 0.01
    if b is None:
        b = np.random.randn() * 0.01
    
    iter_count = 0
    error_history = []
    error = float('inf')
    
    # Main training loop
    while iter_count < max_iter and error > tol:
        y_pred = predict(x, w, b)
        error = np.mean(y_pred != y)
        error_history.append(error)
        
        if error <= tol:
            break
            
        for i in range(n):
            if y_pred[i] != y[i]:
                w = w + learning_rate * y[i] * x[i]
                b = b + learning_rate * y[i]
                
        if iter_count % 200 == 0:
            print(f"Iteration {iter_count}, Error: {error:.4f}, Weight norm: {np.linalg.norm(w):.4f}")
            
        iter_count += 1
    
    print(f"Training completed after {iter_count} iterations")
    print(f"Final error: {error:.4f}")
    
    return w, b, error_history

# Function to evaluate perceptron on test data
def evaluate_perceptron(x_train, y_train, x_test, y_test):
    """Train a perceptron and evaluate it on test data."""
    w, b, error_history = optimize(x_train, y_train)
    
    y_train_pred = predict(x_train, w, b)
    train_accuracy = np.mean(y_train_pred == y_train)
    
    y_test_pred = predict(x_test, w, b)
    test_accuracy = np.mean(y_test_pred == y_test)
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'weights': w,
        'bias': b,
        'error_history': error_history
    }

# Function to visualize learned weights as an image
def visualize_weights(w, shape=(28, 28), digit_pair=None):
    """Visualize the learned weights as an image."""
    plt.figure(figsize=(12, 5))
    
    # Reshape weights to original image dimensions
    weight_img = w.reshape(shape)
    
    # Print statistics about the weights
    print(f"Weight statistics: Mean: {w.mean():.4f}, Min: {w.min():.4f}, Max: {w.max():.4f}")
    
    # Main subplot for combined weights
    plt.subplot(1, 2, 1)
    im = plt.imshow(weight_img, cmap='viridis')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f'Learned Weights for Digits {digit_pair[0]} vs {digit_pair[1]}')
    
    # Plot composite image of both positive and negative weights
    plt.subplot(1, 2, 2)
    composite = np.zeros((*shape, 3))
    pos_weights = weight_img.copy()
    neg_weights = weight_img.copy()
    pos_weights[pos_weights < 0] = 0
    neg_weights[neg_weights > 0] = 0
    neg_weights = abs(neg_weights)
    
    if pos_weights.max() > 0:
        composite[:,:,0] = pos_weights / pos_weights.max()
    if neg_weights.max() > 0:
        composite[:,:,2] = neg_weights / neg_weights.max()
    
    plt.imshow(composite)
    plt.title(f'Composite: Blue={digit_pair[0]}, Red={digit_pair[1]}')
    
    plt.tight_layout()
    plt.show()

# Function to prepare binary classification data
def prepare_binary_data(digit1, digit2, x_train, y_train, x_test, y_test):
    """Prepare binary classification dataset for a digit pair."""
    # Training data
    cond = (y_train == digit1) | (y_train == digit2)
    binary_x_train = x_train[cond]
    binary_y_train = y_train[cond].copy().astype(float)
    binary_y_train[binary_y_train == digit1] = -1
    binary_y_train[binary_y_train == digit2] = 1
    
    # Test data
    cond_test = (y_test == digit1) | (y_test == digit2)
    binary_x_test = x_test[cond_test]
    binary_y_test = y_test[cond_test].copy().astype(float)
    binary_y_test[binary_y_test == digit1] = -1
    binary_y_test[binary_y_test == digit2] = 1
    
    return binary_x_train, binary_y_train, binary_x_test, binary_y_test

# Function to run experiments on digit pairs
def run_digit_pair_experiments(digit_pairs, x_train, y_train, x_test, y_test):
    """Run perceptron experiments on multiple digit pairs."""
    results = {}
    
    for digit1, digit2 in digit_pairs:
        print(f"Training perceptron for digit pair ({digit1}, {digit2})...")
        
        # Prepare binary data
        binary_x_train, binary_y_train, binary_x_test, binary_y_test = prepare_binary_data(
            digit1, digit2, x_train, y_train, x_test, y_test)
        
        # Train and evaluate
        result = evaluate_perceptron(binary_x_train, binary_y_train, binary_x_test, binary_y_test)
        results[f"{digit1}_vs_{digit2}"] = result
        
        print(f"Training accuracy: {result['train_accuracy']:.4f}")
        print(f"Test accuracy: {result['test_accuracy']:.4f}")
        print("---")
        
        # Plot training error curve
        plt.figure(figsize=(10, 6))
        plt.plot(result['error_history'])
        plt.title(f'Training Error Curve for Digits {digit1} vs {digit2}')
        plt.xlabel('Iterations')
        plt.ylabel('Classification Error')
        plt.grid(True)
        plt.show()
        
        # Visualize learned weights
        visualize_weights(result['weights'], digit_pair=(digit1, digit2))
    
    return results

# Function to visualize results
def visualize_results(results):
    """Create a visualization of the perceptron results."""
    pairs = []
    train_accs = []
    test_accs = []
    iterations = []
    
    for pair, result in results.items():
        pairs.append(pair)
        train_accs.append(result['train_accuracy'])
        test_accs.append(result['test_accuracy'])
        iterations.append(len(result['error_history']))
    
    # Create and display a table
    results_df = pd.DataFrame({
        'Digit Pair': pairs,
        'Training Accuracy': train_accs,
        'Test Accuracy': test_accs,
        'Iterations': iterations
    })
    print("\nPerceptron Classification Results for Different Digit Pairs:")
    print(results_df)
    
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(pairs))
    
    plt.bar(index, train_accs, bar_width, label='Training Accuracy', color='skyblue')
    plt.bar(index + bar_width, test_accs, bar_width, label='Test Accuracy', color='orange')
    
    plt.xlabel('Digit Pairs')
    plt.ylabel('Accuracy')
    plt.title('Perceptron Performance Across Different Digit Pairs')
    plt.xticks(index + bar_width/2, pairs)
    plt.legend()
    plt.ylim(0.5, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(train_accs):
        plt.text(i - 0.1, v + 0.02, f'{v:.3f}', color='blue', fontweight='bold')
    
    for i, v in enumerate(test_accs):
        plt.text(i + bar_width - 0.1, v + 0.02, f'{v:.3f}', color='darkred', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def run_najib_task2():
    """Run all steps of the perceptron analysis for MNIST digits."""
    try:
        # Load data using the consistent function
        print("Loading and preparing MNIST data...")
        X_train_flatened, X_test_flatened, y_train, y_test = load_mnist_data()
        
        # Use a subset of training data for faster computation
        sample_size = min(10000, len(y_train))
        indices = np.random.choice(len(y_train), sample_size, replace=False)
        X_train_subset = X_train_flatened[indices]
        y_train_subset = y_train[indices]
        
        # Define digit pairs to experiment with
        digit_pairs = [(0, 1), (3, 8), (4, 9), (5, 6), (1, 7)]
        
        # Run experiments
        results = run_digit_pair_experiments(
            digit_pairs, 
            X_train_subset, 
            y_train_subset, 
            X_test_flatened, 
            y_test
        )
        
        # Visualize results
        visualize_results(results)
        
        print("Perceptron analysis completed successfully!")
        return results
    except Exception as e:
        print(f"Error in perceptron analysis: {e}")
        import traceback
        traceback.print_exc()
        return None



# ----------------------------------------
# ------------ NAJIB Task 3 --------------
# ----------------------------------------  


def create_mlp(input_shape=784, hidden_units=[1000, 1000], output_units=10):
    """
    Create a Multi-Layer Perceptron with the specified architecture.
    
    Parameters:
    input_shape (int): Number of input features
    hidden_units (list): List of hidden layer units
    output_units (int): Number of output units
    
    Returns:
    keras.Model: The compiled MLP model
    """
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

def train_and_evaluate_mlp(model, x_train, y_train, x_test, y_test, batch_size=50, epochs=10):
    """
    Train and evaluate an MLP model.
    
    Parameters:
    model (keras.Model): The model to train
    x_train, y_train: Training data
    x_test, y_test: Test data
    batch_size (int): Batch size for training
    epochs (int): Number of epochs to train for
    
    Returns:
    dict: Results including accuracy, loss, and training history
    """
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        verbose=1
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

def plot_training_curves(history):
    """Plot the training and validation accuracy/loss curves."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def create_and_train_mlp_with_depth(name, hidden_layers, x_train, y_train, x_test, y_test, epochs=10, batch_size=50):
    """
    Create and train an MLP with the specified number of hidden layers.
    
    Parameters:
    name (str): Name for the model
    hidden_layers (list): List of hidden layer units
    x_train, y_train: Training data
    x_test, y_test: Test data
    epochs (int): Number of epochs to train for
    batch_size (int): Batch size for training
    
    Returns:
    dict: Results including accuracy, model, and parameters count
    """
    input_shape = x_train.shape[1]
    
    model = create_mlp(input_shape=input_shape, hidden_units=hidden_layers, output_units=10)
    
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        verbose=0
    )
    
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=0)
    
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
    
    return {
        'name': name,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'parameters': trainable_params,
        'model': model,
        'history': history
    }

def compare_mlp_architectures(architectures, x_train, y_train, x_test, y_test, epochs=10, batch_size=50):
    """
    Train and compare multiple MLP architectures.
    
    Parameters:
    architectures (dict): Dictionary of architecture names and hidden layer configurations
    x_train, y_train: Training data
    x_test, y_test: Test data
    epochs (int): Number of epochs to train for
    batch_size (int): Batch size for training
    
    Returns:
    dict: Results for all architectures
    """
    results = {}
    
    for name, hidden_layers in architectures.items():
        print(f"Training {name} with architecture {hidden_layers}...")
        results[name] = create_and_train_mlp_with_depth(
            name, hidden_layers, x_train, y_train, x_test, y_test, epochs, batch_size
        )
        print(f"  Train accuracy: {results[name]['train_accuracy']:.4f}")
        print(f"  Test accuracy: {results[name]['test_accuracy']:.4f}")
        print(f"  Parameters: {results[name]['parameters']:,}")
        print()
    
    return results

def plot_mlp_comparison(results, architectures):
    """Plot comparison of MLP architectures."""
    # Create a comparison table
    results_table = {
        'MLP': [],
        'Hidden Layers': [],
        'Parameters': [],
        'Train Accuracy': [],
        'Test Accuracy': []
    }
    
    for name, result in results.items():
        results_table['MLP'].append(name)
        results_table['Hidden Layers'].append(len(architectures[name]))
        results_table['Parameters'].append(result['parameters'])
        results_table['Train Accuracy'].append(result['train_accuracy'])
        results_table['Test Accuracy'].append(result['test_accuracy'])
    
    # Plot accuracy vs depth vs parameters
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Accuracy vs. Number of Hidden Layers
    plt.subplot(1, 2, 1)
    plt.plot(results_table['Hidden Layers'], results_table['Train Accuracy'], 'o-', label='Train Accuracy')
    plt.plot(results_table['Hidden Layers'], results_table['Test Accuracy'], 's-', label='Test Accuracy')
    plt.xlabel('Number of Hidden Layers')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of Hidden Layers')
    plt.legend()
    plt.grid(True)
    plt.xticks(results_table['Hidden Layers'])
    
    # Plot 2: Accuracy vs. Number of Parameters
    plt.subplot(1, 2, 2)
    plt.plot(results_table['Parameters'], results_table['Train Accuracy'], 'o-', label='Train Accuracy')
    plt.plot(results_table['Parameters'], results_table['Test Accuracy'], 's-', label='Test Accuracy')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of Parameters')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Print the final results
    for i in range(len(results_table['MLP'])):
        print(f"{results_table['MLP'][i]}: Layers={results_table['Hidden Layers'][i]}, "
              f"Params={results_table['Parameters'][i]:,}, "
              f"Train Acc={results_table['Train Accuracy'][i]:.4f}, "
              f"Test Acc={results_table['Test Accuracy'][i]:.4f}")

def run_najib_task3():
    """Run all steps of the MLP analysis for MNIST digits."""
    try:
        # Load data using the consistent function
        print("Loading and preparing MNIST data...")
        X_train_flatened, X_test_flatened, y_train, y_test = load_mnist_data()
        
        # Get number of features
        nb_features = X_train_flatened.shape[1]  # Should be 784
        
        # Use a subset of training data for faster computation
        sample_size = min(10000, len(y_train))
        indices = np.random.choice(len(y_train), sample_size, replace=False)
        X_train_subset = X_train_flatened[indices]
        y_train_subset = y_train[indices]
        
        # Convert to one-hot encoding
        y_train_one_hot = keras.utils.to_categorical(y_train_subset, 10)
        y_test_one_hot = keras.utils.to_categorical(y_test, 10)
        
        print("Task 3.1: Training base MLP with architecture [784,1000,1000,10]")
        # Create and train the base MLP model
        mlp_model = create_mlp(input_shape=nb_features, hidden_units=[1000, 1000], output_units=10)
        mlp_model.summary()
        
        # Train and evaluate the model
        results = train_and_evaluate_mlp(
            mlp_model, 
            X_train_subset, 
            y_train_one_hot, 
            X_test_flatened, 
            y_test_one_hot
        )
        
        print(f"\nTraining accuracy: {results['train_accuracy']:.4f}")
        print(f"Test accuracy: {results['test_accuracy']:.4f}")
        
        # Plot training curves
        plot_training_curves(results['history'])
        
        print("\nTask 3.2: Comparing MLPs with different depths")
        # Define MLP architectures with different depths
        mlp_architectures = {
            'MLP-2': [1000, 1000],  # 2 hidden layers (original)
            'MLP-3': [800, 800, 800],  # 3 hidden layers
            'MLP-4': [700, 700, 700, 700],  # 4 hidden layers
            'MLP-5': [600, 600, 600, 600, 600],  # 5 hidden layers
            'MLP-7': [500, 500, 500, 500, 500, 500, 500]  # 7 hidden layers
        }
        
        # Train and compare all architectures
        comparison_results = compare_mlp_architectures(
            mlp_architectures,
            X_train_subset,
            y_train_one_hot,
            X_test_flatened,
            y_test_one_hot,
            epochs=10,
            batch_size=50
        )
        
        # Plot comparison results
        plot_mlp_comparison(comparison_results, mlp_architectures)
        
        print("MLP analysis completed successfully!")
        return comparison_results
    except Exception as e:
        print(f"Error in MLP analysis: {e}")
        import traceback
        traceback.print_exc()
        return None




# ----------------------------------------
# ------------ NAJIB Task 4 --------------
# ----------------------------------------  


def create_cnn(input_shape=(28, 28, 1), filters=[32, 64, 128], output_units=10):
    """Create a Convolutional Neural Network with the specified architecture."""
    model = models.Sequential()
    
    # First convolutional layer with stride 1
    model.add(layers.Conv2D(filters[0], kernel_size=(4, 4), strides=(1, 1), padding='same',
                           activation='relu', input_shape=input_shape))
    
    # Add remaining convolutional layers with stride 2
    for i, f in enumerate(filters[1:], 1):
        model.add(layers.Conv2D(f, kernel_size=(4, 4), strides=(2, 2), padding='same',
                               activation='relu'))
    
    # Flatten and add output layer
    model.add(layers.Flatten())
    model.add(layers.Dense(output_units, activation='softmax'))
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data_for_cnn(x_train, x_test):
    """Reshape the data to have the correct format for CNN."""
    edge = int(np.sqrt(x_train.shape[1]))
    x_train_cnn = x_train.reshape(x_train.shape[0], edge, edge, 1)
    x_test_cnn = x_test.reshape(x_test.shape[0], edge, edge, 1)
    return x_train_cnn, x_test_cnn

def train_model(model, x_train, y_train, x_test, y_test, batch_size=50, epochs=10, verbose=1):
    """Train a model and return history and evaluation metrics."""
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        verbose=verbose
    )
    
    train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=0)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    return {
        'model': model,
        'history': history,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
    }

def plot_learning_curves(history):
    """Plot the training and validation accuracy and loss curves."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def create_and_train_cnn(name, filters, x_train, y_train, x_test, y_test, batch_size=50, epochs=10):
    """Create and train a CNN with specified filter configuration."""
    model = create_cnn(input_shape=x_train.shape[1:], filters=filters)
    model._name = name
    
    result = train_model(model, x_train, y_train, x_test, y_test, batch_size, epochs, verbose=0)
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    
    result.update({
        'name': name,
        'parameters': trainable_params
    })
    
    return result

def compare_architectures(architectures, x_train, y_train, x_test, y_test, batch_size=50, epochs=10):
    """Train and compare multiple CNN architectures."""
    results = {}
    for name, filters in architectures.items():
        print(f"Training {name} with filters {filters}...")
        results[name] = create_and_train_cnn(name, filters, x_train, y_train, x_test, y_test, batch_size, epochs)
        print(f"  Train accuracy: {results[name]['train_accuracy']:.4f}")
        print(f"  Test accuracy: {results[name]['test_accuracy']:.4f}")
        print(f"  Parameters: {results[name]['parameters']:,}")
        print()
    return results

def plot_comparison(results, architectures):
    """Plot comparison of CNN architectures."""
    # Create a comparison table
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
    
    # Plot accuracy vs depth vs parameters
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Accuracy vs. Number of Layers
    plt.subplot(1, 2, 1)
    plt.plot(results_table['Layers'], results_table['Train Accuracy'], 'o-', label='Train Accuracy')
    plt.plot(results_table['Layers'], results_table['Test Accuracy'], 's-', label='Test Accuracy')
    plt.xlabel('Number of Convolutional Layers')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of Convolutional Layers')
    plt.legend()
    plt.grid(True)
    plt.xticks(results_table['Layers'])
    
    # Plot 2: Accuracy vs. Number of Parameters
    plt.subplot(1, 2, 2)
    plt.plot(results_table['Parameters'], results_table['Train Accuracy'], 'o-', label='Train Accuracy')
    plt.plot(results_table['Parameters'], results_table['Test Accuracy'], 's-', label='Test Accuracy')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of Parameters')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Print the final results
    for i in range(len(results_table['CNN'])):
        print(f"{results_table['CNN'][i]}: Layers={results_table['Layers'][i]}, "
              f"Params={results_table['Parameters'][i]:,}, "
              f"Train Acc={results_table['Train Accuracy'][i]:.4f}, "
              f"Test Acc={results_table['Test Accuracy'][i]:.4f}")
    
    return results_table

def run_cnn_analysis(x_train_reshaped, y_train, x_test_reshaped, y_test, mlp_results=None):
    """Run the complete CNN analysis pipeline."""
    try:
        # Convert labels to one-hot encoding
        y_train_one_hot = tf.keras.utils.to_categorical(y_train, 10)
        y_test_one_hot = tf.keras.utils.to_categorical(y_test, 10)
        
        # Reshape the data for CNN
        x_train_cnn, x_test_cnn = prepare_data_for_cnn(x_train_reshaped, x_test_reshaped)
        
        # Train base CNN model
        print("Training base CNN model...")
        model = create_cnn(input_shape=x_train_cnn.shape[1:])
        model.summary()
        cnn_result = train_model(model, x_train_cnn, y_train_one_hot, x_test_cnn, y_test_one_hot)
        
        print(f"\nTraining accuracy: {cnn_result['train_accuracy']:.4f}")
        print(f"Test accuracy: {cnn_result['test_accuracy']:.4f}")
        
        plot_learning_curves(cnn_result['history'])
        
        # Define CNN architectures with different depths and widths
        cnn_architectures = {
            'CNN-3': [32, 64, 128],  # 3 layers (original)
            'CNN-4': [32, 64, 96, 128],  # 4 layers
            'CNN-5': [32, 48, 64, 96, 128],  # 5 layers
            'CNN-2': [64, 128],  # 2 layers
            'CNN-6': [16, 32, 48, 64, 96, 128]  # 6 layers
        }
        
        # Compare CNN architectures
        print("\nComparing CNN architectures with different depths and widths...")
        results = compare_architectures(
            cnn_architectures, 
            x_train_cnn, 
            y_train_one_hot, 
            x_test_cnn, 
            y_test_one_hot
        )
        
        # Plot comparison results
        results_table = plot_comparison(results, cnn_architectures)
        
        # Compare CNN to MLP if MLP results are provided
        if mlp_results is not None and 'MLP-2' in mlp_results:
            print("\nComparison between CNN and MLP:")
            print(f"CNN-3 (Original): Test Accuracy={results['CNN-3']['test_accuracy']:.4f}, "
                  f"Parameters={results['CNN-3']['parameters']:,}")
            print(f"MLP-2 (Original): Test Accuracy={mlp_results['MLP-2']['test_accuracy']:.4f}, "
                  f"Parameters={mlp_results['MLP-2']['parameters']:,}")
        
        print("CNN analysis completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error in CNN analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_najib_task4():
    """Run the CNN analysis task."""
    try:
        # Load data using the consistent function
        print("Loading and preparing MNIST data...")
        X_train_flatened, X_test_flatened, y_train, y_test = load_mnist_data()
        
        # Reshape for CNN processing
        x_train_reshaped = X_train_flatened.reshape(-1, 28*28).astype('float32')
        x_test_reshaped = X_test_flatened.reshape(-1, 28*28).astype('float32')
        
        # Convert from [-1, 1] to [0, 1] scale for CNN processing
        x_train_reshaped = (x_train_reshaped + 1) / 2
        x_test_reshaped = (x_test_reshaped + 1) / 2
        
        # Run CNN analysis
        results = run_cnn_analysis(x_train_reshaped, y_train, x_test_reshaped, y_test)
        
        print("CNN analysis completed successfully!")
        return results
    except Exception as e:
        print(f"Error in CNN analysis: {e}")
        import traceback
        traceback.print_exc()
        return None



# ----------------------------------------
# ------------ NAJIB Task 5 --------------
# ----------------------------------------  

# Task 5: Visualizing CNN outcomes

def plot_filters(model, layer_idx, cols=8):
    """Plot the filters of a convolutional layer."""
    layer = model.layers[layer_idx]
    
    if not isinstance(layer, layers.Conv2D):
        print(f"Layer {layer_idx} is not a convolutional layer.")
        return
    
    filters, biases = layer.get_weights()
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

def plot_activation_maps(model, image, layer_indices, digit_class, cols=8):
    """Plot activation maps for a specific image."""
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    # Display input image
    plt.figure(figsize=(5, 5))
    plt.imshow(np.squeeze(image), cmap='gray')
    plt.title(f'Input Image: Digit {digit_class}')
    plt.axis('off')
    plt.show()
    
    # Create models for each layer
    for layer_idx in layer_indices:
        layer = model.layers[layer_idx]
        activation_model = Model(inputs=model.input, outputs=layer.output)
        
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

def generate_deep_dream(model, class_idx, iterations=20, step_size=1.0, octave_scale=1.4, num_octaves=5):
    """Generate a deep dream image for a specific class."""
    # Create a random noise image
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

def visualize_deep_dream_simpler(model, class_indices=[2, 9], input_shape=(28, 28, 1)):
    """A simpler implementation of deep dream visualization."""
    plt.figure(figsize=(len(class_indices) * 5, 5))
    
    for i, class_idx in enumerate(class_indices):
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
    
    plt.tight_layout()
    plt.show()

def visualize_cnn_outcomes(model, x_test, y_test):
    """Visualize CNN filters, activation maps, and generate deep dream images."""
    # Display model summary
    model.summary()
    
    # Visualize filters
    print("\nVisualizing Filters:")
    conv_layer_indices = [i for i, layer in enumerate(model.layers) 
                          if isinstance(layer, layers.Conv2D)]
    
    for layer_idx in conv_layer_indices:
        print(f"Layer {layer_idx}: {model.layers[layer_idx].name}")
        plot_filters(model, layer_idx)
    
    # Find examples of digits '2' and '9'
    digit_2_idx = np.where(np.argmax(y_test, axis=1) == 2)[0][0]
    digit_9_idx = np.where(np.argmax(y_test, axis=1) == 9)[0][0]
    
    digit_2_img = x_test[digit_2_idx]
    digit_9_img = x_test[digit_9_idx]
    
    # Visualize activation maps
    print("\nVisualizing Activation Maps for Digit '2':")
    plot_activation_maps(model, digit_2_img, conv_layer_indices, 2)
    
    print("\nVisualizing Activation Maps for Digit '9':")
    plot_activation_maps(model, digit_9_img, conv_layer_indices, 9)
    
    # Generate deep dream images
    print("\nGenerating Deep Dream Images:")
    visualize_deep_dream_simpler(model, [2, 9])
    
    print("\nDeep Dream Analysis:")
    print("The deep dream images show patterns the model is sensitive to for each digit class.")

def run_najib_task5(model=None, x_test=None, y_test=None):
    """Run the CNN visualization task."""
    try:
        if model is None or x_test is None or y_test is None:
            print("Loading model and data...")
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
            
            # Create and train a simple model if none provided
            if model is None:
                print("Creating and training a simple CNN model...")
                model = create_cnn()
                
                # Train with a small subset for demonstration
                model.fit(x_train[:5000], y_train[:5000], 
                        validation_data=(x_test[:1000], y_test[:1000]),
                        epochs=3, batch_size=64, verbose=1)
        
        # Run visualization
        visualize_cnn_outcomes(model, x_test, y_test)
        
        print("CNN visualization completed successfully!")
        return model
    except Exception as e:
        print(f"Error in CNN visualization: {e}")
        import traceback
        traceback.print_exc()
        return None




# ----------------------------------------
# ------------ NAJIB Task 6 --------------
# ----------------------------------------  


def load_fashion_mnist_data():
    """Load and prepare Fashion MNIST dataset for multi-task learning."""
    # Load Fashion MNIST dataset
    (train_X, train_y_1), (test_X, test_y_1) = keras.datasets.fashion_mnist.load_data()
    
    # Normalize and reshape
    train_X = np.expand_dims(train_X / 255.0, axis=-1)
    test_X = np.expand_dims(test_X / 255.0, axis=-1)
    
    # Create group labels (Task 2)
    # Group 0: Shoes (5,7,9), Group 1: Gendered (3,6,8), Group 2: Uni-Sex (0,1,2,4)
    def create_group_label(y):
        group_labels = np.zeros_like(y)
        group_labels[np.isin(y, [5, 7, 9])] = 0
        group_labels[np.isin(y, [3, 6, 8])] = 1
        group_labels[np.isin(y, [0, 1, 2, 4])] = 2
        return group_labels
    
    train_y_2 = create_group_label(train_y_1)
    test_y_2 = create_group_label(test_y_1)
    
    # Convert to one-hot encoding
    train_y_1 = keras.utils.to_categorical(train_y_1, 10)
    test_y_1 = keras.utils.to_categorical(test_y_1, 10)
    train_y_2 = keras.utils.to_categorical(train_y_2, 3)
    test_y_2 = keras.utils.to_categorical(test_y_2, 3)
    
    return train_X, train_y_1, train_y_2, test_X, test_y_1, test_y_2

def create_single_task_cnn(input_shape, num_classes, task_name):
    """Create a CNN model for a single task."""
    model = models.Sequential(name=f"Single_{task_name}")
    
    # Convolutional layers
    model.add(layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2))
    model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
    
    # Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(3136, activation='relu'))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_multitask_model(input_shape, lambda_value=0.5):
    """Create a multi-task learning model with shared backbone."""
    inputs = keras.Input(shape=input_shape)
    
    # Shared layers
    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    shared_dense = layers.Dense(3136, activation='relu')(x)
    
    # Task-specific layers
    # Task 1: Item Classification
    task1 = layers.Dense(1024, activation='relu')(shared_dense)
    task1 = layers.Dense(100, activation='relu')(task1)
    task1_output = layers.Dense(10, activation='softmax', name='task1_output')(task1)
    
    # Task 2: Group Classification
    task2 = layers.Dense(1024, activation='relu')(shared_dense)
    task2 = layers.Dense(100, activation='relu')(task2)
    task2_output = layers.Dense(3, activation='softmax', name='task2_output')(task2)
    
    model = keras.Model(inputs=inputs, outputs=[task1_output, task2_output])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={'task1_output': 'categorical_crossentropy', 'task2_output': 'categorical_crossentropy'},
        loss_weights={'task1_output': lambda_value, 'task2_output': 1 - lambda_value},
        metrics=['accuracy']
    )
    
    return model

def train_single_task_models(train_X, train_y_1, train_y_2, test_X, test_y_1, test_y_2, batch_size=64, epochs=5):
    """Train individual CNN models for each task."""
    # Create models
    model_task1 = create_single_task_cnn(train_X.shape[1:], 10, "Task1_Item")
    model_task2 = create_single_task_cnn(train_X.shape[1:], 3, "Task2_Group")
    
    # Print model summaries
    print("Task 1 (Item Classification) Model Summary:")
    model_task1.summary()
    print("\nTask 2 (Group Classification) Model Summary:")
    model_task2.summary()
    
    # Train Task 1 model
    print("\nTraining Task 1 (Item Classification) Model...")
    start_time = time.time()
    history_task1 = model_task1.fit(
        train_X, train_y_1,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(test_X, test_y_1),
        verbose=1
    )
    task1_train_time = time.time() - start_time
    
    # Evaluate Task 1 model
    task1_test_loss, task1_test_accuracy = model_task1.evaluate(test_X, test_y_1, verbose=0)
    print(f"Task 1 Test Accuracy: {task1_test_accuracy:.4f}")
    
    # Train Task 2 model
    print("\nTraining Task 2 (Group Classification) Model...")
    start_time = time.time()
    history_task2 = model_task2.fit(
        train_X, train_y_2,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(test_X, test_y_2),
        verbose=1
    )
    task2_train_time = time.time() - start_time
    
    # Evaluate Task 2 model
    task2_test_loss, task2_test_accuracy = model_task2.evaluate(test_X, test_y_2, verbose=0)
    print(f"Task 2 Test Accuracy: {task2_test_accuracy:.4f}")
    
    return {
        'task1': {
            'model': model_task1,
            'history': history_task1,
            'accuracy': task1_test_accuracy,
            'params': model_task1.count_params(),
            'train_time': task1_train_time
        },
        'task2': {
            'model': model_task2,
            'history': history_task2,
            'accuracy': task2_test_accuracy,
            'params': model_task2.count_params(),
            'train_time': task2_train_time
        }
    }

def train_multitask_models(train_X, train_y_1, train_y_2, test_X, test_y_1, test_y_2,
                          lambda_values=[0, 0.25, 0.5, 0.75, 1.0],
                          batch_size=64, epochs=5):
    """Train multiple multi-task models with different lambda values."""
    mtl_results = {}
    
    for lambda_val in lambda_values:
        print(f"\nTraining Multi-Task Model with λ = {lambda_val}")
        
        # Create and compile model
        mtl_model = create_multitask_model(train_X.shape[1:], lambda_val)
        
        # Only print summary for the first model
        if lambda_val == lambda_values[0]:
            mtl_model.summary()
        
        # Train model
        start_time = time.time()
        history = mtl_model.fit(
            train_X,
            {'task1_output': train_y_1, 'task2_output': train_y_2},
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(test_X, {'task1_output': test_y_1, 'task2_output': test_y_2}),
            verbose=1
        )
        train_time = time.time() - start_time
        
        # Evaluate model
        test_results = mtl_model.evaluate(
            test_X,
            {'task1_output': test_y_1, 'task2_output': test_y_2},
            verbose=0
        )
        
        # Extract test accuracies
        task1_accuracy = test_results[3]
        task2_accuracy = test_results[4]
        
        print(f"λ = {lambda_val}:")
        print(f"  Task 1 (Item) Test Accuracy: {task1_accuracy:.4f}")
        print(f"  Task 2 (Group) Test Accuracy: {task2_accuracy:.4f}")
        
        # Store results
        mtl_results[lambda_val] = {
            'model': mtl_model,
            'history': history,
            'task1_accuracy': task1_accuracy,
            'task2_accuracy': task2_accuracy,
            'params': mtl_model.count_params(),
            'train_time': train_time
        }
    
    return mtl_results

def analyze_mtl_results(single_task_results, mtl_results, lambda_values):
    """Analyze and visualize the results of single-task and multi-task models."""
    # Extract accuracies
    task1_accuracies = [mtl_results[lam]['task1_accuracy'] for lam in lambda_values]
    task2_accuracies = [mtl_results[lam]['task2_accuracy'] for lam in lambda_values]
    single_task1_acc = single_task_results['task1']['accuracy']
    single_task2_acc = single_task_results['task2']['accuracy']
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_values, task1_accuracies, 'o-', label='MTL Task 1 (Item)')
    plt.plot(lambda_values, task2_accuracies, 's-', label='MTL Task 2 (Group)')
    plt.axhline(y=single_task1_acc, color='r', linestyle='--',
                label=f'Single Task 1: {single_task1_acc:.4f}')
    plt.axhline(y=single_task2_acc, color='g', linestyle='--',
                label=f'Single Task 2: {single_task2_acc:.4f}')
    
    plt.xlabel('Lambda Value (λ)')
    plt.ylabel('Test Accuracy')
    plt.title('Multi-Task Learning Performance vs Lambda Value')
    plt.grid(True)
    plt.legend()
    plt.xticks(lambda_values)
    plt.tight_layout()
    plt.show()
    
    # Print comparison table
    print("\nResults Comparison:")
    print("-" * 70)
    print(f"{'Model':<20} | {'Task 1 Acc':<10} | {'Task 2 Acc':<10} | {'Parameters':<12}")
    print("-" * 70)
    
    # Single task models
    print(f"{'Single Task 1':<20} | {single_task1_acc:<10.4f} | {'-':<10} | {single_task_results['task1']['params']:<12,}")
    print(f"{'Single Task 2':<20} | {'-':<10} | {single_task2_acc:<10.4f} | {single_task_results['task2']['params']:<12,}")
    
    # Total parameters for both single task models
    total_single_params = single_task_results['task1']['params'] + single_task_results['task2']['params']
    print(f"{'Single Tasks Total':<20} | {'-':<10} | {'-':<10} | {total_single_params:<12,}")
    print("-" * 70)
    
    # MTL models
    for lam in lambda_values:
        model_name = f"MTL (λ={lam})"
        task1_acc = mtl_results[lam]['task1_accuracy']
        task2_acc = mtl_results[lam]['task2_accuracy']
        params = mtl_results[lam]['params']
        print(f"{model_name:<20} | {task1_acc:<10.4f} | {task2_acc:<10.4f} | {params:<12,}")
    
    print("-" * 70)
    
    # Calculate parameter savings
    param_savings = total_single_params - mtl_results[0.5]['params']
    param_savings_percent = (param_savings / total_single_params) * 100
    print(f"Parameter savings with MTL: {param_savings:,} ({param_savings_percent:.2f}%)")
    
    # Find best lambda value
    best_avg_lambda = max(lambda_values, key=lambda lam: (mtl_results[lam]['task1_accuracy'] + 
                                                         mtl_results[lam]['task2_accuracy']) / 2)
    
    best_avg_accuracy = (mtl_results[best_avg_lambda]['task1_accuracy'] + 
                         mtl_results[best_avg_lambda]['task2_accuracy']) / 2
    
    avg_single_acc = (single_task1_acc + single_task2_acc) / 2
    
    print(f"\nBest average performance at λ={best_avg_lambda} with avg accuracy: {best_avg_accuracy:.4f}")
    print(f"Average single task accuracy: {avg_single_acc:.4f}")
    
    if best_avg_accuracy > avg_single_acc:
        print("MTL outperforms the average of single task models!")
    else:
        print("Single task models outperform MTL on average.")

def run_najib_task6(batch_size=64, epochs=3):
    """Run the multi-task learning experiment."""
    print("Task 6: Multi-task Learning with Fashion MNIST")
    
    # Load data
    print("Loading and preparing Fashion MNIST data...")
    train_X, train_y_1, train_y_2, test_X, test_y_1, test_y_2 = load_fashion_mnist_data()
    
    # Train single task models
    print("\nTask 6.1: Training individual models for each task")
    single_task_results = train_single_task_models(
        train_X, train_y_1, train_y_2,
        test_X, test_y_1, test_y_2,
        batch_size=batch_size, epochs=epochs
    )
    
    # Train multi-task models
    print("\nTask 6.2: Training multi-task models with different lambda values")
    lambda_values = [0, 0.25, 0.5, 0.75, 1.0]
    mtl_results = train_multitask_models(
        train_X, train_y_1, train_y_2,
        test_X, test_y_1, test_y_2,
        lambda_values=lambda_values,
        batch_size=batch_size, epochs=epochs
    )
    
    # Analyze results
    print("\nAnalyzing multi-task learning results:")
    analyze_mtl_results(single_task_results, mtl_results, lambda_values)
    
    return single_task_results, mtl_results

