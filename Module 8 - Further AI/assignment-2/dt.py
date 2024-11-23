# %%
# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# %%
# ID3 Decision Tree

class ID3Node:
    # Represents a node in the decision tree
    # feature - which attribute to test
    # branches - dictionary mapping feature values to child nodes
    # value - the prediction (only for leaf nodes)
    def __init__(self, feature=None, branches=None, value=None):
        self.feature = feature  # The feature to split on at this node
        self.branches = branches or {}  # Subtrees for each feature value
        self.value = value  # Prediction stored at leaf nodes


class DecisionTreeID3:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth  # Maximum depth of the tree, limits complexity
        self.root = None  # Root node of the tree


    def entropy(self, labels):
        # Calculate entropy of the target label
        probs = labels.value_counts(normalize=True)  # Proportion of each class in label
        return -sum(probs * np.log2(probs))  # Entropy formula: -Σp*log2(p)


    def information_gain(self, labels, labels_split):
        # Calculate information gain from a split
        parent_entropy = self.entropy(labels)  # Entropy before splitting
        # Weighted sum of entropies of the child nodes
        weighted_entropy = sum((len(subset) / len(labels)) * self.entropy(subset) for subset in labels_split)
        return parent_entropy - weighted_entropy  # Reduction in entropy after the split


    def find_best_feature(self, features, labels):
        # Find the feature that provides the best split
        best_feature = None  # The feature with the highest information gain
        best_gain = -1  # Track the highest information gain
        best_splits = None  # Store the resulting splits for the best feature

        for feature in features.columns:  # Iterate through each feature
            # Create splits by grouping the data by unique feature values
            splits = {value: labels[features[feature] == value] for value in features[feature].unique()}
            labels_split = list(splits.values())  # Convert to list of Series for calculation

            gain = self.information_gain(labels, labels_split)  # Calculate information gain for the feature
            if gain > best_gain:  # Update if the current feature has better information gain
                best_gain = gain
                best_feature = feature
                best_splits = splits

        return best_feature, best_splits  # Return the feature and its splits


    def build_tree(self, features, labels, depth=0):
        # Recursively build the decision tree
        if len(set(labels)) == 1:  # If all target labels are the same, it's a pure node
            return ID3Node(value=labels.iloc[0])  # Create a leaf node with the target value
        if not features.columns.size or (self.max_depth and depth >= self.max_depth):
            # If no features left or max depth reached, return majority class
            return ID3Node(value=labels.value_counts().idxmax())  # Most common target label

        # Find the best feature and splits for this level
        best_feature, best_splits = self.find_best_feature(features, labels)
        if not best_feature:  # If no meaningful split is found, return majority class
            return ID3Node(value=labels.value_counts().idxmax())

        # Create a decision node with branches for each unique value of the best feature
        branches = {}
        for value, subset in best_splits.items():  # For each feature value
            subset_features = features[features[best_feature] == value].drop(columns=[best_feature])  # Remove used feature
            branches[value] = self.build_tree(subset_features, subset, depth + 1)  # Recursively build the tree

        return ID3Node(feature=best_feature, branches=branches)  # Return the decision node



    def fit(self, features, labels):
        # Train the decision tree by building it from the data
        self.root = self.build_tree(features, labels)  # Build the tree starting from the root


    def predict_single(self, node, feature):
        # Predict the label for a single instance by traversing the tree
        if node.value is not None:  # If it's a leaf node, return the stored value
            return node.value
        feature_value = feature[node.feature]  # Get the value of the feature at this node
        if feature_value in node.branches:  # Check if there's a branch for this value
            return self.predict_single(node.branches[feature_value], feature)  # Traverse to the child node
        return None  # If no branch matches, return None (optional: handle this case separately)


    def predict(self, features):
        # Predict the labels for multiple instances
        return [self.predict_single(self.root, feature) for _, feature in features.iterrows()]  # Apply predict_single to each row

# %%
# CART Decision Tree

class CARTNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Index of feature to split on
        self.threshold = threshold  # Threshold value for numerical splits
        self.left = left           # Left subtree
        self.right = right         # Right subtree
        self.value = value         # Prediction value for leaf nodes


class CARTDecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_impurity_decrease=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.root = None
        self.feature_names = None
        self.means = None
        self.stds = None
        
    def gini(self, y):
        """Calculate Gini impurity"""
        if len(y) == 0:
            return 0
        proportions = np.unique(y, return_counts=True)[1] / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def find_best_split(self, X, y):
        """Find the best split using Gini impurity"""
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        parent_gini = self.gini(y)
        
        for feature in range(n_features):
            # Get unique values for the feature
            feature_values = np.unique(X[:, feature])
            
            # Try all possible thresholds
            for threshold in feature_values:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                # Skip if split would result in empty node
                if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
                    continue
                
                # Calculate weighted Gini impurity
                left_gini = self.gini(y[left_mask])
                right_gini = self.gini(y[right_mask])
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                
                weighted_gini = (n_left * left_gini + n_right * right_gini) / n_samples
                
                # Check if this split improves Gini impurity enough
                impurity_decrease = parent_gini - weighted_gini
                if impurity_decrease < self.min_impurity_decrease:
                    continue
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold
    
    def build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_classes == 1:
            # Create leaf node
            return CARTNode(value=np.argmax(np.bincount(y)))
        
        # Find best split
        best_feature, best_threshold = self.find_best_split(X, y)
        
        # If no valid split found, create leaf node
        if best_feature is None:
            return CARTNode(value=np.argmax(np.bincount(y)))
        
        # Create split masks
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build left and right subtrees
        left_subtree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return CARTNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree
        )
    

    def standardize(self, X):
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        return (X - self.means) / self.stds
    
    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
            X = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
        X = self.standardize(X)
        self.root = self.build_tree(X, y)

    def predict_single(self, x, node):
        """Predict single instance"""
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self.predict_single(x, node.left)
        return self.predict_single(x, node.right)
    
    def predict(self, X):
        """Predict multiple instances"""
        X = np.asarray(X)
        return np.array([self.predict_single(x, self.root) for x in X])

# %%
# DT Visulalisation


def visualise_ID3_DT(tree):
    """
    Creates a horizontal tree visualization using NetworkX
    """
    G = nx.Graph()
    pos = {}
    labels = {}
    
    def add_nodes(node, x=0, y=0, parent=None, level=0):
        node_id = str(id(node))
        
        # Create node label
        if node.value is not None:
            label = f"Predict: {node.value}"
        else:
            label = f"Split: {node.feature}"
            
        # Add node
        G.add_node(node_id)
        pos[node_id] = (x, y)
        labels[node_id] = label
        
        # Add edge from parent
        if parent:
            G.add_edge(parent, node_id)
        
        # Process children with vertical spacing
        if node.branches:
            children = list(node.branches.items())
            total_height = len(children)
            for i, (value, child) in enumerate(children):
                # Calculate position for child
                child_y = y - (total_height/2) + i
                child_x = x + 2  # Fixed horizontal spacing
                
                # Add child node
                child_id = add_nodes(child, child_x, child_y, node_id, level+1)
                
                # Add edge label
                edge_center = ((x + child_x)/2, (y + child_y)/2)
                plt.text(edge_center[0], edge_center[1], value, 
                        ha='center', va='center',
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
                
        return node_id
    
    plt.figure(figsize=(15, 10))
    
    # Build graph starting from root
    add_nodes(tree.root)
    
    # Draw the graph
    nx.draw(G, pos,
            labels=labels,
            with_labels=True,
            node_color='lightblue',
            node_size=3000,
            font_size=8,
            font_weight='bold',
            font_color='black',
            width=1,
            edge_color='gray')
    
    plt.axis('off')
    plt.title("Decision Tree (Horizontal Layout)")
    plt.tight_layout()
    plt.show()




def visualise_CART_DT(tree):
    G = nx.Graph()
    pos = {}
    labels = {}
    node_colors = {}
    
    def add_nodes(node, x=0, y=0, parent=None, level=0):
        node_id = str(id(node))
        
        if node.value is not None:
            label = f"Quality {node.value}"
            quality_range = (3, 8)
            color_val = (node.value - quality_range[0]) / (quality_range[1] - quality_range[0])
            node_colors[node_id] = plt.cm.YlGn(color_val)
        else:
            feature_name = tree.feature_names[node.feature]
            original_threshold = (node.threshold * tree.stds[node.feature]) + tree.means[node.feature]
            label = f"{feature_name}\n≤ {original_threshold:.2f}"
            node_colors[node_id] = 'lightblue'
            
        G.add_node(node_id)
        pos[node_id] = (x, y)
        labels[node_id] = label
        
        if parent:
            G.add_edge(parent, node_id)
            
        if node.left is not None:
            child_y = y - 1.5
            child_x = x - 2 ** (3 - level)
            add_nodes(node.left, child_x, child_y, node_id, level+1)
            
        if node.right is not None:
            child_y = y - 1.5
            child_x = x + 2 ** (3 - level)
            add_nodes(node.right, child_x, child_y, node_id, level+1)
            
        return node_id
    
    plt.figure(figsize=(20, 12))
    add_nodes(tree.root)
    
    nx.draw(G, pos,
            labels=labels,
            with_labels=True,
            node_color=[node_colors[node] for node in G.nodes()],
            node_size=3000,
            font_size=9,
            font_weight='bold',
            font_color='black',
            width=1,
            edge_color='gray')
    
    plt.title("Wine Quality Decision Tree", pad=20, fontsize=14)
    plt.axis('off')
    plt.show()

# %%
# Run ID3 DT - Weather Data

data = pd.read_csv('weather-data.csv')

features = data.drop(columns=['Decision', 'Day'])  # Remove both Decision and Day
labels = data['Decision']

tree = DecisionTreeID3(max_depth=3)
tree.fit(features, labels)

predictions = tree.predict(features)
print("Predictions:", predictions)

accuracy = sum(pred == actual for pred, actual in zip(predictions, labels)) / len(labels)
print(f"Accuracy: {accuracy:.2%}")

visualise_ID3_DT(tree)

# %%
# Run CART DT - Weather Data

data = pd.read_csv('winequality-red.csv')

features = data.drop('quality', axis=1)
labels = data['quality']

cart_tree = CARTDecisionTree(max_depth=4, min_samples_split=20, min_impurity_decrease=0.0)
cart_tree.fit(features, labels)

# Make predictions and calculate accuracy
predictions = cart_tree.predict(features.values)
accuracy = sum(pred == actual for pred, actual in zip(predictions, labels)) / len(labels)

# Show tree with accuracy
print(f"Model Accuracy: {accuracy:.2%}")

# Visualize the tree
# visualise_CART_DT(cart_tree) 

# %%



