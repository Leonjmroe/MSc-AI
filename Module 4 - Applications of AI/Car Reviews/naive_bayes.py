import numpy as np


class NaiveBayesClassifier:
    def __init__(self, training_data, testing_data):
        self.training_data = training_data
        self.testing_data = testing_data
        
    
    
    def probability_calc(self):
        label_arr = self.training_data[:, 0]
        # Initialise dictionaries for storing prior probabilities and feature probabilities
        prior_probs = {}
        all_feature_probs = {}
        # Laplace smoothing factor to avoid division by zero
        laplace_alpha = 1
        unique_labels = np.unique(label_arr)
        
        for label in unique_labels:
            # Count occurrences of each class label to calculate prior probabilities
            label_count = np.sum(label == label_arr)
            prior_probs[label] = np.log(label_count / len(label_arr))
            probabilities = np.array([])
    
            # Initialise dictionary for storing probabilities of each feature given the class
            feature_probs = {}
            # Extract rows corresponding to the current class
            features = self.training_data[self.training_data[:, 0] == label][:, 1:]
    
            for feature in range(np.shape(features[0])[0]):  # Loop through all features 
                # Calculate probability of each feature given the class with Laplace smoothing
                prob = np.log((np.sum(features[:, feature]) + laplace_alpha) / \
                       (len(features) + laplace_alpha * 2))  # Corrected smoothing formula
                probabilities = np.append(probabilities, prob) 
            
            all_feature_probs[label] = probabilities  
            
        return prior_probs, all_feature_probs



    def testing(self, prior_probs, all_feature_probs, test_element):
        label_probs = {}
        for label in np.unique(self.training_data[:, 0]):
            probabilities = np.array([])
            
            for feature in range(len(test_element) - 1):  # Skip label in test_element
                if test_element[feature + 1] == 1:  # Consider feature if present
                    feature_prob = all_feature_probs[label][feature]  
                    final_prob = prior_probs[label] + feature_prob
                    probabilities = np.append(probabilities, final_prob)
            
            # Calculate class probability by multiplying prior with product of feature probabilities
            label_probs[label] = np.sum(probabilities)
    
        # Predict class with highest probability
        prediction = max(label_probs, key=label_probs.get)
        return prediction



    def predictor(self, prior_probs, all_feature_probs):
        predictions = np.array([])
        for test_element in self.testing_data:
            prediction = self.testing(prior_probs, all_feature_probs, test_element)
            predictions = np.append(predictions, prediction)
        return predictions


    
    def run(self):
        prior_probs, all_feature_probs = self.probability_calc()
        predictions = self.predictor(prior_probs, all_feature_probs)
        return predictions 
