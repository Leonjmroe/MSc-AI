# This is the class I used to tune my logistic - bayes classifier. There are 5 different hyperparameters that I tuned 

class HyperparameterOptimiser:
    def __init__(self, data, training_cycle_params, learning_rate_params, lr_bias_params, buffer_divisor_params, lambda_params, data_split_count, data_test_ratio, data_split_toggle):
        self.data = data
        self.training_cycle_params = training_cycle_params
        self.learning_rate_params = learning_rate_params
        self.lr_bias_params = lr_bias_params
        self.buffer_divisor_params = buffer_divisor_params
        self.lambda_params = lambda_params
        self.data_split_count = data_split_count
        self.data_test_ratio = data_test_ratio
        self.optimal_params = {}
        self.best_accuracy = 0
        self.data_split_toggle = data_split_toggle


    # Grid search to loop through all my hyperparameters 
    def grid_search(self):
        best_accuracy = 0
        optimised_parameters = {'learning_rate': None, 'training_cycles': None}
        
        for learning_rate in self.learning_rate_params:
            print('Learning Rate: ', learning_rate)
            for cycles in self.training_cycle_params:
                print('Cycle: ', cycles)
                for lr_bias in self.lr_bias_params:
                    print('lr_bias: ', lr_bias)
                    for buffer_divisor in self.buffer_divisor_params:
                        print('buffer_divisor: ', buffer_divisor)
                        for lambda_ in self.lambda_params:
                            print('lambda_: ', lambda_)
 
                            self.data_run(learning_rate, cycles, lr_bias, buffer_divisor, lambda_)

    
    # Find the highest accuracy parameter set and save
    def data_run(self, learning_rate, cycles, lr_bias, buffer_divisor, lambda_):

        if self.data_split_toggle == False:
            test_data = testing_spam
            train_data = training_spam
        
        classifier = LogisticRegressionClassifier(train_data, int(cycles), learning_rate, lr_bias, buffer_divisor, lambda_)
        predictions = classifier.predict(test_data[:, 1:])[0]
        accuracy = np.count_nonzero(predictions == test_data[:, 0])/test_data[:, 0].shape[0]
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.optimal_params['learning_rate'] = learning_rate
            self.optimal_params['training_cycles'] = cycles
            self.optimal_params['lr_bias'] = lr_bias
            self.optimal_params['buffer_divisor'] = buffer_divisor
            self.optimal_params['lambda_'] = lambda_
            print('\n')
            print(f"New best accuracy: {self.best_accuracy:.4f} with Cycles={cycles}, LR={learning_rate}, LR Bias={lr_bias}, Buffer Divisor={buffer_divisor}, Lambda={lambda_}")
            print('\n')

    

    # Code I used to test my model against various splits of training and test data (I aggregated all 1500 data points together for this)
    def split_accuracy_test(self, cycles, learning_rate):
        dataset_size = self.data.shape[0]
        test_size = int(dataset_size * self.data_test_ratio)
        split_accuracies = np.array([])
        
        for _ in range(self.data_split_count):
            np.random.shuffle(data)
            test_data = self.data[:test_size]
            train_data = self.data[test_size:]

            classifier = LogisticRegressionClassifier(train_data, int(cycles), learning_rate)
            predictions = classifier.predict(test_data[:, 1:])
            accuracy = np.count_nonzero(predictions == test_data[:, 0])/test_data[:, 0].shape[0]
    
            split_accuracies = np.append(split_accuracies, accuracy)
            print('Split ' + str(_), 'Accuracy: ', accuracy)
    
        print('Average accuracy: ', np.mean(split_accuracies))
        print('\n')
        return np.mean(split_accuracies)

    

    def run(self):
        optimised_parameters = self.grid_search()
        print("Optimised Parameters: ", self.optimal_params)
        print("Best Accuracy: ", self.best_accuracy)



# Example showing a all data file I used for the batch testing 
data = np.genfromtxt(open("data/all_spam_data.csv"), delimiter=",", encoding='utf-8-sig')


hyperparameter_optimiser = HyperparameterOptimiser( 
            data, 
            training_cycle_params = np.arange(10000, 11000, 250).tolist(), 
            learning_rate_params = np.arange(0.09, 0.1, 0.002).tolist(), 
            lr_bias_params = np.arange(1.0, 1.1, 0.02).tolist(), 
            buffer_divisor_params = np.arange(0.7, 0.9, 0.025).tolist(), 
            lambda_params = np.arange(-1, 1, 0.1).tolist(), 
            data_split_count = 30, 
            data_test_ratio = 0.2,
            data_split_toggle = False )

# Currently set up on a class that has 
hyperparameter_optimiser.run()