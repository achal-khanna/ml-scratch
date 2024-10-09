import numpy as np
from collections import Counter

# Class defining KNN structure
class KNN:
    # The input arrays are to be given as numpy arrays with each row specifying a new element/ point. 
    def __init__(self, k, X_train: np.ndarray, X_test: np.ndarray, Y_train: np.ndarray, Y_test: np.ndarray = None, distance_type = 'Euclidean', learning_type = 'Classification'):
        self.k = k
        self.distance_type = distance_type
        self.learning_type = learning_type        
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

    # The task is performed by computing (and storing them in an array) the required type of distance between all X_train and the single point.
    # The array is sorted and the first k values are taken for evaluation. 
    def operation(self, X_point: np.array):
        distance_array = self.X_train - X_point
        if (distance_array.ndim == 1):
            distance_array = distance_array.reshape(-1, 1)
            
        distance_array = np.sum(np.square(distance_array), axis = 1) if (self.learning_type == 'Euclidean') else (np.sum(np.abs(distance_array), axis = 1))

        sorted_indices = np.argsort(distance_array)
        distance_array = distance_array[sorted_indices]
        helper_array = self.Y_train[sorted_indices]

        # It is assumed that the variables in y are independent and hence, the frequecy of each category can be evaluated alone
        if (self.learning_type == 'Classification'):
            # The types of all columns are implicitly converted into str if even a single column contains str. 
            result_array = ([""] * len(self.Y_train[0])) if (type(self.Y_train[0]) == np.ndarray) else ""

            if (type(self.Y_train[0]) == np.ndarray): 
                counter_array = [(Counter(helper_array[:self.k, column])).most_common() for column in range(len(helper_array[0]))] 

                # In case of a tie, the element which was the closest to it will decide the labels
                for column in range(len(counter_array)):
                    result_array[column] = counter_array[column][0][0]

            else:
                counter_array = (Counter(helper_array[:self.k])).most_common()
                result_array = counter_array[0][0] 

        else:
            # The result will be a numerical value, hence 0 is used as an indicator
            result_array = np.array([0] * len(self.Y_train[0])) if (type(self.Y_train[0]) == np.ndarray) else 0

            if (type(self.Y_train[0]) == np.ndarray): 
                result_array = np.mean(helper_array[:self.k], axis = 0)

            else:
                result_array = np.mean(np.array(helper_array[:self.k]))

        return result_array   


    # The above task is repeated for all the points in X_test
    def operation_stacked(self):
        result = []
        for points in range(len(self.X_test)):
            prediction = self.operation(self.X_test[points])
            result.append(list(prediction) if (type(prediction) == np.ndarray) else prediction)
        
        return result
    
    def mse(self):
        if (self.Y_test is None):
            print("Did not input Y_test, thus can't calculate MSE")
            return None
        prediction = np.array(self.operation_stacked())
        mse = np.mean(np.square(prediction - self.Y_test), axis = 0)
        return mse

    def mae(self):
        if (self.Y_test is None):
            print("Did not input Y_test, thus can't calculate MAE")
            return None
        prediction = np.array(self.operation_stacked())
        mae = np.mean(np.abs(prediction - self.Y_test), axis = 0)
        return mae

    def accuracy(self):
        if (self.Y_test is None):
            print("Did not input Y_test, thus can't calculate accuracy")
            return None
        prediction = np.array(self.operation_stacked())
        accuracy = (prediction == self.Y_test)
        correct_count = np.sum(accuracy, axis = 0)
        total_count = len(accuracy)

        return correct_count/total_count*100
        

