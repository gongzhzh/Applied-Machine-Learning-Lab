import numpy as np
from sklearn.base import BaseEstimator
from scipy.linalg.blas import ddot, dscal, daxpy

class LinearClassifier(BaseEstimator):
    """
    General class for binary linear classifiers. Implements the predict
    function, which is the same for all binary linear classifiers. There are
    also two utility functions.
    """

    def decision_function(self, X):
        """
        Computes the decision function for the inputs X. The inputs are assumed to be
        stored in a matrix, where each row contains the features for one
        instance.
        """
        return X.dot(self.w)

    def predict(self, X):
        """
        Predicts the outputs for the inputs X. The inputs are assumed to be
        stored in a matrix, where each row contains the features for one
        instance.
        """

        # First compute the output scores
        scores = self.decision_function(X)

        # Select the positive or negative class label, depending on whether
        # the score was positive or negative.
        out = np.select([scores >= 0.0, scores < 0.0],
                        [self.positive_class,
                         self.negative_class], default=self.negative_class)
        return out

    def find_classes(self, Y):
        """
        Finds the set of output classes in the output part Y of the training set.
        If there are exactly two classes, one of them is associated to positive
        classifier scores, the other one to negative scores. If the number of
        classes is not 2, an error is raised.
        """
        classes = sorted(set(Y))
        if len(classes) != 2:
            raise Exception("this does not seem to be a 2-class problem")
        self.positive_class = classes[1]
        self.negative_class = classes[0]

    def encode_outputs(self, Y):
        """
        A helper function that converts all outputs to +1 or -1.
        """
        return np.array([1 if y == self.positive_class else -1 for y in Y])

class Pegasos(LinearClassifier):
    
    def __init__(self, n_iter=100, lambda_param=0.0001):
        self.n_iter = n_iter
        self.lambda_param = lambda_param

    def fit(self, X, Y, seed=None):
        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        self.find_classes(Y)

        # Convert all outputs to +1 (for the positive class) or -1 (negative).
        Ye = self.encode_outputs(Y)
        
        # If necessary, convert the sparse matrix returned by a vectorizer
        # into a normal NumPy matrix.
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        
        # Initialize the weight vector to all zeros.
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)

        # Pegasos Algorithm
        # Iteration Counter
        t = 0  
        rng = np.random.default_rng(seed) 
        for i in range(self.n_iter):
            # Select Training Pair
            for idx in rng.permutation(n_samples):
                x, y = X[idx], Ye[idx]
                # Increment Iteration
                t += 1
                eta = 1.0 / (self.lambda_param * t)
                score = np.dot(self.w, x)
                if y * score < 1:
                    self.w = (1 - eta * self.lambda_param) * self.w + eta * y * x
                else:
                    self.w = (1 - eta * self.lambda_param) * self.w
                B = 1/np.sqrt(self.lambda_param)
                norm = np.linalg.norm(self.w)
                if norm > B: self.w *= B / norm
            # for x, y in zip(X, Ye):
            #     # Increment Iteration
            #     t += 1
            #     eta = 1.0 / (self.lambda_param * t)
            #     score = np.dot(self.w, x)
            #     if y * score < 1:
            #         self.w = (1 - eta * self.lambda_param) * self.w + eta * y * x
            #     else:
            #         self.w = (1 - eta * self.lambda_param) * self.w
        
        
class LogisticRegression(LinearClassifier):
    """
    Implementation of the Pegasos algorithm for binary linear classification.
    """

    def __init__(self, n_iter=100, lambda_param=0.0001):
        self.n_iter = n_iter
        self.lambda_param = lambda_param

    def fit(self, X, Y):
        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        self.find_classes(Y)

        # Convert all outputs to +1 (for the positive class) or -1 (negative).
        Ye = self.encode_outputs(Y)
        
        # If necessary, convert the sparse matrix returned by a vectorizer
        # into a normal NumPy matrix.
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        
        # Initialize the weight vector to all zeros.
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)

        # Logistic Regression Algorithm
        # Iteration Counter
        t = 0  
    
        for i in range(self.n_iter):
            for x, y in zip(X, Ye):
                t += 1
                eta = 1.0 / (self.lambda_param * t)
    
                # Logistic Loss
                part = y * np.dot(self.w, x)
                loss = self.lambda_param * self.w - (y * x) / (1 + np.exp(part))
    
                # Gradient descent step
                self.w = self.w - eta * loss
                
class Pegasos_opt(LinearClassifier):
    
    def __init__(self, n_iter=100, lambda_param=0.0001):
        self.n_iter = n_iter
        self.lambda_param = lambda_param

    def fit(self, X, Y):
        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        self.find_classes(Y)

        # Convert all outputs to +1 (for the positive class) or -1 (negative).
        Ye = self.encode_outputs(Y)
        
        # If necessary, convert the sparse matrix returned by a vectorizer
        # into a normal NumPy matrix.
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        
        # Initialize the weight vector to all zeros.
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)

        # Pegasos Algorithm
        # Iteration Counter
        t = 0  
        
        for i in range(self.n_iter):
            # Select Training Pair
            for x, y in zip(X, Ye):
                # Increment Iteration
                t += 1
                eta = 1.0 / (self.lambda_param * t)
                scale = 1.0 - eta * self.lambda_param
                w = np.ascontiguousarray(self.w, dtype=np.float64)
                x = np.ascontiguousarray(x, dtype=np.float64)
                score = ddot(w, x) 
                dscal(scale, self.w) 
                if y * score < 1:
                    self.w = (1 - eta * self.lambda_param) * self.w + eta * y * x
                else:
                    self.w = (1 - eta * self.lambda_param) * self.w

class Pegasos_vec_scale(LinearClassifier):
    
    def __init__(self, n_iter=100, lambda_param=0.0001):
        self.n_iter = n_iter
        self.lambda_param = lambda_param
    
    def fit(self, X, Y):
        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        self.find_classes(Y)

        # Convert all outputs to +1 (for the positive class) or -1 (negative).
        Ye = self.encode_outputs(Y)
        
        # If necessary, convert the sparse matrix returned by a vectorizer
        # into a normal NumPy matrix.
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        
        # Initialize the weight vector to all zeros.
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)

        # Pegasos Algorithm 
        alpha = 1 
        t = 0

        w = np.ascontiguousarray(self.w, dtype=np.float64)
        for i in range(self.n_iter):
            # Select Training Pair
            for x, y in zip(X, Ye):
                t += 1
                eta = 1.0 / (self.lambda_param * t)
                scale = 1.0 - eta * self.lambda_param
                x = np.ascontiguousarray(x, dtype=np.float64)
                score = alpha * ddot(w, x) 
                dscal(scale, w)
                if y * score < 1:
                    daxpy(x, w, a=(eta * y / alpha))
                
                alpha = (1 - eta * self.lambda_param) * alpha
                alpha = max(alpha, 1e-8)
        
        self.w = alpha * w


