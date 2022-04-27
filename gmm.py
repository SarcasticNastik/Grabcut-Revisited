import numpy as np
from sklearn.cluster import KMeans

class GMM:
    def __init__(self, X=None, num_components=5):
        self.num_components = num_components
        self.num_features = X.shape[1]
        self.samples = np.zeros(self.num_components)
        self.coefficients = np.zeros(self.num_components)
        self.means = np.zeros((self.num_components, self.num_features))
        self.covariance = np.zeros((self.num_components,self.num_features, self.num_features))

        ## Initializing GMM weights with KMeans
        labels = KMeans(n_clusters=self.num_components,n_init=1).fit(X).labels_
        self.fit(X,labels)

    def score(self, X, c_i=None)->np.ndarray:
        """Predict probabilities of samples belong to component ci """
        scores = np.zeros(X.shape[0])
        if self.coefficients[c_i] > 0:
            Δ = X - self.means[c_i]
            x1 = np.dot(np.linalg.inv(self.covariance[c_i]), Δ.T).T
            Σ = np.sum(Δ * x1, axis=1)
            score = np.exp(-Σ/2) / (np.sqrt(2 * np.pi * np.linalg.det(self.covariance[c_i])))
            return score
        
    def prob(self, X)->np.ndarray:
        """ Predict probability (weighted score) of samples belong to the GMM """
        probs = list(map(lambda x: self.score(X,x), list(range(self.num_components))))
        return np.dot(self.coefficients,probs)
    
    def component(self, X)->np.ndarray:
        """ Predicts which GMM component the samples belong to """
        probs = np.array(list(map(lambda x: self.score(X,x), list(range(self.num_components))))).T
        return np.argmax(probs,axis=1)


    def fit(self, X, labels) ->None:
        """ Computes mean and co-variance """
        self.samples = np.zeros(self.num_components)
        self.coefficients = np.zeros(self.num_components)

        # Generating a frequency map
        unique_labels, cnt = np.unique(labels,return_counts=True)
        self.samples[unique_labels] = cnt

        variance = 0.01
        for c_i in unique_labels:
            num = self.samples[c_i]

            self.coefficients[c_i] = self.samples[c_i] / np.sum(self.samples)
            self.means[c_i] = np.mean(X[c_i==labels], axis=0)
            if self.samples[c_i] <= 1:
                self.covariance[c_i] = 0
            else:
                self.covariance[c_i] = np.cov(X[c_i == labels].T)

            Δ = np.linalg.det(self.covariance[c_i])
            if Δ <= 0:
                self.covariance[c_i] += np.identity(self.num_features) * variance
                Δ = np.linalg.det(self.covariance[c_i])
