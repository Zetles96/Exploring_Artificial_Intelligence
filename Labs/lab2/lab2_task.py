import numpy as np


class NaiveBayesClassifier:
    def __init__(self):
        self.train_dataset = None
        self.train_labels = None
        self.train_size = 0
        self.num_features = 0
        self.num_classes = 0
        self.num_feature_categories = 0

    def fit(self, train_dataset, train_labels):
        self.train_dataset = train_dataset
        self.train_labels = train_labels
        # TO DO: Calculate the following values
        self.train_size =
        self.num_features =
        self.num_classes =
        self.num_feature_categories =

    def estimate_class_prior(self):
        deltas = (np.arange(self.num_classes) == self.train_labels.reshape(-1, 1))
        # TODO: Write the rest of the code for computing the class_prior
        return class_prior

    def estimate_likelihoods(self):
        # TO DO: Write the code that will calculate the likelihoods for each feature. Note that some calculations in Task 2 may be repeated for this Task.
        likelihoods = []
        for feature in np.arange(self.num_features):
            feature_likelihood =
            likelihoods.append(feature_likelihood)
        return likelihoods

    def predict(self, test_dataset):
        test_size = test_dataset.shape[0]
        class_prior = self.estimate_class_prior()
        likelihoods = self.estimate_likelihoods()
        class_prob = np.tile(np.log(class_prior), (test_size, 1))
        for feature in np.arange(self.num_features):
            feature_likelihood = likelihoods[feature]
            # TO DO: Change the class_prob value based on the likelihood
        return test_predict


if __name__ == '__main__':
    train_dataset = np.load("train_dataset.npy")
    test_dataset = np.load("test_dataset.npy")
    train_labels = np.load("train_labels.npy")
    test_labels = np.load("test_labels.npy")
    nb_model = NaiveBayesClassifier()
    nb_model.fit(train_dataset, train_labels)
    print(f"After fitting the training data, the train size is\
  {nb_model.train_size}, the number of features is {nb_model.num_features},\
  the number of class labels is {nb_model.num_classes}.")  # should be 128, 6, 2
    class_prior = nb_model.estimate_class_prior()
    print(f"The class priors are {class_prior}.")  # should be [0.31538462 0.68461538]
    likelihoods = nb_model.estimate_likelihoods()
    print(
        f"The likelihoods of the first feature (Age) are \n {likelihoods[0]}.")  # The rows are feature categories and the columns are label categories
    # should be [[0.06666667 0.05376344]
    #            [0.02222222 0.07526882]
    #            [0.28888889 0.4516129 ]
    #            [0.35555556 0.24731183]
    #            [0.26666667 0.17204301]]
    test_predict = nb_model.predict(test_dataset)
    print(
        f"The predictions for test data are:\n {test_predict}")  # should be [1 1 1 0 0 0 1 1 1 0 0 1 0 1 0 1 1 0 0 1 1 1 0 0 0 1 1 0 1 1 1 0 0 1 0 1 1
    #            0 1 0 0 0 1 0 0 0 1 0 1 1 1 0 0 0 0]

    accuracy_score = np.sum(test_predict == test_labels) / test_labels.shape[0]

    print(accuracy_score)  # should be 0.67
