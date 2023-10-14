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
        # Hint 1: *train_dataset* has shape *(train_size, num_features)*.
        self.train_size = self.train_dataset.shape[0]
        self.num_features = self.train_dataset.shape[1]
        # Hint 2: *num_classes* is the maximum number in *train_labels* + 1 since the class index starts from $0$.
        self.num_classes = np.max(self.train_labels) + 1
        # Hint 3: *num_feature_categories* is the maximum number of each feature column in *train_dataset* + 1, and will result in a Numpy 1D array with shape *(num_features,)*. Specify the correct values for the parameter *axis*.
        # Hint 4: You may use Numpy's *amax()* function.
        self.num_feature_categories = np.amax(self.train_dataset, axis=0) + 1

    def estimate_class_prior(self):
        deltas = (np.arange(self.num_classes) == self.train_labels.reshape(-1, 1))
        # TODO: Write the rest of the code for computing the class_prior
        # Hint 1: You may notice that a variable called <code>deltas</code> has been defined for you. <code>deltas</code> is an array of value sized <code>(128, 2)</code>. The image below can help you understand what this is:
        # Hint 2: When you apply Numpy's *sum()* function on an array of <code>True</code> and <code>False</code> values, it will count the number of <code>True</code> elements.
        #
        # https://numpy.org/doc/stable/reference/generated/numpy.sum.html
        #
        # Hint 3: Specify the correct values for the parameter *axis* and keep care for Numpy broadcasting.
        class_prior = (np.sum(deltas, axis=0) + 1) / (self.train_size + self.num_classes)
        return class_prior

    def estimate_likelihoods(self):
        # TO DO: Write the code that will calculate the likelihoods for each feature. Note that some calculations in Task 2 may be repeated for this Task.
        # <mark>Return a list with a number of elements *self.num_features*.</mark> The elements of the list are <mark>Numpy 2D arrays with shape *(self.num_feature_categories, self.num_classes)*.</mark>
        #
        # Note that we can adopt the add-one-count trick, as mentioned in the lab2 review.
        #
        # $P(feature_i=m|label=j)=\frac{\text{count_samples}(feature_i = m \& label = j)+1}{\text{count_samples}(label=j)+\text{count_feature_categories}}$
        #
        # Hint 1: You may use Numpy's *transpose(), dot(), sum()* functions.
        #
        # Hint 2: Specify the correct values for the parameter *axis* and keep care for Numpy broadcasting.
        # Create an empty list to store likelihoods for each feature
        likelihoods = []

        # Iterate over each feature
        for feature in range(self.num_features):
            feature_likelihood = np.zeros((self.num_feature_categories[feature], self.num_classes))

            # Iterate over each class label
            for label in range(self.num_classes):
                # Filter the dataset where the feature equals m and label equals j
                filtered_data = self.train_dataset[self.train_labels == label, feature]

                # Calculate the count of samples where feature equals m and label equals j
                count_samples = np.bincount(filtered_data, minlength=self.num_feature_categories[feature])

                # Calculate the likelihood using the add-one-count trick
                likelihood = (count_samples + 1) / (len(filtered_data) + self.num_feature_categories[feature])

                # Store the likelihood in the corresponding position in the array
                feature_likelihood[:, label] = likelihood

            # Append the feature likelihood to the list
            likelihoods.append(feature_likelihood)

        return likelihoods

    def predict(self, test_dataset):
        # Implement the function *predict(self, test_dataset)*. For the sake of simplicity, you can assume that ties will never occur.
        #
        # <mark>Return the predicted labels (integer value from $0$ to *self.num_classes*$-1$) of *test_dataset* as a Numpy 1D array with shape *(test_size, )*</mark>.
        #
        # Hint 1: *test_dataset* has shape *(test_size, num_features)*.
        #
        # Hint 2: Recall that we select the class label with the highest posterior probability to classify a testing data sample:
        #
        # $P(label|features)=\frac{P(label)P(f_1|label)P(f_2|label)P(f_3|label)\ldots P(f_m|label)}{\sum_{j=1}^n P(f_1|label_j)P(f_2|label_j)P(f_3|label_j)\ldots P(f_m|label_j)P(label_j)}$
        #
        # In practical implementation, it is not necessary to compute the posterior probabilities of all classes for all testing data samples. For a specific testing sample, the denominator is the same for all labels, so we only need to compare the numerators. Thus, the predicted class can be defined as:
        #
        # $\text{Predicted class} = \text{argmax}_{label} \log(P(label)) + \sum_{i=1}^m \log(P(f_i|label)$
        #
        # Hint 3: We can get the values inside the *argmax()* function by looping through each feature $i$ and adding one $\log(P(f_{i}|label)$ value at each iteration.
        #
        # Hint 4: The calculated *class_prior* has shape *(self.num_classes,)*. In the partial code given below, Numpy's *tile()* function has been used to create an array with shape *(self.test_size, self.num_classes)* to make it simpler to apply the method suggested in Hint 3.
        #
        # Hint 5: You may use Numpy's *log(), dot(), argmax()* functions.
        test_size = test_dataset.shape[0]
        class_prior = self.estimate_class_prior()
        likelihoods = self.estimate_likelihoods()
        class_prob = np.tile(np.log(class_prior), (test_size, 1))
        for feature in np.arange(self.num_features):
            feature_likelihood = likelihoods[feature]
            # TO DO: Change the class_prob value based on the likelihood
            # Calculate the logarithm of likelihood for the current feature
            log_likelihood = np.log(feature_likelihood)

            # Use broadcasting to add the log likelihood values to class_prob
            class_prob += log_likelihood[test_dataset[:, feature]]

            # Predict the class label for each sample by finding the argmax
        test_predict = np.argmax(class_prob, axis=1)
        return test_predict


# if __name__ == '__main__':
#     train_dataset = np.load("train_dataset.npy")
#     test_dataset = np.load("test_dataset.npy")
#     train_labels = np.load("train_labels.npy")
#     test_labels = np.load("test_labels.npy")
#     nb_model = NaiveBayesClassifier()
#     nb_model.fit(train_dataset, train_labels)
#     print(f"After fitting the training data, the train size is\
#   {nb_model.train_size}, the number of features is {nb_model.num_features},\
#   the number of class labels is {nb_model.num_classes}.")  # should be 128, 6, 2
#     class_prior = nb_model.estimate_class_prior()
#     print(f"The class priors are {class_prior}.")  # should be [0.31538462 0.68461538]
#     likelihoods = nb_model.estimate_likelihoods()
#     print(
#         f"The likelihoods of the first feature (Age) are \n {likelihoods[0]}.")  # The rows are feature categories and the columns are label categories
#     # should be [[0.06666667 0.05376344]
#     #            [0.02222222 0.07526882]
#     #            [0.28888889 0.4516129 ]
#     #            [0.35555556 0.24731183]
#     #            [0.26666667 0.17204301]]
#     test_predict = nb_model.predict(test_dataset)
#     print(
#         f"The predictions for test data are:\n {test_predict}")  # should be [1 1 1 0 0 0 1 1 1 0 0 1 0 1 0 1 1 0 0 1 1 1 0 0 0 1 1 0 1 1 1 0 0 1 0 1 1
#     #            0 1 0 0 0 1 0 0 0 1 0 1 1 1 0 0 0 0]
#
#     accuracy_score = np.sum(test_predict == test_labels) / test_labels.shape[0]
#
# print(accuracy_score)  # should be 0.67
