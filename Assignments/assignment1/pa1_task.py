# Task 0: setup
# Task 0.1: Import libraries
import numpy as np
import pandas as pd


# Task 0.2 Read dataset


# Task 1: Data Preprocessing
# Task 1.1: Data Splitting
# Now `data` is our dataset with shape (4898, 12). We need to split the data for training and testing purposes. In this project, we let the training data contain the first 4000 rows and the testing data contain the remaining 898 rows. Also, we need to split the data into features and label (commonly called X and y in machine learning) arrays.
#
# Todo:
# Split the Pandas dataframe `data` and store in Numpy arrays `X_train`, `y_train`, `X_test`, `y_test`.
#
# Remarks:
# 1. This task would not be graded.
# 2. As we use many numpy functions later on, the output should be numpy arrays.
#
# Pandas functions you may use:
# `pandas.DataFrame.drop`, `pandas.DataFrame.iloc`, `pandas.DataFrame.to_numpy`


# # Task 1.2: Data Normalization
#
# Normalization is a fundamental preprocessing step in machine learning. It helps to ensure fair treatment of features, facilitate efficient optimization, enhance interpretability, handle different measurement units, and mitigate the impact of outliers. By normalizing the data, we can improve the accuracy and reliability of machine learning models.
#
# Let's introduce 2 common normalization methods: [**Min-Max Normalization** ](https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)) and [ **Z-score Normalization**](https://en.wikipedia.org/wiki/Feature_scaling#Standardization_(Z-score_Normalization)). Suppose $X:(x_1, x_2, ..., x_n)$ is a column (corresponding to a feature), then
# 1. min-max normalization:
# $\displaystyle X_{\text{min-max-normalized}} = \frac{X-X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}$
# 2. z-score normalization:
# $\displaystyle X_{\text{Z-score-normalized}} = \frac{X-\mu_X}{\sigma_X}$
#
# Todo:
# Please implement `min_max_normalization(input_array)` and `z_score_normalization(input_array)`.
#
# Numpy functions you may use:
# `numpy.mean`, `numpy.min`, `numpy.max`, `numpy.std` ...
def min_max_normalization(input_array: np.ndarray) -> np.ndarray:
    """
    Min-Max Normalization
    :param input_array: input array to be normalized
    :return: normalized array
    """
    return (input_array - np.min(input_array, axis=0)) / (np.max(input_array, axis=0) - np.min(input_array, axis=0))


def z_score_normalization(input_array: np.ndarray) -> np.ndarray:
    """
    Z-score Normalization
    :param input_array: input array to be normalized
    :return: normalized array
    """
    return (input_array - np.mean(input_array, axis=0)) / np.std(input_array, axis=0)


# # Task 2: KNN Model
# Now, the training data and testing data are ready. Let's build the KNN model!
#
# In this session, we break down the KNN model into the following functional parts:
# 1. Distance Calculation
# 2. K Nearest Neighbors Finding
# 3. Prediction Generation

# # Task 2.1: Distance Calculation
#
# In KNN, distance calculation plays an important role as we need to find the k nearest neighbors to make the prediction. Here, we introduce 2 common distance calculation methods: [**Euclidean Distance**](https://en.wikipedia.org/wiki/Euclidean_distance) and [**Manhattan Distance**](https://en.wikipedia.org/wiki/Taxicab_geometry). Suppose we are calculating the distance between $X:(x_1, x_2, ... ,x_n)$ and $Y:(y_1, y_2, ... y_n)$ in n-dimensional space.
#
# 1. Euclidean Distance:
# $\displaystyle d(X, Y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$
# 2. Manhattan Distance:
# $\displaystyle d(X, Y) = \sum_{i=1}^{n} | x_i - y_i |$
#
# Todo:
# Please implement `euclidean_distance(X_train, X_test)` and `manhattan_distance(X_train, X_test)`.
#
# Numpy functions you may use:
# `numpy.expand_dims`, `numpy.sqrt`, `numpy.sum` ...
def euclidean_distance(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Euclidean Distance
    :param X_train: training data
    :param X_test: testing data
    :return: distance array
    """
    # Expand the dimensions of X_test to match the shape of X_train
    X_test = np.expand_dims(X_test, axis=1)

    # Calculate the distance between X_train and X_test
    distance = np.sqrt(np.sum((X_train - X_test) ** 2, axis=2))

    return distance


def manhattan_distance(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Manhattan Distance
    :param X_train: training data
    :param X_test: testing data
    :return: distance array
    """
    # Expand the dimensions of X_test to match the shape of X_train
    X_test = np.expand_dims(X_test, axis=1)

    # Calculate the distance between X_train and X_test
    distance = np.sum(np.abs(X_train - X_test), axis=2)

    return distance


# # Task 2.2: Find K Nearest Neighbors
#
# Now we have the distance calculation functions; the next step is to find the k nearest neighbors for each test point.
#
# Todo:
# Please implement `find_k_nearest_neighbor(distance, y_train, k)`
#
# Remarks:
# 1. In case there is a tie, which means more than 1 training points share the same distance between a testing point, we consider the training point with smaller index as smaller (Can search the concept of stable sort).
#
#
# Numpy functions you may use:
# `numpy.argsort`, `numpy.take`, `numpy.take_along_axis` ...
# the function should return the following
# y_neighbor: numpy array of shape (num_rows_test, k), the labels of the k nearest neighbors of each test point
# distance_neighbor: numpy array of shape (num_rows_test, k),  the distance between each test point and its k nearest neighbors
def find_k_nearest_neighbor(distance: np.ndarray, y_train: np.ndarray, k: int) -> tuple:
    """
    Find K Nearest Neighbors
    :param distance: distance array
    :param y_train: training labels
    :param k: number of neighbors
    :return: y_neighbor, distance_neighbor
    """
    # Find the indices of the k nearest neighbors
    indices = np.argsort(distance, axis=1)[:, :k]

    # Find the labels of the k nearest neighbors
    y_neighbor = np.take(y_train, indices)

    # Find the distance between each test point and its k nearest neighbors
    distance_neighbor = np.take_along_axis(distance, indices, axis=1)

    return y_neighbor, distance_neighbor


# # Task 2.3: Weighted Average Prediction
#
# In weighted average prediction, each data point's contribution to the final prediction is weighted based on its importance or relevance. The weights can be assigned manually or determined through a learning algorithm. Higher weights indicate higher importance (we set the weights manually here). The final prediction is obtained by taking the weighted average of the predictions made on each data point.
#
# Target:
# Suppose the labels of k nearest neighbors of a test point are $Y:(y_1, y_2, ..., y_k)$, and the manually assigned weights are $W:(w_1, w_2, ..., w_k)$. Then the prediction value of this point should be
#
# $$
#   y_{\text{pred}} = \frac{y_1 w_1 + \cdots + y_k w_k}{w_1 + \cdots + w_k}
#  = \frac{\displaystyle \sum_{i=1}^{k} y_i w_i}{\displaystyle \sum_{i=1}^{k} w_i}
# $$
# Todo:
# Please implement `weighted_average_predict(y_neighbor, weights=None)`
#
# Remarks:
# 1. the parameter `weights` here is optional. If no `weights` array is passed into the function, then we should treat each of k nearest neighbors equally.
#
# Numpy functions you may use:
# `numpy.expand_dims`, `numpy.sum` ...
def weighted_average_predict(y_neighbor: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """
    Weighted Average Prediction
    :param y_neighbor: labels of k nearest neighbors
    :param weights: weights array
    :return: prediction array
    """
    if weights is None:
        weights = np.ones(y_neighbor.shape[1])  # Equal weights if not provided

    # Calculate the weighted average prediction
    prediction = np.sum(y_neighbor * weights, axis=1) / np.sum(weights)

    return prediction


# # Task 2.4: Distance-based Prediction
#
# Distance-based weighted prediction assigns weights to data points based on their proximity or similarity to the query point. The idea is that closer data points are more likely to influence the prediction more than those farther away. Here, we use a common method: let the weights are inversely proportional to the distance from the query point.
#
# Target:
# Suppose the labels of k nearest neighbors of a test point are $Y:(y_1, y_2, ..., y_k)$, and the distances between each neighbor and the test point are $D:(d_1, d_2, ..., d_k)$. Then
# $\displaystyle y_{\text{pred}} = \frac{\sum_{i=1}^{k} y_iw_i}{\sum_{i=1}^{k} w_i} $ where $\displaystyle w_i = \frac{1}{d_i + \varepsilon}$.
# Notice we use $\varepsilon$ here to avoid division by zero problem.
#
# Todo:
# Please implement `distance_based_predict(y_neighbor, distance_neighbor, epsilon=1)`
#
# Numpy functions you may use:
# `numpy.sum` ...
def distance_based_predict(y_neighbor: np.ndarray, distance_neighbor: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Distance-based Prediction
    :param y_neighbor: labels of k nearest neighbors
    :param distance_neighbor: distance array
    :param epsilon: epsilon
    :return: prediction array
    """
    # Calculate the weights
    weights = 1 / (distance_neighbor + epsilon)

    # Calculate the weighted sum of the neighbor labels for each test point
    weighted_sum = np.sum(y_neighbor * weights, axis=1)

    # Calculate the sum of the weights for each test point
    sum_weights = np.sum(weights, axis=1)

    # Calculate the distance-based prediction for each test point
    prediction = weighted_sum / sum_weights

    return prediction


# # Task 3: Metric Analyzer
#
# Using appropriate metrics to analyze machine learning models is of utmost importance as it enables quantitative performance assessment, facilitates model comparison and provides valuable insights into the model's effectiveness, aiding in informed decision-making and continuous improvement of the learning algorithms. In this task, we introduce 3 metrics.
#
# Suppose $A:(a_1, a_2, ..., a_n)$ is actual labels and $P:(p_1, p_2, ... , p_n)$ is predicted labels.
# The 3 metrics here to analyze the prediction quality are:
# 1.   [Mean Absolute Error](https://en.wikipedia.org/wiki/Mean_absolute_error): $\displaystyle \text{MAE} = \frac{1}{n} \sum_{i=1}^{n}{| a_i - p_i |}$
#
# 2.   [Mean Square Error](https://en.wikipedia.org/wiki/Mean_squared_error): $\displaystyle \text{MSE} = \frac{1}{n} \sum_{i=1}^n {(a_i - p_i)^2}$
#
# 3.   [Mean Absolute Percentage Error](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error): $\displaystyle \text{MAPE} = \frac{1}{n}\sum_{i=1}^{n}{\lvert \frac{a_i - p_i}{a_i} \rvert}$
#
# Todo:
# Please implement `metric_analyze(y_true, y_pred)`
#
# Numpy functions you may use:
# `numpy.mean`, `numpy.min`, `numpy.absolute`...
def metric_analyze(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    Metric Analyzer
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: MAE, MSE, MAPE
    """
    # Calculate the mean absolute error
    mae = np.mean(np.abs(y_true - y_pred))

    # Calculate the mean square error
    mse = np.mean((y_true - y_pred) ** 2)

    # Calculate the mean absolute percentage error
    mape = np.mean(np.abs((y_true - y_pred) / y_true))

    return mae, mse, mape


# # Task 4: D-Fold Cross-validation
#
# [D-fold cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation) is a widely used technique in machine learning for evaluating model performance. It involves dividing the dataset into k subsets or folds, training the model D times using different folds as the test set, and the rest as the training set. By rotating the folds as the test set, D-fold cross-validation provides a more reliable estimate of the model's generalization ability. The performance metrics from each iteration are then averaged to assess the model's effectiveness. This approach is valuable for model evaluation, hyperparameter tuning, and comparing different algorithms.
#
# [Good explanation](https://scikit-learn.org/stable/modules/cross_validation.html)
# [Video introduction](https://www.youtube.com/watch?v=TIgfjmp-4BA&ab_channel=Udacity)
# # Task 4.1: Split D Folds
# To do D-fold cross-validation, we need the D folds of training and testing data. We can first divide the original dataset into k equal-sized parts. Each part represents a distinct subset of the data and is used as training and testing data during cross-validation. Specifically, each fold serves as the test set once while the remaining D-1 folds are used for training.
#
# Todo:
# Please implement `split_d_fold(X, y, d)`
#
# Remarks:
# 1. To distinguish with "k" in "k nearest neighbor", we use "d" to represent the number of folds in this task.
# 2. In theory, it's better to shuffle the data before splitting. But we don't do here for consistent behaviour.
# 3. In this task, you should adopt a direct and simple splitting method: Suppose the input contains $n=m*d+r$ records (r=n%d), the sizes of test folds should be: $(m+1, ..., m+1, m, ..., m)$, where m+1 appears r times. And we pick out the test sets from the beginning of the array.
# For example, if the data is [a, b, c, d, e, f, g, h] and d is 3, 8 = 3*2 + 2, so the sizes of test folds are (3, 3, 2). Then according to our rule, the test folds are [a,b,c], [d,e,f], [g,h].
# 4. The order of training folds and test folds matters and should be corresponding. This means the i-th training fold and the i-th test fold should be able to be combined into one original data set. You must follow this rule.
#
# Numpy functions you may use:
# `numpy.array_split`, `numpy.concatenate` ...
# it should return the following
# train_d_folds: a pyhon list of length d, each entry is a training fold, each fold contains both features and labels
# test_d_folds: a python list of length d, each entry is a testing fold, each fold contains both features and labels
# the the i-th entry of train_d_folds and test_d_folds are corresponding
def split_d_fold(X: np.ndarray, y: np.ndarray, d: int) -> tuple:
    """
    Split D Folds
    :param X: features
    :param y: labels
    :param d: number of folds
    :return: train_d_folds, test_d_folds
    """
    data = np.concatenate((X, y[:, np.newaxis]), axis=1)  # Combine features and labels for better data structure

    num_rows = X.shape[0]
    fold_sizes = [num_rows // d] * d  # Initialize fold sizes equally
    remainder = num_rows % d

    # Adjust fold sizes to distribute the remainder across the first 'remainder' folds
    for i in range(remainder):
        fold_sizes[i] += 1

    # Initialize variables to store training and testing folds
    train_d_folds = []
    test_d_folds = []

    # Split the data into training and testing folds based on fold_sizes
    start_idx = 0
    for fold_size in fold_sizes:
        end_idx = start_idx + fold_size
        test_fold = data[start_idx:end_idx, :]
        train_fold_mask = np.ones(data.shape[0], dtype=bool)
        train_fold_mask[start_idx:end_idx] = False
        train_fold = data[train_fold_mask, :]

        train_d_folds.append(train_fold)
        test_d_folds.append(test_fold)

        start_idx = end_idx

    return train_d_folds, test_d_folds


# # Task 4.2: Cross-Validation
# Now, we have the training and test folds in hand. Let's use these folds to validate the performance of knn model composed with functional parts.
#
# Todo:
# Please implement `cross_validate`, a skeleton code is provided for you.
#
# Recommanded steps:
# 1. Read and understand the provided code.
# 2. Generate X_train, X_test, y_train, y_test for each round.
# 3. Make prediction with the KNN model composed with proper implemented functional parts.
# 4. Analyze the prediction with proper metric and store the result in scores.
#
# Remarks:
# 1. The KNN models to be validated here use Euclidean distance and weighted-average prediction that treat every neighbor equally. The only difference is the k of each model.
# 2. MSE is used as metric.
# 3. No external function is allowed.
# 4. During grading, we will pass the training and test folds (not generating from your `split_d_fold` function). And we only care about the return mean_scores.
# should return
# mean_scores: numpy array of shape(len(k_list), ), the mean score for each k
def cross_validate(train_d_folds: np.ndarray, test_d_folds: np.ndarray, k_list: np.ndarray) -> np.ndarray:
    # please write pydoc
    """
    Cross-Validation
    :param train_d_folds:  a pyhon list of length d, each entry is a training fold (numpy array)
    :param test_d_folds: a python list of length d, each entry is a test fold (numpy array)
    :param k_list: a python list, contains the k(s) to be validated
    :return: the i-th train_fold and test_fold are corresponding
    """
    scores = np.zeros((len(k_list), len(train_d_folds)))

    for k_index, k in enumerate(k_list):
        for fold in range(len(train_d_folds)):
            # Step 2: Generate X_train, X_test, y_train, y_test for each round
            X_train = train_d_folds[fold][:, :-1]  # Features in training data
            y_train = train_d_folds[fold][:, -1]  # Labels in training data
            X_test = test_d_folds[fold][:, :-1]  # Features in test data
            y_test = test_d_folds[fold][:, -1]  # Labels in test data

            def euclidean_distance(x1, x2):
                return np.sqrt(np.sum((x1 - x2) ** 2))

            def knn_predict(X_train, y_train, x_test, k):
                distances = [euclidean_distance(x_test, x_train) for x_train in X_train]
                k_indices = np.argsort(distances)[:k]
                k_nearest_labels = [y_train[i] for i in k_indices]
                return np.mean(k_nearest_labels)

            predictions = [knn_predict(X_train, y_train, x_test, k) for x_test in X_test]
            mse = np.mean((predictions - y_test) ** 2)

            # Store the metric in scores
            scores[k_index, fold] = mse

    mean_scores = np.mean(scores, axis=1, keepdims=False)
    return mean_scores


# # Task 4.3: Find the Best K
# Suppose we used `cross_validate` to get the mean score for each k in the k-list. Now it's time to decide which k is the best. Suppose we first care about the prediction quality (which means score, in our case, the lower score means better prediction). When the prediction qualities for 2 k are the same, we pick the smaller one to improve efficiency.
#
# Todo:
# Please implement `find_best_k`
def find_best_k(k_list: np.ndarray, mean_scores: np.ndarray) -> int:
    """
    Find the Best K
    :param k_list: a python list, contains the k(s) to be validated
    :param mean_scores: a numpy array of shape(len(k_list), ), the mean score for each k
    :return: the best k
    """
    # todo start #
    # Creates a list of tuples where the first element is the mean score and the second element is the corresponding k
    score_k_pairs = [(score, k) for score, k in zip(mean_scores, k_list)]

    # Sorts the list of tuples in ascending order based on mean score and then in ascending order based on k
    sorted_pairs = sorted(score_k_pairs, key=lambda x: (x[0], x[1]))

    # Return the k value of the first tuple in the sorted list (smallest mean score, smallest k)
    best_k = sorted_pairs[0][1]
    # todo end #
    return best_k


if __name__ == '__main__':
    data = pd.read_csv('winequality-white.csv', delimiter=';')
    X_train = data.iloc[:4000, :-1].to_numpy()
    y_train = data.iloc[:4000, -1].to_numpy()
    X_test = data.iloc[4000:, :-1].to_numpy()
    y_test = data.iloc[4000:, -1].to_numpy()
    print("X_train shape: {} and y_train shape: {}".format(X_train.shape, y_train.shape))
    print("X_test shape: {} and y_test shape: {}".format(X_test.shape, y_test.shape))
    normalized_X_train = min_max_normalization(X_train)
    normalized_X_test = min_max_normalization(X_test)
    distance = euclidean_distance(normalized_X_train, normalized_X_test)
    y_neighbor, distance_neighbor = find_k_nearest_neighbor(distance, y_train, 11)
    y_pred = weighted_average_predict(y_neighbor)
    print(metric_analyze(y_test, y_pred))
    train_d_folds, test_d_folds = split_d_fold(normalized_X_train, y_train, 5)
    list_k = [11, 13, 15, 17, 19]
    print(find_best_k(list_k, cross_validate(train_d_folds, test_d_folds, list_k)))
