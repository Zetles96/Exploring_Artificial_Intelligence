import numpy as np


def min_max_normalization_sol(input_array):
    min_array = np.min(input_array, axis=0, keepdims=True)
    max_array = np.max(input_array, axis=0, keepdims=True)
    normalized_array = (input_array - min_array) / (max_array - min_array)
    return normalized_array


def z_score_normalization_sol(input_array):
    mean_array = np.mean(input_array, axis=0)
    std = np.std(input_array, axis=0)
    normalized_array = (input_array - mean_array) / std
    return normalized_array


def euclidean_distance_sol(X_train, X_test):
    difference = np.expand_dims(X_train, axis=0) - np.expand_dims(X_test, axis=1)
    distance = np.sqrt(np.sum(np.square(difference), axis=2, keepdims=False))
    return distance


def manhattan_distance_sol(X_train, X_test):
    difference = np.expand_dims(X_train, axis=0) - np.expand_dims(X_test, axis=1)
    distance = np.sum(np.abs(difference), axis=2, keepdims=False)
    return distance


def find_k_nearest_neighbor_sol(distance, y_train, k):
    neighbor_index = np.argsort(distance, kind="stable", axis=1)[:, :k]
    y_neighbor = np.take(y_train[:, np.newaxis], neighbor_index)
    distance_neighbor = np.take_along_axis(distance, neighbor_index, axis=1)
    return y_neighbor, distance_neighbor


def weighted_average_predict_sol(y_neighbor, weights=None):
    if weights is None:
        weights = np.ones(y_neighbor.shape[1])
    weighted_sum = np.sum(y_neighbor * np.expand_dims(weights, axis=0), axis=1)
    sum_of_weights = np.sum(weights)
    prediction = weighted_sum / sum_of_weights
    return prediction


def distance_based_predict_sol(y_neighbor, distance_neighbor, epsilon=1):
    weights = 1.0 / (distance_neighbor + epsilon)
    weighted_sum = np.sum(y_neighbor * weights, axis=1)
    sum_of_weights = np.sum(weights, axis=1)
    prediction = weighted_sum / sum_of_weights
    return prediction


def metric_analyze_sol(y_true, y_pred):
    mae = np.mean(np.absolute(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    mape = np.mean(np.abs((y_true - y_pred) / y_true))
    return mae, mse, mape


def split_d_fold_sol(X, y, d):
    data = np.concatenate((X, y[:, np.newaxis]), axis=1)  # for better data structure
    val_d_folds = np.array_split(data, d, axis=0)
    train_d_folds = [
        np.concatenate((val_d_folds[0:fold] + val_d_folds[fold + 1 : d]), axis=0)
        for fold in range(d)
    ]
    return train_d_folds, val_d_folds


def cross_validate_sol(train_d_folds, val_d_folds, k_list):
    scores = np.zeros((len(k_list), len(train_d_folds)))
    for k_index, k in enumerate(k_list):
        for fold in range(len(train_d_folds)):
            X_train, y_train = (train_d_folds[fold])[:, :-1], (train_d_folds[fold])[
                :, -1
            ]
            X_test, y_test = (val_d_folds[fold])[:, :-1], (val_d_folds[fold])[:, -1]
            distance = euclidean_distance_sol(X_train, X_test)
            y_neighbor, distance_neighbor = find_k_nearest_neighbor_sol(
                distance, y_train, k
            )
            y_pred = weighted_average_predict_sol(y_neighbor)
            scores[k_index, fold] = metric_analyze_sol(y_test, y_pred)[1]
    mean_scores = np.mean(scores, axis=1, keepdims=False)
    return mean_scores


def find_best_k_sol(k_list, mean_scores):
    indices = np.where(mean_scores == np.min(mean_scores))
    best_k = np.min(np.array(k_list)[indices])
    return best_k
