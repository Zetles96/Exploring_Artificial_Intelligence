import numpy as np
import sys
import os


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


if __name__ == "__main__":
    print("Hidden Test cross-validation (Task 4.2)")
    try:
        with HiddenPrints():
            from pa1_sol import split_d_fold_sol
            from pa1_task import cross_validate
            from pa1_sol import cross_validate_sol

            X_train = np.load("./mixed_data.npz")["X_train"]
            y_train = np.load("./mixed_data.npz")["y_train"]

            train_d_folds, test_d_folds = split_d_fold_sol(X_train, y_train, 5)

            scores = cross_validate(train_d_folds, test_d_folds, [7, 11, 15, 19, 23])
            scores_sol = cross_validate_sol(
                train_d_folds, test_d_folds, [7, 11, 15, 19, 23]
            )
        print((np.abs(scores - scores_sol) > 0.001).sum() == 0)
    except Exception as e:
        print(e)
