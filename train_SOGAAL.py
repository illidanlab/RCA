import os

import time
import numpy as np
import copy
import sklearn
from sklearn.impute import KNNImputer, SimpleImputer
import tqdm

import argparse
from sklearn.model_selection import train_test_split
from pyod.models.so_gaal import SO_GAAL
from sklearn.model_selection import GridSearchCV

# from sklearn.svm import OneClassSVM
from data_process import RealDataset


class Solver_SO_GAAL:
    def __init__(
        self,
        data_name,
        seed=0,
        learning_rate=1e-3,
        training_ratio=0.8,
        validation_ratio=0.1,
        missing_ratio=0.5,
        max_epochs=100,
        z_dim=10,
        batch_size=64,
    ):
        # Data loader
        # read data here
        self.max_epochs = max_epochs
        self.z_dim = z_dim
        self.batch_size = batch_size
        np.random.seed(seed)
        data_path = "./data/" + data_name + ".npy"
        self.result_path = "./results/{}/{}/SO_GAAL/{}/".format(
            data_name, missing_ratio, seed
        )

        self.learning_rate = learning_rate
        self.dataset = RealDataset(data_path, missing_ratio=missing_ratio)
        self.seed = seed

        self.data_path = data_path
        self.data_anomaly_ratio = self.dataset.__anomalyratio__()
        self.input_dim = self.dataset.__dim__()
        self.data_normaly_ratio = 1 - self.data_anomaly_ratio
        n_sample = self.dataset.__len__()
        self.n_train = int(n_sample * training_ratio)
        self.n_validation = int(n_sample * validation_ratio)
        self.n_test = n_sample - self.n_train - self.n_validation
        self.best_model = None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.dataset.x,
            self.dataset.y,
            test_size=1 - config.training_ratio - config.validation_ratio,
            random_state=seed,
        )

        print(
            "|data dimension: {}|data noise ratio:{}".format(
                self.dataset.__dim__(), self.data_anomaly_ratio
            )
        )

    def train(self):
        model = SO_GAAL(
            stop_epochs=self.max_epochs, contamination=self.data_anomaly_ratio
        )
        model.fit(self.X_train)

        self.best_model = model

    def test(self):
        print("======================TEST MODE======================")
        pred = self.best_model.predict(self.X_test)

        gt = self.y_test.astype(int)

        from sklearn.metrics import (
            precision_recall_fscore_support as prf,
            accuracy_score,
            roc_auc_score,
        )

        auc = roc_auc_score(gt, -self.best_model.decision_function(self.X_test))
        pred = pred < 0
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = prf(gt, pred, average="binary")

        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, AUC-score : {:0.4f}".format(
                accuracy, precision, recall, f_score, auc
            )
        )

        os.makedirs(self.result_path, exist_ok=True)

        np.save(
            self.result_path + "result.npy",
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f_score,
                "auc": auc,
            },
        )
        return accuracy, precision, recall, f_score, auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AnomalyDetection")
    parser.add_argument(
        "--algorithm", type=str, default="IsolationForest", required=False
    )
    parser.add_argument("--seed", type=int, default=0, required=False)
    parser.add_argument("--data", type=str, default="letter", required=False)
    parser.add_argument("--missing_ratio", type=float, default=0.5, required=False)

    parser.add_argument("--training_ratio", type=float, default=0.599, required=False)
    parser.add_argument("--validation_ratio", type=float, default=0.01, required=False)
    parser.add_argument("--batch_size", type=int, default="128", required=False)
    parser.add_argument("--max_epochs", type=int, default="100", required=False)
    parser.add_argument("--z_dim", type=int, default="10", required=False)
    config = parser.parse_args()

    np.random.seed(config.seed)

    Solver = Solver_SO_GAAL(
        data_name=config.data,
        seed=config.seed,
        missing_ratio=config.missing_ratio,
        training_ratio=config.training_ratio,
        validation_ratio=config.validation_ratio,
        max_epochs=config.max_epochs,
        z_dim=config.z_dim,
        batch_size=config.batch_size,
    )

    Solver.train()
    Solver.test()
