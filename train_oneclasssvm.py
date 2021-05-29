import os
import numpy as np

import argparse
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.impute import KNNImputer, SimpleImputer
from data_process import RealDataset


class Solver_OCSVM:
    def __init__(
        self,
        data_name,
        missing_ratio=0.0,
        seed=0,
        learning_rate=1e-3,
        training_ratio=0.8,
    ):
        # Data loader
        # read data here
        np.random.seed(seed)
        data_path = "./data/" + data_name + ".npy"
        self.result_path = "./results/{}/{}/OCSVM/{}/".format(data_name, missing_ratio, seed)
        self.missing_ratio = missing_ratio
        self.learning_rate = learning_rate
        self.dataset = RealDataset(data_path, missing_ratio=missing_ratio)
        self.seed = seed

        self.data_path = data_path
        self.data_anomaly_ratio = self.dataset.__anomalyratio__()
        self.input_dim = self.dataset.__dim__()
        self.data_normaly_ratio = 1 - self.data_anomaly_ratio
        n_sample = self.dataset.__len__()
        self.n_train = int(n_sample * training_ratio)
        self.n_test = n_sample - self.n_train
        if missing_ratio == 0.0:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.dataset.x,
                self.dataset.y,
                test_size= 1 - config.training_ratio,
                random_state=seed,
            )
        if missing_ratio > 0.0:
            x = self.dataset.x
            m = self.dataset.m
            x_with_missing = x
            x_with_missing[m == 0] = np.nan
            # imputer = KNNImputer(n_neighbors=2)
            imputer = SimpleImputer()
            x = imputer.fit_transform(x_with_missing)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                x,
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
        model = OneClassSVM()
        model.fit(self.X_train)
        self.best_model = model

    def test(self):
        print("======================TEST MODE======================")
        # pred = self.best_model.predict(self.X_test)
        score = self.best_model.score_samples(self.X_test)
        thresh = np.percentile(score, self.data_anomaly_ratio * 100)
        print("Threshold :", thresh)

        pred = (score < thresh).astype(int)
        # pred = pred < 0
        gt = self.y_test.astype(int)

        from sklearn.metrics import (
            precision_recall_fscore_support as prf,
            accuracy_score,
            roc_auc_score
        )
        auc = roc_auc_score(gt, -self.best_model.decision_function(self.X_test))

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = prf(gt, pred, average="binary")

        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, AUC: {:0.4f}".format(
                accuracy, precision, recall, f_score, auc
            )
        )

        os.makedirs(self.result_path, exist_ok=True)

        np.save(
            self.result_path + "result.npy",
            {
                "auc": auc,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f_score,
            },
        )
        return accuracy, precision, recall, f_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AnomalyDetection")
    parser.add_argument("--algorithm", type=str, default="AutoEncoder", required=False)
    parser.add_argument("--seed", type=int, default=0, required=False)
    parser.add_argument("--data", type=str, default="musk", required=False)
    parser.add_argument("--missing_ratio", type=float, default=0.0, required=False)
    parser.add_argument("--training_ratio", type=float, default=0.5, required=False)
    parser.add_argument("--validation_ratio", type=float, default=0.1, required=False)
    config = parser.parse_args()

    np.random.seed(config.seed)

    Solver = Solver_OCSVM(
        data_name=config.data,
        seed=config.seed,
        missing_ratio=config.missing_ratio,
        training_ratio=config.training_ratio,
        validation_ratio=config.validation_ratio,
    )

    Solver.train()
    Solver.test()
