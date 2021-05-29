import torch as torch
import os
import random
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
import argparse

from models.RCA import SingleAE
from data_process import RealDataset


class Solver_RCA_Multi:
    def __init__(
        self,
        data_name,
        n_member=2,
        start_ratio=0.0,
        decay_ratio=0.01,
        hidden_dim=128,
        z_dim=10,
        seed=0,
        learning_rate=1e-3,
        batch_size=128,
        training_ratio=0.8,
        validation_ratio=0.1,
        max_epochs=100,
        coteaching=1.0,
        oe=0.0,
        missing_ratio=0.0,
        knn_impute=False,
    ):
        # Data loader
        # read data here
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        use_cuda = torch.cuda.is_available()
        self.data_name = data_name
        self.knn_impute = knn_impute
        self.device = torch.device("cuda" if use_cuda else "cpu")
        data_path = "./data/" + data_name + ".npy"
        self.missing_ratio = missing_ratio
        self.model_save_path = "./trained_model/{}/{}/{}-RCA/{}/".format(
            data_name, missing_ratio, n_member, seed
        )
        if oe == 0.0:
            self.result_path = "./results/{}/{}/{}-RCA/{}/".format(
                data_name, missing_ratio, n_member, seed
            )
        else:
            self.result_path = "./results/{}/{}/{}-RCA_{}/{}/".format(
                data_name, missing_ratio, n_member, oe, seed
            )

        os.makedirs(self.model_save_path, exist_ok=True)
        self.learning_rate = learning_rate
        self.dataset = RealDataset(data_path, missing_ratio=self.missing_ratio)
        self.seed = seed
        self.start_ratio = start_ratio

        self.decay_ratio = decay_ratio
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.max_epochs = max_epochs
        self.coteaching = coteaching
        self.start_ratio = start_ratio
        self.data_path = data_path

        self.data_anomaly_ratio = self.dataset.__anomalyratio__() + oe

        self.input_dim = self.dataset.__dim__()
        self.data_normaly_ratio = 1 - self.data_anomaly_ratio

        n_sample = self.dataset.__len__()
        self.n_train = int(n_sample * (training_ratio))
        # self.n_validation = int(n_sample * validation_ratio)
        self.n_test = n_sample - self.n_train
        print(
            "|data dimension: {}|data noise ratio:{}".format(
                self.dataset.__dim__(), self.data_anomaly_ratio
            )
        )

        self.decay_ratio = abs(self.start_ratio - (1 - self.data_anomaly_ratio)) / (
            self.max_epochs / 2
        )
        training_data, testing_data = data.random_split(
            dataset=self.dataset, lengths=[self.n_train, self.n_test]
        )

        self.training_loader = data.DataLoader(
            training_data, batch_size=batch_size, shuffle=True
        )

        self.testing_loader = data.DataLoader(
            testing_data, batch_size=self.n_test, shuffle=False
        )
        self.n_member = n_member
        self.ae = None
        self.discriminator = None
        self.build_model()
        self.print_network()

    def build_model(self):
        self.ae = []
        for _ in range(self.n_member):
            ae = SingleAE(
                input_dim=self.input_dim, hidden_dim=self.hidden_dim, z_dim=self.z_dim
            )
            ae = ae.to(self.device)
            self.ae.append(ae)

    def print_network(self):
        num_params = 0
        for p in self.ae[0].parameters():
            num_params += p.numel()
        print(
            "The number of parameters: {}, number of networks".format(
                num_params, self.n_member
            )
        )

    def train(self):
        optimizer = []
        for i in range(self.n_member):
            optimizer.append(
                torch.optim.Adam(self.ae[i].parameters(), lr=self.learning_rate)
            )
            self.ae[i].eval()

        loss_mse = torch.nn.MSELoss(reduction="none")
        if self.data_name == "optdigits":
            loss_mse = torch.nn.BCELoss(reduction="none")

        min_val_error = 1e10
        for epoch in tqdm(range(self.max_epochs)):  # train 3 time classifier
            for i, (x, y) in enumerate(self.training_loader):
                x = x.to(self.device).float()
                # m = m.to(self.device).float()
                n = x.shape[0]
                n_selected = int(n * (1 - self.start_ratio))

                if config.coteaching == 0.0:
                    n_selected = n
                if i == 0:
                    current_ratio = "{}/{}".format(n_selected, n)

                selected_all_model = []
                with torch.no_grad():
                    for model_idx in range(self.n_member):
                        self.ae[model_idx].eval()
                        xhat = self.ae[model_idx](x.float())
                        error = loss_mse(xhat, x)
                        error = error.sum(dim=1)
                        _, index = torch.sort(error)
                        index = index[:n_selected]
                        selected_all_model.append(index)

                    random.shuffle(selected_all_model)

                for model_idx in range(self.n_member):
                    optimizer[model_idx].zero_grad()
                    self.ae[model_idx].train()
                    xhat = self.ae[model_idx](x[selected_all_model[model_idx]])
                    error = loss_mse(xhat, x[selected_all_model[model_idx]])
                    error = error.mean()
                    error.backward()
                    optimizer[model_idx].step()

            if self.start_ratio < self.data_anomaly_ratio:
                self.start_ratio = min(
                    self.data_anomaly_ratio, self.start_ratio + self.decay_ratio
                )
            if self.start_ratio > self.data_anomaly_ratio:
                self.start_ratio = max(
                    self.data_anomaly_ratio, self.start_ratio - self.decay_ratio
                )  # 0.0005 for 0.1 anomaly, 0.0001 for 0.001 anomaly

            # with torch.no_grad():
            #     self.ae.eval()
            #     for i, (x, y, m) in enumerate(self.testing_loader):
            #         x = x.to(self.device).float()
            #         m = m.to(self.device).float()
            #         # y = y.to(device)
            #         x = x.float()
            #         _, _, xhat1, xhat2 = self.ae(x, x, m, m)
            #         error1 = loss_mse(xhat1, x)
            #         error2 = loss_mse(xhat2, x)
            #         error1 = error1.sum(dim=1)
            #         error2 = error2.sum(dim=1)
            #
            #         n_val = x.shape[0]
            #         n_selected = int(n_val * (1 - self.data_anomaly_ratio))
            #         if self.coteaching == 0.0:
            #             n_selected = n
            #         # n_selected = n_val
            #         _, index1 = torch.sort(error1)
            #         _, index2 = torch.sort(error2)
            #         index1 = index1[:n_selected]
            #         index2 = index2[:n_selected]
            #
            #         x1 = x[index2, :]
            #         x2 = x[index1, :]
            #         m1 = m[index2, :]
            #         m2 = m[index1, :]
            #         z1, z2, xhat1, xhat2 = self.ae(x1, x2, m1, m2)
            #         val_loss = loss_mse(x1, xhat1) + loss_mse(x2, xhat2)
            #         val_loss = val_loss.sum()
            #         if val_loss < min_val_error:
            #             # print(epoch)
            #             min_val_error = val_loss
            #             torch.save(
            #                 self.ae.state_dict(),
            #                 os.path.join(self.model_save_path, "parameter.pth"),
            #             )

            # scheduler.step()

    def test(self):
        print("======================TEST MODE======================")
        # self.dagmm.load_stat
        # self.ae.load_state_dict(torch.load(self.model_save_path + "parameter.pth"))
        # self.ae.eval()
        mse_loss = torch.nn.MSELoss(reduction="none")
        if self.data_name == "optdigits":
            mse_loss = torch.nn.BCELoss(reduction="none")

        error_list = []
        for _ in range(1000):  # ensemble score over 100 stochastic feedforward
            with torch.no_grad():
                error_average = torch.zeros(self.n_test).cuda()
                for model in self.ae:
                    model.train()
                    for _, (x, y) in enumerate(self.testing_loader):
                        y = y.data.cpu().numpy()
                        x = x.to(self.device).float()
                        # m = m.to(self.device).float()
                        xhat = model(x.float())
                        error = mse_loss(xhat, x)
                        error = error.sum(dim=1)
                        error_average = error_average + error

                error = error_average.data.cpu().numpy()
                error_list.append(error)
        error_list = np.array(error_list)
        # error_list = np.percentile(error, )
        error = error_list.mean(axis=0)
        from sklearn.metrics import (
            precision_recall_fscore_support as prf,
            accuracy_score,
            roc_auc_score,
        )

        gt = y.astype(int)
        auc = roc_auc_score(gt, error)

        thresh = np.percentile(error, self.dataset.__anomalyratio__() * 100)
        print("Threshold :", thresh)

        pred = (error > thresh).astype(int)
        gt = y.astype(int)
        auc = roc_auc_score(gt, error)
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = prf(gt, pred, average="binary")

        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, AUC : {:0.4f}".format(
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
        print("result save to {}".format(self.result_path))
        return accuracy, precision, recall, f_score, auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RCA")
    parser.add_argument("--algorithm", type=str, default="RCA", required=False)
    parser.add_argument("--seed", type=int, default=5, required=False)
    parser.add_argument("--decay", type=float, default=0.001, required=False)
    parser.add_argument("--data", type=str, default="letter", required=False)
    parser.add_argument("--max_epochs", type=int, default=50, required=False)
    parser.add_argument("--knn_impute", type=bool, default=False, required=False)
    parser.add_argument("--hidden_dim", type=int, default=256, required=False)
    parser.add_argument("--batch_size", type=int, default=128, required=False)
    parser.add_argument("--oe", type=float, default=0.0, required=False)
    parser.add_argument("--training_ratio", type=float, default=0.599, required=False)
    parser.add_argument("--validation_ratio", type=float, default=0.001, required=False)
    parser.add_argument("--learning_rate", type=float, default=3e-4, required=False)
    parser.add_argument("--start_ratio", type=float, default=0.0, required=False)
    parser.add_argument("--z_dim", type=int, default=10, required=False)
    parser.add_argument("--coteaching", type=float, default=1.0, required=False)
    parser.add_argument("--n_member", type=int, default=3, required=False)
    parser.add_argument("--missing_ratio", type=float, default=0.0, required=False)
    config = parser.parse_args()

    """
    read data
    """
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = True

    Solver = Solver_RCA_Multi(
        data_name=config.data,
        hidden_dim=config.hidden_dim,
        z_dim=config.z_dim,
        seed=config.seed,
        start_ratio=config.start_ratio,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        decay_ratio=config.decay,
        training_ratio=config.training_ratio,
        validation_ratio=config.validation_ratio,
        max_epochs=config.max_epochs,
        missing_ratio=config.missing_ratio,
        knn_impute=config.knn_impute,
        n_member=config.n_member,
        oe=config.oe,
    )

    Solver.train()
    Solver.test()
    print("Data {} finished".format(config.data))
