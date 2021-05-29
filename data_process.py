from six.moves import cPickle as pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import torch as torch
from torch.utils.data import Dataset


def load_dict(filename_):
    with open(filename_, "rb") as f:
        ret_di = pickle.load(f)
    return ret_di


class RealDataset(Dataset):
    def __init__(self, path, missing_ratio):
        scaler = MinMaxScaler()

        data = np.load(path, allow_pickle=True)
        data = data.item()
        self.missing_ratio = missing_ratio
        self.x = data["x"]
        self.y = data["y"]

        n, d = self.x.shape
        mask = np.random.rand(n, d)
        mask = (mask > self.missing_ratio).astype(float)
        if missing_ratio > 0.0:
            self.x[mask == 0] = np.nan
            imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
            self.x = imputer.fit_transform(self.x)
            scaler.fit(self.x)
            self.x = scaler.transform(self.x)
        else:
            scaler.fit(self.x)
            self.x = scaler.transform(self.x)

    def __len__(self):
        return self.x.shape[0]

    def __dim__(self):
        if len(self.x.shape) > 2:
            raise Exception("only handles single channel data")
        else:
            return self.x.shape[1]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(np.array(self.x[idx, :])),
            torch.from_numpy(np.array(self.y[idx])),
        )

    def __sample__(self, num):
        len = self.__len__()
        index = np.random.choice(len, num, replace=False)
        return self.__getitem__(index)

    def __anomalyratio__(self):
        return self.y.sum() / self.y.shape[0]
