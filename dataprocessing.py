import os
from typing import List

import torch
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

# 设置随机种子
torch.manual_seed(0)
np.random.seed(0)

class MultiviewData(Dataset):
    # def __init__(self, db, device, path="datasets/",):
    #     self.data_views = list()
    def __init__(self,
                 db: str,
                 device: torch.device,
                 path: str = "datasets/",
                 training: bool = True,
                 use_cache: bool = True,
                 ):
        super().__init__()
        self.device = device
        self.training = training
        self.data_views: List[np.ndarray] = []
        self.labels = None
        self.num_views = 0

        if db == "MSRCv1":
            mat = sio.loadmat(os.path.join(path, 'MSRCv1.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "ORL":
            mat = sio.loadmat(os.path.join(path, 'ORL.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "MNIST-USPS":
            mat = sio.loadmat(os.path.join(path, 'MNIST_USPS.mat'))
            X1 = mat['X1'].astype(np.float32)
            X2 = mat['X2'].astype(np.float32)
            self.data_views.append(X1.reshape(X1.shape[0], -1))
            self.data_views.append(X2.reshape(X2.shape[0], -1))
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "BDGP":
            mat = sio.loadmat(os.path.join(path, 'BDGP.mat'))
            X1 = mat['X1'].astype(np.float32)
            X2 = mat['X2'].astype(np.float32)
            self.data_views.append(X1)
            self.data_views.append(X2)
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "Fashion":
            mat = sio.loadmat(os.path.join(path, 'Fashion.mat'))
            X1 = mat['X1'].reshape(mat['X1'].shape[0], mat['X1'].shape[1] * mat['X1'].shape[2]).astype(np.float32)
            X2 = mat['X2'].reshape(mat['X2'].shape[0], mat['X2'].shape[1] * mat['X2'].shape[2]).astype(np.float32)
            X3 = mat['X3'].reshape(mat['X3'].shape[0], mat['X3'].shape[1] * mat['X3'].shape[2]).astype(np.float32)
            self.data_views.append(X1)
            self.data_views.append(X2)
            self.data_views.append(X3)
            self.num_views = len(self.data_views)
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "COIL20":
            mat = sio.loadmat(os.path.join(path, 'COIL20.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "hand":
            mat = sio.loadmat(os.path.join(path, 'handwritten.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "scene":
            mat = sio.loadmat(os.path.join(path, 'Scene15.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "NUSWIDEOBJ":
            mat = sio.loadmat(os.path.join(path, 'NUSWIDEOBJ.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "cifar10":
            mat = sio.loadmat(os.path.join(path, 'cifar10.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            scaler = MinMaxScaler()
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        else:
            raise NotImplementedError

        for idx in range(self.num_views):
            print(f"View {idx} data shape: {self.data_views[idx].shape}")
        print(f"Labels shape: {self.labels.shape}")
        print(f"Labels distribution: {np.bincount(self.labels)}")

        for idx in range(self.num_views):
            self.data_views[idx] = torch.from_numpy(self.data_views[idx]).to(device)

    def __len__(self):
        return len(self.labels)

    # def __getitem__(self, index):
    #     sub_data_views = list()
    #     for view_idx in range(self.num_views):
    #         data_view = self.data_views[view_idx][index]
    #         data_view = self.augment_data(data_view)
    #         sub_data_views.append(data_view)
    #     return sub_data_views, self.labels[index]

    def __getitem__(self, index):
        sub_views = []
        for v in range(self.num_views):
            x = self.data_views[v][index]
            if self.training:
                x = self.augment_data(x)
            sub_views.append(x)
        return sub_views, int(self.labels[index])

    # 外部可随时切换模式
    def set_training(self, mode: bool = True):
        self.training = mode

    def augment_data(self, data):
        # 添加噪声
        noise = torch.randn_like(data) * 0.01
        data = data + noise

        # 随机掩码
        mask = torch.bernoulli(torch.full(data.shape, 0.9)).to(data.device)
        data = data * mask

        # 随机丢弃
        if torch.rand(1).item() > 0.9:
            data = torch.zeros_like(data)

        # 添加更多的噪声或其他数据增强方法
        if torch.rand(1).item() > 0.5:
            data = data + torch.randn_like(data) * 0.005

        # 对数变换
        if torch.rand(1).item() > 0.5:
            data = torch.log1p(data)

        # 数据标准化
        data = (data - data.mean()) / (data.std() + 1e-9)

        # 检查是否有 NaN 或 Inf 值
        if torch.isnan(data).any() or torch.isinf(data).any():
            print("Data contains NaN or Inf values!")

        return data


def get_multiview_data(mv_data, batch_size):
    num_views = len(mv_data.data_views)
    num_samples = len(mv_data.labels)
    num_clusters = len(np.unique(mv_data.labels))

    mv_data_loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    return mv_data_loader, num_views, num_samples, num_clusters

def get_all_multiview_data(mv_data):
    num_views = len(mv_data.data_views)
    num_samples = len(mv_data.labels)
    num_clusters = len(np.unique(mv_data.labels))

    mv_data_loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=num_samples,
        shuffle=True,
        drop_last=False,
    )

    return mv_data_loader, num_views, num_samples, num_clusters
