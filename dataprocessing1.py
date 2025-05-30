# import os
# import torch
# import numpy as np
# import scipy.io as sio
# from sklearn.preprocessing import MinMaxScaler
# from torch.utils.data import Dataset
#
# # 设置随机种子
# torch.manual_seed(0)
# np.random.seed(0)
#
# class MultiviewData(Dataset):
#     def __init__(self, db, device, path="datasets/"):
#         self.data_views = list()
#
#         if db == "MSRCv1":
#             mat = sio.loadmat(os.path.join(path, 'MSRCv1.mat'))
#             X_data = mat['X']
#             self.num_views = X_data.shape[1]
#             for idx in range(self.num_views):
#                 self.data_views.append(X_data[0, idx].astype(np.float32))
#             scaler = MinMaxScaler()
#             for idx in range(self.num_views):
#                 self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
#             self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
#
#         elif db == "ORL":
#             mat = sio.loadmat(os.path.join(path, 'ORL.mat'))
#             X_data = mat['X']
#             self.num_views = X_data.shape[1]
#             for idx in range(self.num_views):
#                 self.data_views.append(X_data[0, idx].astype(np.float32))
#             scaler = MinMaxScaler()
#             for idx in range(self.num_views):
#                 self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
#             self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
#
#         elif db == "MNIST-USPS":
#             mat = sio.loadmat(os.path.join(path, 'MNIST_USPS.mat'))
#             X1 = mat['X1'].astype(np.float32)
#             X2 = mat['X2'].astype(np.float32)
#             self.data_views.append(X1.reshape(X1.shape[0], -1))
#             self.data_views.append(X2.reshape(X2.shape[0], -1))
#             self.num_views = len(self.data_views)
#             self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
#
#         elif db == "BDGP":
#             mat = sio.loadmat(os.path.join(path, 'BDGP.mat'))
#             X1 = mat['X1'].astype(np.float32)
#             X2 = mat['X2'].astype(np.float32)
#             self.data_views.append(X1)
#             self.data_views.append(X2)
#             self.num_views = len(self.data_views)
#             self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
#
#         elif db == "Fashion":
#             mat = sio.loadmat(os.path.join(path, 'Fashion.mat'))
#             X1 = mat['X1'].reshape(mat['X1'].shape[0], mat['X1'].shape[1] * mat['X1'].shape[2]).astype(np.float32)
#             X2 = mat['X2'].reshape(mat['X2'].shape[0], mat['X2'].shape[1] * mat['X2'].shape[2]).astype(np.float32)
#             X3 = mat['X3'].reshape(mat['X3'].shape[0], mat['X3'].shape[1] * mat['X3'].shape[2]).astype(np.float32)
#             self.data_views.append(X1)
#             self.data_views.append(X2)
#             self.data_views.append(X3)
#             self.num_views = len(self.data_views)
#             self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
#
#         elif db == "COIL20":
#             mat = sio.loadmat(os.path.join(path, 'COIL20.mat'))
#             X_data = mat['X']
#             self.num_views = X_data.shape[1]
#             for idx in range(self.num_views):
#                 self.data_views.append(X_data[0, idx].astype(np.float32))
#             scaler = MinMaxScaler()
#             for idx in range(self.num_views):
#                 self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
#             self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
#
#         elif db == "hand":
#             mat = sio.loadmat(os.path.join(path, 'handwritten.mat'))
#             X_data = mat['X']
#             self.num_views = X_data.shape[1]
#             for idx in range(self.num_views):
#                 self.data_views.append(X_data[0, idx].astype(np.float32))
#             scaler = MinMaxScaler()
#             for idx in range(self.num_views):
#                 self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
#             self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
#
#         elif db == "scene":
#             mat = sio.loadmat(os.path.join(path, 'Scene15.mat'))
#             X_data = mat['X']
#             self.num_views = X_data.shape[1]
#             for idx in range(self.num_views):
#                 self.data_views.append(X_data[0, idx].astype(np.float32))
#             scaler = MinMaxScaler()
#             for idx in range(self.num_views):
#                 self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
#             self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
#
#         elif db == "NUSWIDEOBJ":
#             mat = sio.loadmat(os.path.join(path, 'NUSWIDEOBJ.mat'))
#             X_data = mat['X']
#             self.num_views = X_data.shape[1]
#             for idx in range(self.num_views):
#                 self.data_views.append(X_data[0, idx].astype(np.float32))
#             scaler = MinMaxScaler()
#             for idx in range(self.num_views):
#                 self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
#             self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
#
#         elif db == "cifar10":
#             mat = sio.loadmat(os.path.join(path, 'cifar10.mat'))
#             X_data = mat['X']
#             self.num_views = X_data.shape[1]
#             for idx in range(self.num_views):
#                 self.data_views.append(X_data[0, idx].astype(np.float32))
#             scaler = MinMaxScaler()
#             for idx in range(self.num_views):
#                 self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
#             self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
#
#         else:
#             raise NotImplementedError
#
#         for idx in range(self.num_views):
#             print(f"View {idx} data shape: {self.data_views[idx].shape}")
#         print(f"Labels shape: {self.labels.shape}")
#         print(f"Labels distribution: {np.bincount(self.labels)}")
#
#         for idx in range(self.num_views):
#             self.data_views[idx] = torch.from_numpy(self.data_views[idx]).to(device)
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, index):
#         sub_data_views = list()
#         for view_idx in range(self.num_views):
#             data_view = self.data_views[view_idx][index]
#             data_view = self.augment_data(data_view)
#             sub_data_views.append(data_view)
#         return sub_data_views, self.labels[index]
#
#     def augment_data(self, data):
#         # 添加噪声
#         noise = torch.randn_like(data) * 0.01
#         data = data + noise
#
#         # 随机掩码
#         mask = torch.bernoulli(torch.full(data.shape, 0.9)).to(data.device)
#         data = data * mask
#
#         # 随机丢弃
#         if torch.rand(1).item() > 0.9:
#             data = torch.zeros_like(data)
#
#         # 添加更多的噪声或其他数据增强方法
#         if torch.rand(1).item() > 0.5:
#             data = data + torch.randn_like(data) * 0.005
#
#         # 对数变换
#         if torch.rand(1).item() > 0.5:
#             data = torch.log1p(data)
#
#         # 数据标准化
#         data = (data - data.mean()) / (data.std() + 1e-9)
#
#         # 检查是否有 NaN 或 Inf 值
#         if torch.isnan(data).any() or torch.isinf(data).any():
#             print("Data contains NaN or Inf values!")
#
#         return data
#
#
# def get_multiview_data(mv_data, batch_size):
#     num_views = len(mv_data.data_views)
#     num_samples = len(mv_data.labels)
#     num_clusters = len(np.unique(mv_data.labels))
#
#     mv_data_loader = torch.utils.data.DataLoader(
#         mv_data,
#         batch_size=batch_size,
#         shuffle=True,
#         drop_last=False,
#     )
#
#     return mv_data_loader, num_views, num_samples, num_clusters
#
# def get_all_multiview_data(mv_data):
#     num_views = len(mv_data.data_views)
#     num_samples = len(mv_data.labels)
#     num_clusters = len(np.unique(mv_data.labels))
#
#     mv_data_loader = torch.utils.data.DataLoader(
#         mv_data,
#         batch_size=num_samples,
#         shuffle=True,
#         drop_last=False,
#     )
#
#     return mv_data_loader, num_views, num_samples, num_clusters

import os
import pickle
import torch
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler        # 改为 μ-σ 标准化
from torch.utils.data import Dataset
from typing import List

torch.manual_seed(0)
np.random.seed(0)


def _fit_or_load_scalers(data_views: List[np.ndarray],
                         db_name: str,
                         path: str,
                         use_cache: bool = True):
    """
    对每个视图做 Standard-Scaler。
    为了避免重复 fit，可把 scaler 持久化到磁盘（datasets/<db>_scaler.pkl）。
    """
    scaler_path = os.path.join(path, f'{db_name}_scaler.pkl')
    scalers = []

    if use_cache and os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
    else:
        for v in range(len(data_views)):
            scaler = StandardScaler().fit(data_views[v])
            scalers.append(scaler)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scalers, f)

    # 变换
    for v in range(len(data_views)):
        data_views[v] = scalers[v].transform(data_views[v])
    return data_views


class MultiviewData(Dataset):
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

        # 在所有数据加载后再调用归一化
        # 假设这里有加载数据的代码

        # ---------- 读 .mat ----------
        if db == "MSRCv1":
            mat = sio.loadmat(os.path.join(path, 'MSRCv1.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "ORL":
            mat = sio.loadmat(os.path.join(path, 'ORL.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "MNIST-USPS":
            mat = sio.loadmat(os.path.join(path, 'MNIST_USPS.mat'))
            X1 = mat['X1'].astype(np.float32)
            X2 = mat['X2'].astype(np.float32)
            self.data_views.extend([X1.reshape(X1.shape[0], -1),
                                    X2.reshape(X2.shape[0], -1)])
            self.num_views = 2
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "BDGP":
            mat = sio.loadmat(os.path.join(path, 'BDGP.mat'))
            self.data_views.extend([mat['X1'].astype(np.float32),
                                    mat['X2'].astype(np.float32)])
            self.num_views = 2
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "Fashion":
            mat = sio.loadmat(os.path.join(path, 'Fashion.mat'))
            X1 = mat['X1'].reshape(mat['X1'].shape[0], -1).astype(np.float32)
            X2 = mat['X2'].reshape(mat['X2'].shape[0], -1).astype(np.float32)
            X3 = mat['X3'].reshape(mat['X3'].shape[0], -1).astype(np.float32)
            self.data_views.extend([X1, X2, X3])
            self.num_views = 3
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "COIL20":
            mat = sio.loadmat(os.path.join(path, 'COIL20.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "hand":
            mat = sio.loadmat(os.path.join(path, 'handwritten.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "scene":
            mat = sio.loadmat(os.path.join(path, 'Scene15.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "NUSWIDEOBJ":
            mat = sio.loadmat(os.path.join(path, 'NUSWIDEOBJ.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        elif db == "cifar10":
            mat = sio.loadmat(os.path.join(path, 'cifar10.mat'))
            X_data = mat['X']
            self.num_views = X_data.shape[1]
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)

        else:
            raise NotImplementedError

        # ---------- 统一标准化 ----------
        self.data_views = _fit_or_load_scalers(self.data_views, db, path, use_cache)
            # ---------- 应用标准化 ----------
        # if normalized:
        #     # 使用新添加的标准化方法
        #     self._normalize_views()
        # else:
        #         # 使用原有的标准化方法
        #     self.data_views = _fit_or_load_scalers(self.data_views, db, path, use_cache)

        # ---------- 转成 tensor ----------
        for idx in range(self.num_views):
            self.data_views[idx] = torch.from_numpy(self.data_views[idx]).to(device)

        # 打印信息
        for idx in range(self.num_views):
            print(f"View {idx} data shape: {self.data_views[idx].shape}")
        print(f"Labels shape: {self.labels.shape}")
        print(f"Labels distribution: {np.bincount(self.labels)}")


    # ---------- 标准化 ----------
    # def _normalize_views(self):
    #     """对所有视图进行标准化"""
    #     for i in range(len(self.data_views)):
    #         data = self.data_views[i]
    #         # 确保数据是Tensor
    #         if not isinstance(data, torch.Tensor):
    #             data = torch.tensor(data, device=self.device)
    #
    #         # 计算平均值和标准差
    #         mean = torch.mean(data, dim=0, keepdim=True)
    #         std = torch.std(data, dim=0, keepdim=True) + 1e-8  # 避免除零
    #
    #         # 应用z-score标准化
    #         self.data_views[i] = (data - mean) / std

    # ---------- 数据增广 ----------
    # def _augment(self, data: torch.Tensor) -> torch.Tensor:
    #     noise = torch.randn_like(data) * 0.01
    #     data = data + noise
    #     mask = torch.bernoulli(torch.full(data.shape, 0.9, device=data.device))
    #     data = data * mask
    #     if torch.rand(1, device=data.device).item() > 0.9:
    #         data = torch.zeros_like(data)
    #     if torch.rand(1, device=data.device).item() > 0.5:
    #         data = data + torch.randn_like(data) * 0.005
    #     if torch.rand(1, device=data.device).item() > 0.5:
    #         # clip 以免 log(负)
    #         data = torch.clamp(data, min=-0.99)
    #         data = torch.log1p(data)
    #     # 标准化
    #     std = data.std()
    #     data = (data - data.mean()) / (std + 1e-9)
    #     # 检查
    #     if torch.isnan(data).any() or torch.isinf(data).any():
    #         print("[Warning] sample contains NaN / Inf")
    #     return data
    # def _augment(self, data: torch.Tensor) -> torch.Tensor:
    #     # 减少增广强度
    #     if torch.rand(1).item() > 0.7:  # 降低增广概率
    #         noise = torch.randn_like(data) * 0.005  # 减小噪声强度
    #         data = data + noise
    #
    #     # 避免全零样本
    #     if torch.rand(1).item() > 0.95:  # 降低掩码概率
    #         mask = torch.bernoulli(torch.full(data.shape, 0.95, device=data.device))
    #         data = data * mask
    #
    #     # 移除对数变换或使用更安全的版本
    #     # 注释掉危险的对数变换
    #
    #     # 更安全的标准化
    #     std = data.std()
    #     if std > 1e-6:  # 只有标准差足够大才标准化
    #         data = (data - data.mean()) / std
    #
    #     return data
    def _augment(self, data: torch.Tensor) -> torch.Tensor:
        # 更温和的噪声
        noise = torch.randn_like(data) * 0.005  # 从0.01降到0.005
        data = data + noise

        # 更高的保留率
        mask = torch.bernoulli(torch.full(data.shape, 0.95, device=data.device))  # 从0.9提高到0.95
        data = data * mask

        # 移除完全置零的操作
        # if torch.rand(1, device=data.device).item() > 0.9:
        #     data = torch.zeros_like(data)

        # 移除额外的噪声
        # if torch.rand(1, device=data.device).item() > 0.5:
        #     data = data + torch.randn_like(data) * 0.005

        # 移除对数变换 - 这对MNIST-USPS这种数据不合适
        # if torch.rand(1, device=data.device).item() > 0.5:
        #     data = torch.clamp(data, min=-0.99)
        #     data = torch.log1p(data)

        # 不进行额外的标准化，因为数据已经预处理过
        # data = (data - data.mean()) / (data.std() + 1e-9)

        return data

    # ---------- Dataset 接口 ----------
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sub_views = []
        for v in range(self.num_views):
            x = self.data_views[v][index]
            if self.training:
                x = self._augment(x)
            sub_views.append(x)
        return sub_views, int(self.labels[index])

    # 外部可随时切换模式
    def set_training(self, mode: bool = True):
        self.training = mode



def get_multiview_data(mv_data,
                       batch_size: int,
                       shuffle: bool = True,
                       drop_last: bool = False):
    """
    生成常规 DataLoader
    """
    num_views = len(mv_data.data_views)
    num_samples = len(mv_data.labels)
    num_clusters = len(np.unique(mv_data.labels))

    loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return loader, num_views, num_samples, num_clusters


def get_all_multiview_data(mv_data):
    """
    一次性将整个数据集打成一个 batch（用于推理）
    """
    num_views = len(mv_data.data_views)
    num_samples = len(mv_data.labels)
    num_clusters = len(np.unique(mv_data.labels))

    loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=num_samples,
        shuffle=True,
        drop_last=False,
    )
    return loader, num_views, num_samples, num_clusters





