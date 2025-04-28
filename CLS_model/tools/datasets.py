import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Iterable, Dict, Union, List
import torch
from torch.utils.data.dataset import Dataset
from tools.general import load_json


class WSIDataset(Dataset):
    def __init__(self,
                 data_csv: Union[str, Path],
                 indices: Iterable[str] = None,
                 num_sample_patches: int = None,
                 fixed_size: bool = False,
                 shuffle: bool = False,
                 patch_random: bool = False) -> None:
        super(WSIDataset, self).__init__()
        self.data_csv = data_csv
        self.indices = indices
        self.num_sample_patches = num_sample_patches
        self.fixed_size = fixed_size
        self.patch_random = patch_random
        self.samples = self.process_data()
        if self.indices is None:
            self.indices = self.samples.index.values
        if shuffle:
            self.shuffle()
        self.patch_dim = np.load(self.samples.iat[0, 0])['img_features'].shape[-1]
        self.patch_features = self.load_patch_features()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        case_id = self.indices[index]
        patch_feature = self.patch_features[case_id]
        patch_feature = self.sample_feat(patch_feature)
        if self.fixed_size:
            patch_feature = self.fix_size(patch_feature)
        patch_feature = torch.as_tensor(patch_feature, dtype=torch.float32)

        label = self.samples.at[case_id, 'label']
        label = torch.tensor(label, dtype=torch.long)
        return patch_feature, label, case_id

    def shuffle(self) -> None:
        random.shuffle(self.indices)

    def process_data(self):
        data_csv = pd.read_csv(self.data_csv)
        data_csv.set_index(keys='case_id', inplace=True)
        if self.indices is not None:
            samples = data_csv.loc[self.indices]
        else:
            samples = data_csv
        return samples

    def load_patch_features(self) -> Dict[str, np.ndarray]:
        patch_features = {}
        for case_id in self.indices:
            patch_features[case_id] = np.load(self.samples.at[case_id, 'features_filepath'])['img_features']
        return patch_features

    def sample_feat(self, patch_feature: np.ndarray) -> np.ndarray:
        num_patches = patch_feature.shape[0]
        if self.num_sample_patches is not None and num_patches > self.num_sample_patches:
            sample_indices = np.random.choice(num_patches, size=self.num_sample_patches, replace=False)
            sample_indices = sorted(sample_indices)
            patch_feature = patch_feature[sample_indices]
        if self.patch_random:  # no
            np.random.shuffle(patch_feature)
        return patch_feature

    def fix_size(self, patch_feature: np.ndarray) -> np.ndarray:
        if patch_feature.shape[0] < self.num_sample_patches:
            margin = self.num_sample_patches - patch_feature.shape[0]
            feat_pad = np.zeros(shape=(margin, self.patch_dim))
            feat = np.concatenate((patch_feature, feat_pad))
        else:
            feat = patch_feature[:self.num_sample_patches]
        return feat


class WSIWithCluster(WSIDataset):

    def __init__(self,
                 data_csv: Union[str, Path],
                 indices: Iterable[str] = None,
                 num_sample_patches: int = None,
                 fixed_size: bool = False,
                 shuffle: bool = False,
                 patch_random: bool = False) -> None:
        super(WSIWithCluster, self).__init__(data_csv, indices, num_sample_patches, fixed_size, shuffle, patch_random)
        self.num_clusters = int(4)
        self.cluster_indices = self.load_cluster_indices()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, List[List[int]], torch.Tensor, str]:
        case_id = self.indices[index]
        patch_feature, cluster_indices = self.patch_features[case_id], self.cluster_indices[case_id]
        patch_feature = torch.as_tensor(patch_feature, dtype=torch.float32)
        label = self.samples.at[case_id, 'label']
        label = torch.tensor(label, dtype=torch.long)
        return patch_feature, cluster_indices, label, case_id

    def load_cluster_indices(self) -> Dict[str, List[List[int]]]:
        cluster_indices = {}
        for case_id in self.indices:
            cluster_indices[case_id] = load_json(self.samples.at[case_id, 'clusters_json_filepath'])
        return cluster_indices



def mixup(inputs: torch.Tensor, alpha: Union[float, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = inputs.shape[0]
    lambda_ = alpha + torch.rand(size=(batch_size, 1), device=inputs.device) * (1 - alpha)
    rand_idx = torch.randperm(batch_size, device=inputs.device)
    a = torch.stack([lambda_[i] * inputs[i] for i in range(batch_size)])
    b = torch.stack([(1 - lambda_[i]) * inputs[rand_idx[i]] for i in range(batch_size)])
    outputs = a + b
    return outputs, lambda_, rand_idx


def sample_feats(feat_list: List[torch.Tensor],
                 clusters_list: List[List[List[int]]],
                 feat_size: int = 1024) -> List[List[torch.Tensor]]:
    batch_size = len(feat_list)
    device = feat_list[0].device
    F_Ck = []
    for i in range(batch_size):
        sample_clusters = []
        num_patch = feat_list[i].shape[-2]
        sample_ratio = feat_size / num_patch
        num_feats_cluster = [len(c) for c in clusters_list[i]]
        for j, cluster in enumerate(clusters_list[i]):
            sample_size = round(num_feats_cluster[j] * sample_ratio)
            sample_size = max(1, min(sample_size, len(cluster)))
            if sample_size < len(cluster):
                perm = torch.randperm(len(cluster), device=device)
                selected_indices = [cluster[idx.item()] for idx in perm[:sample_size]]
            else:
                selected_indices = cluster

            cluster_feats = feat_list[i][:, selected_indices, :].squeeze(0)
            sample_clusters.append(cluster_feats)
        F_Ck.append(sample_clusters)

    return F_Ck


def fusion_features_weighted(F_Ck: List[List[torch.Tensor]],
                             weights: torch.Tensor,
                             feat_size: int = 1024) -> torch.Tensor:

    batch_size = len(F_Ck)
    result_feats = []
    for i in range(batch_size):
        batch_weights = weights[i]
        batch_clusters = F_Ck[i]
        batch_weights = torch.abs(batch_weights)
        normalized_weights = batch_weights / batch_weights.sum()

        weighted_clusters = []
        for j, cluster_feats in enumerate(batch_clusters):

            weighted_cluster = cluster_feats * normalized_weights[j]
            weighted_clusters.append(weighted_cluster)

        all_weighted_features = torch.cat(weighted_clusters, dim=0)  # [total_features, feature_dim]

        feature_dim = all_weighted_features.shape[-1]

        if all_weighted_features.shape[0] > feat_size:

            final_features = all_weighted_features[:feat_size]
        else:
            repeat_times = (feat_size + all_weighted_features.shape[0] - 1) // all_weighted_features.shape[0]
            repeated_features = torch.cat([all_weighted_features] * repeat_times, dim=0)
            final_features = repeated_features[:feat_size]

        result_feats.append(final_features.unsqueeze(0))

    return torch.cat(result_feats, dim=0)

