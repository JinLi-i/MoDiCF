import torch
import numpy as np
import copy
from utils import *
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

def load_multimodal_dataset(dataset, MR, complete='zero', seed=0,
                            device=torch.device('cpu'), dtype=torch.float32,
                            normalize=False, reduced=False, dim=100):
    root_path = './data/'

    if dataset == 'tiktok':
        path = root_path + 'tiktok/'
        modalities = ['image', 'text', 'audio']

    elif dataset == 'baby':
        path = root_path + 'baby/'
        modalities = ['image', 'text']

    elif dataset == 'sports':
        path = root_path + 'sports/'
        modalities = ['image', 'text']

    elif dataset == 'allrecipes':
        path = root_path + 'allrecipes/'
        modalities = ['image', 'text']

    else:
        raise "Dataset not found"

    ori_data = []
    input_size = []
    min_data = []
    max_data = []
    for modality in modalities:
        fea = np.load(path + modality + '_feat.npy')
        ori_data.append(fea)
        input_size.append(fea.shape[1])
        min_data.append(np.min(fea))
        max_data.append(np.max(fea))
    n_sample = ori_data[0].shape[0]

    if reduced:
        before = copy.deepcopy(ori_data)
        svd = []
        input_size = [min(before[i].shape[1], dim) for i in range(len(modalities))]
        for i in range(len(modalities)):
            data, i_svd = reduce_dim(before[i], min(before[i].shape[1], dim), seed=seed)
            ori_data[i] = data
            svd.append(i_svd)

    if normalize:
        ori_data, min_data, max_data = normalize_data(ori_data)

    incomplete_data, missing_items, indicator = incomplete(ori_data, seed=seed, rate=MR, complete=complete, path=path)

    ori_data = [torch.tensor(data, dtype=dtype).to(device) for data in ori_data]
    incomplete_data = [torch.tensor(data, dtype=dtype).to(device) for data in incomplete_data]

    if not reduced:
        svd = None

    return ori_data, incomplete_data, modalities, n_sample, input_size, missing_items, indicator, svd, min_data, max_data

def normalize_data(data):
    min_data, max_data, norm_data = [], [], []
    for i in range(len(data)):
        min_val = np.min(data[i], axis=0)
        max_val = np.max(data[i], axis=0)
        norm_data.append((data[i] - min_val) / (max_val - min_val))
        min_data.append(min_val)
        max_data.append(max_val)
    return norm_data, min_data, max_data

def denormalize_data(data, min_data, max_data):
    original_data = []
    for i in range(len(data)):
        d = data[i] * (max_data[i] - min_data[i]) + min_data[i]
        original_data.append(d)
    return original_data

def save_multimodal_dataset(dataset, data, modalities):
    path = './gen_dataset/' + dataset + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    for i, modality in enumerate(modalities):
        np.save(path + modality + '_feat.npy', data[i])

def reduce_dim(data, dim, seed=0):
    svd = TruncatedSVD(n_components=dim, random_state=seed)
    reduced_data = svd.fit_transform(data)
    return reduced_data, svd

def transform_back(data, svd):
    ori_data = svd.inverse_transform(data)
    return ori_data

def incomplete(fea, seed=0, rate=0.3, complete='zero', path=''):
    np.random.seed(seed)
    complete = complete.lower()

    n_modality = len(fea)
    n_item = fea[0].shape[0]

    if complete == 'mean':
        replace = None
        print("Preprocess missing values with mean")
    elif complete == 'zero':
        replace = [np.zeros(data.shape[1]) for data in fea]
        print("Preprocess missing values with zero")
    elif complete == 'random':
        replace = [np.random.uniform(np.min(data, axis=0), np.max(data, axis=0), size=data.shape[1]) for data in fea]
        print("Preprocess missing values with random")
    elif complete == 'none':
        replace = [np.random.uniform(np.min(data, axis=0), np.max(data, axis=0), size=data.shape[1]) for data in fea]
        print("None preprocess")
    elif complete == 'nn':
        replace = None
        print("Preprocess missing values with Nearest Neighbor")
    else:
        raise "Unimplemented complete strategy"

    np.random.seed(seed)
    protected_indices = np.random.randint(n_modality, size=n_item)

    candidate_data = []

    for m in range(n_modality):
        for sample_idx in range(n_item):
            if protected_indices[sample_idx] != m:
                candidate_data.append((m, sample_idx))

    np.random.shuffle(candidate_data)
    num_missing_entries = int(n_item * n_modality * rate)
    selected_for_missing = candidate_data[:num_missing_entries]

    indicator = np.ones((n_item, n_modality), dtype=int)

    incomplete_data = [data.copy() for data in fea]
    for m, sample_idx in selected_for_missing:
        incomplete_data[m][sample_idx, :] = np.nan
        indicator[sample_idx, m] = 0

    if complete == 'mean':
        replace = [np.nanmean(data, axis=0) for data in incomplete_data]
    elif complete == 'nn':
        file = path + 'nn_complete_seed' + str(seed) + '_MR' + str(rate) + '.npy'
        if os.path.exists(file):
            loaded_data = np.load(file, allow_pickle=True)
            incomplete_data = [loaded_data[f'modality_{i}'] for i in range(len(loaded_data))]
            assert len(incomplete_data) == n_modality
        else:
            compressed_fea = []
            for m in range(n_modality):
                svd = TruncatedSVD(n_components=min(fea[m].shape[1], 128), random_state=seed)
                reduced_data = svd.fit_transform(fea[m])
                compressed_fea.append(reduced_data)

            complete_samples = np.all([~np.isnan(incomplete_data[i]).any(axis=1) for i in range(n_modality)], axis=0)

            nn_models = []
            for m in range(n_modality):
                search_space = compressed_fea[m][complete_samples]
                if search_space.shape[0] > 0:
                    nn = NearestNeighbors(n_neighbors=1)
                    nn.fit(search_space)
                    nn_models.append(nn)
                else:
                    nn_models.append(None)

            for sample_idx in range(n_item):
                if indicator[sample_idx].min() == 1:
                    continue

                available_modality = protected_indices[sample_idx]

                feature_vector = compressed_fea[available_modality][sample_idx].reshape(1, -1)

                nn = nn_models[available_modality]
                if nn is None:
                    for m in range(n_modality):
                        if indicator[sample_idx, m] == 0:
                            incomplete_data[m][sample_idx, :] = np.zeros(fea[m].shape[1])
                    continue

                _, indices = nn.kneighbors(feature_vector)

                nearest_idx = indices[0][0]

                global_nearest_idx = np.where(complete_samples)[0][nearest_idx]
                for m in range(n_modality):
                    if indicator[sample_idx, m] == 0:
                        incomplete_data[m][sample_idx, :] = fea[m][global_nearest_idx, :]

            np.savez(file, **{f'modality_{i}': data for i, data in enumerate(incomplete_data)})
        print("NN missing values imputed")

    for m, sample_idx in selected_for_missing:
        if complete != 'none' and complete != 'nn':
            incomplete_data[m][sample_idx] = replace[m]

    missing_items = {sample_idx for _, sample_idx in selected_for_missing}
    missing_items = list(missing_items)

    return incomplete_data, missing_items, indicator