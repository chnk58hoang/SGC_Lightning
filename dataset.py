from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, feature_tensor, label_tensor):
        self.feature_tensor = feature_tensor
        self.label_tensor = label_tensor

    def __len__(self):
        return self.feature_tensor.size(0)

    def __getitem__(self, idx):
        return self.feature_tensor[idx], int(self.label_tensor[idx])
