import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from naslib.utils.custom_dataset import CustomDataset


class TabNasTorchDataset(Dataset):
    def __init__(self, root_dir, name, queue, norm="min_max"):
        super().__init__()

        bin_exists = False
        num_exists = False

        assert queue in ["train", "test", "val"]

        self.root_dir = root_dir
        self.name = name
        self.type = queue

        x_num = f"X_num_{queue}.npy"
        x_bin = f"X_bin_{queue}.npy"
        y = f"Y_{queue}.npy"

        if (root_dir / name / x_bin).exists():
            bin_exists = True

        if (root_dir / name / x_num).exists():
            num_exists = True

        self.y = np.load(root_dir / name / y)

        if num_exists:
            x_num = np.load(root_dir / name / x_num)
            x_num = (x_num - np.min(x_num, axis=0)) / (
                np.max(x_num, axis=0) - np.min(x_num, axis=0)
            )

        if bin_exists:
            x_bin = np.load(root_dir / name / x_bin)

        if num_exists and bin_exists:
            self.x = np.concatenate((x_num, x_bin), axis=1)
        elif num_exists:
            self.x = x_num
        elif bin_exists:
            self.x = x_bin
        else:
            raise NotImplementedError

        self.num_features = self.x.shape[1]
        self.num_classes = int(np.max(self.y)) + 1

    def __getitem__(self, i):
        x = self.x[i]
        y = self.y[i]

        return x, y.astype(np.int64)

    def __len__(self):
        return len(self.x)


class TabNasDataset(CustomDataset):
    def __init__(self, config, ds_train, ds_test, mode="train"):
        super().__init__(config, mode)
        self.ds_train = ds_train
        self.ds_test = ds_test

    def get_transforms(self, config):
        return Compose([ToTensor()]), Compose([ToTensor()])

    def get_data(self, data, train_transform, valid_transform):
        train_data = self.ds_train
        test_data = self.ds_test

        return train_data, test_data
