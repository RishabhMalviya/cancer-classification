# pylint: disable=arguments-differ
# pylint: disable=unused-argument
# pylint: disable=abstract-method
import os
import numpy as np
import lightning.pytorch as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

from cancer_classification.paths import RAW_DATA_DIR


class NCT_CRC_HE_100K__DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=3, split_dir_name='splits'):
        """
        Initialization of inherited lightning data module
        """
        super().__init__()

        self.full_set = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.split_dir_name = split_dir_name
        self.train_split_file_name = 'train_indices.npy'
        self.val_split_file_name = 'val_indices.npy'
        self.test_split_file_name = 'test_indices.npy'


        # transforms for images
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15)
        ])


    def _save_splits(self, train_indices, val_indices, test_indices):
        """
        Save the train, validation, and test splits' indices to disk
        """
        os.makedirs(self.split_dir_name, exist_ok=True)

        np.save(os.path.join(RAW_DATA_DIR, self.split_dir_name, self.train_split_file_name), train_indices)
        np.save(os.path.join(RAW_DATA_DIR, self.split_dir_name, self.val_split_file_name  ), val_indices  )
        np.save(os.path.join(RAW_DATA_DIR, self.split_dir_name, self.test_split_file_name ), test_indices )


    def _load_splits(self):
        """
        Load the train, validation, and test splits' indices from disk
        """
        train_indices = np.load(os.path.join(RAW_DATA_DIR, self.split_dir_name, self.train_split_file_name))
        val_indices   = np.load(os.path.join(RAW_DATA_DIR, self.split_dir_name, self.val_split_file_name  ))
        test_indices  = np.load(os.path.join(RAW_DATA_DIR, self.split_dir_name, self.test_split_file_name ))

        return train_indices, val_indices, test_indices


    def setup(self, stage=None):
        """
        Downloads the data, parse it and split it into train, validation, and test data

        :param stage: Stage - training or testing
        """
        # Load the dataset with transforms
        self.full_set = datasets.ImageFolder(RAW_DATA_DIR, transform=self.transform)

        # Split by indices or make new split & save indices
        indices_exist = [
            os.path.exists(os.path.join(self.split_dir_name, file_name))
            for file_name in [
                self.train_split_file_name,
                self.val_split_file_name,
                self.test_split_file_name
            ]
        ]
        if all(indices_exist):
            train_indices, val_indices, test_indices = self._load_splits()
        else:
            train_size = int(0.8 * len(self.full_set))
            val_size = int(0.1 * len(self.full_set))
            test_size = len(self.full_set) - train_size - val_size

            train_indices, val_indices, test_indices = random_split(range(len(self.full_set)), [train_size, val_size, test_size])

            self._save_splits(train_indices, val_indices, test_indices)

        # Create subsets using the indices
        self.train_set = Subset(self.full_set, train_indices)
        self.val_set = Subset(self.full_set, val_indices  )
        self.test_set = Subset(self.full_set, test_indices )


    def _create_data_loader(self, data_subset, shuffle=False):
        """
        Generic data loader function

        :param df: Input tensor

        :return: Returns the constructed dataloader
        """
        return DataLoader(data_subset, shuffle=shuffle, batch_size=self.batch_size, num_workers=self.num_workers)


    def train_dataloader(self):
        """
        :return: output - Train data loader for the given input
        """
        return self._create_data_loader(self.train_set, shuffle=True)


    def val_dataloader(self):
        """
        :return: output - Validation data loader for the given input
        """
        return self._create_data_loader(self.val_set)


    def test_dataloader(self):
        """
        :return: output - Test data loader for the given input
        """
        return self._create_data_loader(self.test_set)


    def predict_dataloader(self):
        """
        :return: output - Predict data loader for the given input
        """
        return self._create_data_loader(self.full_set)
