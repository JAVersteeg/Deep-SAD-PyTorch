from .base_dataset import BaseADDataset
from torch.utils.data import DataLoader

class CrackDataset(BaseADDataset):
    """CrackDataset class for CRACK data set."""

    def __init__(self, root: str):
        super().__init__(root)

    def loaders(self, batch_size:int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
          DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                num_workers=num_workers)
        test_loader2 = DataLoader(dataset=self.test_set_corner, batch_size=batch_size, shuffle=shuffle_test,
                                num_workers=num_workers)
        return train_loader, test_loader, test_loader2