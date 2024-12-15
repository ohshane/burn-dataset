from typing import List
from pathlib import Path

import pandas as pd
import PIL
from PIL import Image, ImageOps
from torch.utils.data import Dataset


class _ImageDatasetBase(Dataset):
    def __init__(
            self,
            meta_path: str,
            images_path: str,
            dataset_name: str,
        ):

        meta_path = Path(meta_path)
        images_path = Path(images_path)
        assert meta_path.exists()
        assert images_path.exists()
        self.meta_path = meta_path
        self.images_path = images_path
        self.dataset_name = dataset_name
    
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class _DataFrameImageDatasetBase(_ImageDatasetBase):
    def __init__(
            self,
            df_apply: callable = None,
            image_column: str = "file_name",
            image_convert: str = "RGB",
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)

        _df = pd.read_csv(self.meta_path)
        self._df = _df
        self.df_apply = df_apply
        self.image_column = image_column
        self.image_convert = image_convert

    @property
    def df(self):
        if self.df_apply is not None:
            return self.df_apply(self._df)
        return self._df
    
    def to_csv(self, *args, **kwargs):
        self.df.to_csv(*args, **kwargs)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx) -> tuple[PIL.Image.Image, dict]:
        row = self.df.iloc[idx]
        image = Image.open(self.images_path / row[self.image_column]).convert(self.image_convert)
        return image, dict(row)
    
    def __add__(self, other):
        df_left  = self.df.copy()
        df_right = other.df.copy()

        df_left["__dataset_name__"] = self.dataset_name
        df_right["__dataset_name__"] = other.dataset_name

        df = pd.concat([df_left, df_right], axis=0, join='outer')
        self._df = df
        self.dataset_name = f"{self.dataset_name}+{other.dataset_name}"
        self.df_apply = None

        print(f"[concat warning] attr dataset_name (str) is setted to {self.dataset_name}.")
        print(f"[concat warning] attr df_apply (callable) is setted to {self.df_apply}.")
        return self


class DataFrameImageDataset(_DataFrameImageDatasetBase):
    def __init__(
            self,
            return_transform: List[callable], 
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.return_transform = return_transform
    
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        results = []
        for t in self.return_transform:
            results.append(t(item))
        return results

    def __add__(self, other):
        super().__add__(other)
        self.return_transform = None

        print(f"[concat warning] attr return_transform (callable) is setted to {self.return_transform}.")
        return self


class JonathanMarkerDataset(DataFrameImageDataset):
    def __init__(
            self,
            name: str,
            root_path: str = "/datahub/home/lab/skinex_burn/data/jonathan-marker",
            *args, **kwargs
        ):
        """
        Args:
            name : should be one of [
                image_3_1
                image_740
                image_2000_A
                image_2000_B
                image_10000
            ]
        """
        
        root_path = Path(root_path)
        assert root_path.exists()

        super().__init__(
            dataset_name = name,
            meta_path = root_path / name / "meta.csv",
            images_path = root_path / name / "images",
            *args, **kwargs
        )
        assert name in [p.name for p in root_path.glob("*") if p.is_dir()]
  