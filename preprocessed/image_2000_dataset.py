from .image_2000_A_dataset import dataset as dataset_a
from .image_2000_B_dataset import dataset as dataset_b

dataset = dataset_a + dataset_b

if __name__ == "__main__":
    print(dataset.df)
    print(dataset[0])