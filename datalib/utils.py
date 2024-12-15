import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm


def norm(Dataset, size=256):
    dataset = Dataset(
        transform=T.Compose([
            T.Resize((size,size)),
            T.ToTensor(),
        ]),
    )

    mean  = torch.zeros(3)
    std   = torch.zeros(3)
    for img, label in tqdm(DataLoader(dataset, num_workers=8, batch_size=8)):
        mean += img.mean(axis=(2,3)).sum(0)
        std  += img.std(axis=(2,3)).sum(0)

    mean /= len(dataset)
    std  /= len(dataset)

    return mean.tolist(), std.tolist()