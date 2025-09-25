from multiprocessing import cpu_count
from torchvision import datasets, transforms
import torch
import multiprocessing


# def get_dataloader(root_path, image_size, batch_size, workers=8):
#     transform = transforms.Compose(
#         [
#             transforms.Resize(image_size),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ]
#     )

#     dataset = datasets.CIFAR10(
#         root=root_path, download=True, train=True, transform=transform
#     )

#     # Get indices of samples with label 2 (birds)
#     car_indices = [i for i, (_, label) in enumerate(dataset) if label == 1]

#     # Create a subset using those indices
#     car_subset = torch.utils.data.Subset(dataset, car_indices)

#     dataloader = torch.utils.data.DataLoader(
#         car_subset, batch_size=batch_size, shuffle=True, num_workers=workers,
#         pin_memory=True, persistent_workers=True
#     )

#     return dataloader

# def collate_fn(batch):
    
#     return (
#         torch.stack([x[0] for x in batch]), 
#         torch.tensor([x[1] for x in batch])
#     )


def get_dataloader(root_path, image_size, batch_size, workers=multiprocessing.cpu_count()):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset_train = datasets.StanfordCars(
        root=root_path, download=False, split='train', transform=transform
    )
    
    dataset_test = datasets.StanfordCars(
        root=root_path, download=False, split='test', transform=transform
    )

    dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_test])
    
    print(f"Using {workers} workers")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=True, 
        persistent_workers=True if workers > 0 else False,
#         collate_fn=collate_fn
    )

    return dataloader
