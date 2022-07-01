import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class VMDataset(Dataset):

    def __init__(self, data, targets, classes, transform=transforms.ToTensor()):
        
        """Args:
             
             images (Numpy Array): Data
             targets (Numpy Array): targets corresponding to each data item
             transform (Optional): If any torch transform has to be performed on the dataset
             
        """
        
        "Attributes self.data and self.targets must be initialized."
        
        #<--Data must be initialized as self.data, self.train_data or self.test_data
        self.data = data
        #<--Targets must be initialized as self.targets, self.test_labels or self.train_labels
        self.targets = targets

        self.classes = classes

        #<--If any transforms have to be performed on the dataset
        self.transform = transform
        
    def __len__(self):
        
        """Required Method
            
           Returns:
        
                Length [int]: Length of Dataset/batches
        
        """
        
        return len(self.data)
    

    def __getitem__(self, idx):
        
        """Required Method
        
           The output of this method must be torch tensors since torch tensors are overloaded 
           with share() method which is used to share data to workers.
        
           Args:
                 
                 idx [integer]: The index of required batch/example
                 
           Returns:
                 
                 Data [Torch Tensor]:     The training examples
                 Target [ Torch Tensor]:  Corresponding labels of training examples 
        
        """
        
        sample = self.data[idx]
        target = int(self.targets[idx])
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

def load_datasets(dataset_type, data_path="../../dataset"):
    
    transform = load_default_transform(dataset_type)

    if dataset_type == 'CIFAR10':
        train_dataset = datasets.CIFAR10(data_path, train = True, download = True, transform=transform)
        test_dataset = datasets.CIFAR10(data_path, train = False, transform=transform)

    elif dataset_type == 'CIFAR100':
        train_dataset = datasets.CIFAR100(data_path, train = True, download = True, transform=transform)
        test_dataset = datasets.CIFAR100(data_path, train = False, transform=transform)

    return train_dataset, test_dataset

def load_default_transform(dataset_type):
    dataset_transform = []
    
    if dataset_type == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        dataset_transform = transforms.Compose([
                           transforms.ToTensor(),
                           normalize
                         ])

    elif dataset_type == 'CIFAR100':
        dataset_transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ])
                  
    return dataset_transform


def load_customized_transform(dataset_type):
    dataset_transform = []
    
    if dataset_type == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        dataset_transform = transforms.Compose([
                           transforms.ToPILImage(),
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomCrop(32, 4),
                           transforms.ToTensor(),
                           normalize
                         ])

    elif dataset_type == 'CIFAR100':
          dataset_transform = transforms.Compose([
                           transforms.ToPILImage(),
                           transforms.RandomHorizontalFlip(1.0),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ])
                       
    return dataset_transform