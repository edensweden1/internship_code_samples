# imports
import rdex.datasets
from rdex.data_processing import SAR_processing
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import transforms

# defining a custom dataset class that inherits from torch.utils.data.Dataset
# (may need to be changed but in the meantime this should work)
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        """
        Args:
            data (list): List of (data, label) tuples.
        """
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
# adding channels to images
class Add_Channels:
    def __init__(self, n_channels=1):
        self.n_channels = n_channels

    def __call__(self, img: torch.Tensor):
        """Ensures the image has 'n_channels' and consistent dimensions."""
        if img.dim() == 2: # from (H, W) -> (1, H, W)
            img = img.unsqueeze(0)
        elif img.dim() == 3 and img.shape[2] == 1:
            img = img.permute(2, 0, 1) # from (H, W, 1) -> (1, H, W)
        img = img.repeat(self.n_channels, 1, 1) # From (1, H, W) -> (n_channels, H, W)
        return img


# TO DO: Transform for sparse data (KaST)
class Condense_Data:
    pass

    # ideas for data handling: 
        # option 1: replace nan with 0's or some other default value
        # option 2: min-max normalization
        # option 3: use Scipy COO or CSR
    # need to convert the encoded data to class labels


# for all SAR dataloaders with REAL data
class SAR_dataloader:
    def __init__(self, dset_names, n_channels=3, batch_size=32):
        """
        Initializes the SAR dataloader

        Args:
            dset_names (list): List of real SAR dataset names in rdex.datasets
            n_channels (int): Number of channels for image data.
            batch_size (int): Batch size for data loaders.
        """
        self.dset_names = dset_names
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.datasets = [getattr(rdex.datasets, name) for name in dset_names]
        self.transform_funcs = transforms.Compose([SAR_processing,Add_Channels(n_channels=self.n_channels),Condense_Data])
        self.label_dict = self._initialize_label_dict()

    def __len__(self):
        return len(self.data)
    
    def _initialize_label_dict(self):
        """
        Creates a unified global label dictionary for all datasets

        Returns:
            dict: Mapping of global_labels to indices
        """
        label_dicts = {}
        for dset_name in self.dset_names:
            dset = getattr(rdex.datasets, dset_name)
            labels = dset.label_dict

            # specific processing for certain datasets
            if dset_name == 'MAD98':
                try: # in a try except block since the code will keep trying to execute the
                     # the following code after the first time
                    labels['ZSU23'] = labels.pop('ZSU23-4')
                    labels['SA6TEL-3'] = labels.pop('SA-6')
                except KeyError:
                    pass # do nothing
                
            if dset_name == 'KaST':
                try:
                    labels['ZSU23'] = labels.pop('ZSU23-4')
                    # updating duplicate label indices so it is correctly 
                    # merged into the global dict
                    labels.update({
                        'man-made point-like': 16,
                        'man-made distributed': 17,
                        'man-made extended': 18,
                        'natural distributed': 19,
                        'vehicle class 1': 20,
                        'vehicle class 2': 21,
                    })
                except:
                    pass

            inv_labels = {v: k.lower() for k, v in labels.items() if k is not None}
            label_dicts[dset_name] = inv_labels

        # combine all datasets into a shared global label set
        global_labels = set()
        for labels in label_dicts.values():
            global_labels.update(labels.values())

        # create global label mapping
        return {idx: label for idx, label in enumerate(sorted(global_labels))}

    # note: the label mapping is not working for some reason and needs to be modified
    def _load_and_combine_datasets(self, dset):
        """
        Loads and concatenates datasets with the specified transformations

        Args:
            dataset: Dataset object from rdex.datasets
        
        Returns:
            dataset: Torch dataset object
        """
        SAR_dataset_list = []

        # Load the dataset
        if dset.__name__ == "rdex.datasets.MAD98":
            dataset = dset.load_as_torch(transforms=self.transform_funcs, load_all=True)
        else:
            dataset = dset.load_as_torch(transforms=self.transform_funcs)

        # print(f"Loaded dataset '{dset.__name__}': {len(dataset)} samples")
        print(f"Sample loaded from dataset '{dset.__name__}': {dataset[0]}")


        remapped_dataset = []
        for _, (data, label) in enumerate(dataset):

            if isinstance(label, torch.Tensor):
                if label.numel() == 1:
                    label = label.item()
                elif label.numel() > 1:
                    label = torch.argmax(label).item()

            if label is None:
                print(f"Warning: Skipping sample with None label in dataset '{dset.__name__}'")
                break

            # Map numeric labels to their names
            if isinstance(label, int):
                label = dset.label_dict.get(label)
                if label is None:
                    print(f"Label '{label}' not found in dataset label_dict!")
                    break

            # Map to global dictionary
            remapped_label = self.label_dict.get(label)
            if remapped_label is None:
                raise ValueError(f"Label '{label}' in dataset '{dset.__name__}' not found in global label dictionary!")

            if not remapped_dataset:
                raise ValueError(f"No valid samples found in dataset '{dset.__name__}'!")

            remapped_dataset.append((data, remapped_label))

        remapped_dataset = Dataset(remapped_dataset)
        SAR_dataset_list.append(remapped_dataset)

        SAR_dataset = ConcatDataset(SAR_dataset_list)

        return SAR_dataset

    
    def _split_data(self, data, train_split=0.64, val_split=0.16):
        """
        Splits the dataset into training, validation and test sets

        Args:
            dataset (Dataset): Torch dataset object
            train_split (float): Proportion for training set
            val_split (float): Proportion for validation set
        
        Returns:
            tuple: Training, validation, and test subsets.
        """
        print(f'Full dataloader size = {len(data)}')
        train_size = int(train_split * len(data))
        val_size = int(val_split * len(data))
        test_size = len(data) - train_size - val_size
        return random_split(data, [train_size, val_size, test_size])


    def convert_tensor_labels(self, data):
        """
        Converts tensor labels to integers if necessary

        Args:
            data (tuple): A sample containing an image and its label
        
        Returns:
            tuple: Transformed data sample
        """
        image, label = data
        if isinstance(label, torch.Tensor) and label.dim() == 0:
            label = label.item()
        return image, label

    def prepare_dataloaders(self, train, val, test, conversion_needed=False):
        """
        Prepares dataloaders for the train, validation, and test set.

        Args:
            train, val, test: Dataset subsets
            converison_needed (bool): Whether label conversion is required
        
        Returns:
            tuple: Train, validation, and test dataloaders.
        """
        if conversion_needed:
            train = [self.convert_tensor_labels(data) for data in train]
            val = [self.convert_tensor_labels(data) for data in val]
            test = [self.convert_tensor_labels(data) for data in test]
        
        train_loader = DataLoader(train, batch_size=self.batch_size)
        val_loader = DataLoader(val, batch_size=self.batch_size)
        test_loader = DataLoader(test, batch_size=self.batch_size)
        return train_loader, val_loader, test_loader
    
    def process_datasets(self):
        """
        Processes all datasets, generating train, validation, and test dataloaders

        Yields:
            tuple: Dataset name, train_loader, val_loader, test_loader, and number of classes
        """

        # use the pre-computed label dictionary to determine the total number of classes
        global_dict = self._initialize_label_dict()
        print(global_dict) # for debugging 
        num_classes = len(global_dict)
        print(f'The total number of classes across all datasets is: {num_classes}')
        
        for dataset_name in self.dset_names:
            dataset_module = getattr(rdex.datasets, dataset_name)
            loaded_data = self._load_and_combine_datasets(dataset_module)

            # split datasets and prepare dataloaders
            train, val, test = self._split_data(loaded_data)
            print(f'Training set: {len(train)}, Validation set: {len(val)}, Testing set: {len(test)}')

            train_loader, val_loader, test_loader = self.prepare_dataloaders(train, val, test)

        # if we just want this to return the one massive dataset change to a return statement
        # note: this functionality is from the previous iteration of the code
        # if changed to return from yield eliminate the try except blocks in
        # _initialize label dict

        yield dataset_name, train_loader, val_loader, test_loader, num_classes

    print('Data Loaded Successfully!')