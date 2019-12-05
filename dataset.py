
import torchvision.transforms as transforms
import torch

from PIL import Image
import os

#Run dierctory and load images
def get_images_from_path(root):
    images = []
    for _,_,filename in os.walk(root):
        for i, img_name in enumerate(filename):
            path = os.path.join(root, img_name)
            sample = Image.open(path)
            sample = sample.convert(mode='L')
            images.append(sample)
            if i > 250:
                break
    print('{} Images loaded'.format(len(images)))
    return images


#Custom dataset, only transform PIL to TENSOR
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.data = get_images_from_path(root)

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        target = self.data[index]
        if self.transform:
            target = self.transform(target)

        sample = target
        sample, target = transforms.ToTensor()(sample), transforms.ToTensor()(target)

        return sample, target



# the output of torchivision datasets are PIL images of range [0,1]
# gonna transform them to tensors [-1,1]
def get_dataset(batch_size, dataset, shuffle):
    path = '../datasets/' + dataset
    trainset = CustomDataset(root=path,
                             transform=None) #RGB images, loaded with PNG, with pixel values from 0 to 255
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=shuffle, num_workers=0)
    return trainloader