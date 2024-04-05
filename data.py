
'''class CustomDataset(Dataset):
   def __init__(self, h5_files):
       self.h5_files = h5_files
       self.data = []
       self.targets = []
       for h5_file in h5_files:
           with h5py.File(h5_file, 'r') as hf:
               data = hf['rgb'][:]
               data = np.transpose(data, (0, 1, 2, 3))
               self.data.append(data)
               targets = hf['targets'][:, 0]
               self.targets.append(targets)
       self.data = np.concatenate(self.data, axis=0)
       self.targets = np.concatenate(self.targets, axis=0)

   def __len__(self):
       return len(self.data)

   def __getitem__(self, index):
       image = self.data[index]
       angle = self.targets[index]
#       print("Image shape:", image.shape)
#       image_resized = cv2.resize(image, (320, 70))
#       print("Resized image shape:", image_resized.shape)
       return image, angle
       
 data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Lambda(lambda x: x.permute(1, 2, 0))
])      



train_h5_folder = '/home/wyl123/minidata/AgentHuman/SeqTrain/'
train_h5_files = [os.path.join(train_h5_folder, filename) for filename in os.listdir(train_h5_folder) if filename.endswith('.h5')]
train_h5_files = train_h5_files[:50]  


custom_dataset = CustomDataset(train_h5_files)'''


'''class CustomDataset(Dataset):
    def __init__(self, img_dir, txt_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        # Read text file with image names and steering angles
        with open(txt_file, 'r') as file:
            lines = file.readlines()
            self.image_names = [line.split()[0] for line in lines]  # Assuming image names are in the first column
            self.steering_angles = [float(line.split()[1]) for line in lines]  # Assuming angles are in the second column

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_names[idx])
        image = Image.open(img_path).convert('RGB')
        steering_angle = torch.tensor(self.steering_angles[idx], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, steering_angle

image_file = '/home/wyl123/minidata/driving_dataset/'
angle_file = '/home/wyl123/minidata/driving_dataset/data.txt'

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Lambda(lambda x: x.permute(1, 2, 0))
])

custom_dataset = CustomDataset(img_dir=image_file, txt_file=angle_file, transform=data_transform)'''

class CustomDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name = os.path.join(self.image_dir, self.data.iloc[index, 0])
        image = Image.open(img_name)
        angle = float(self.data.iloc[index, 3])

        if self.transform:
            image = self.transform(image)

        return image, angle


csv_file = '/home/wyl123/minidata/data1/driving_log.csv'
image_dir = '/home/wyl123/minidata/data1/IMG/'


data_transform = transforms.Compose([  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Lambda(lambda x: x.permute(1, 2, 0))
])


custom_dataset = CustomDataset(csv_file=csv_file, image_dir=image_dir, transform=data_transform)