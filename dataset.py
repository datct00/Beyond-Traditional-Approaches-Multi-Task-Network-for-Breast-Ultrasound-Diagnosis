from lib import Dataset, os, cv2, train_test_split, torch

class BUSI(Dataset):
    def __init__(self, dataset_dir, input_size=(512,512), transform=None, target_transform=None, is_train=True):
        self.input_size = input_size
        self.dataset_dir = dataset_dir
        self.is_train = is_train
        if not os.path.exists(self.dataset_dir):
            raise ValueError('BUSI dataset not found at {}'.format(self.dataset_dir))

        for _, _, files in os.walk(self.dataset_dir):
            for file in files:
                if "_mask_1" in file:
                    raise Exception("This class requires BUSI dataset with combined mask. It can be done by running the BUSI() function in the process_data.py at utils folder")

        self.transform = transform
        self.target_transform = target_transform
        self.train_set, self.val_set = self._get_images()

        if self.is_train:
            self.images = self.train_set
        else:
            self.images = self.val_set


    def __len__(self):
        if self.is_train:
            return len(self.train_set)
        else:
            return len(self.val_set)

    def __getitem__(self, idx):
        label, image_path, mask_path = self.images[idx]
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)


        image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_NEAREST)


        #Normalize
        mask = mask/255
        mask = torch.from_numpy(mask).long()
        mask = torch.nn.functional.one_hot(mask, num_classes=2).permute(2,0,1).long()

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return image, mask, label

    @property
    def info(self):
        print(f"Dataset: BUSI")
        print(f"Train: {len(self.train_set)} images")
        print("-"*20)
        print(f"Benign: {len([image for image in self.train_set if image[0] == 0])} images")
        print(f"Malignant: {len([image for image in self.train_set if image[0] == 1])} images")
        print(f"Normal: {len([image for image in self.train_set if image[0] == 2])} images")
        print("-"*20)
        print(f"Val: {len(self.val_set)} images")
        print("-"*20)
        print(f"Benign: {len([image for image in self.val_set if image[0] == 0])} images")
        print(f"Malignant: {len([image for image in self.val_set if image[0] == 1])} images")
        print(f"Normal: {len([image for image in self.val_set if image[0] == 2])} images")
        print("-"*20)

    def _get_images(self):
        benign, malignant, normal = [], [], []
        benign_images = [os.path.join(self.dataset_dir, 'benign', file) for file in os.listdir(os.path.join(self.dataset_dir, 'benign')) if file.endswith('.png')]
        malignant_images = [os.path.join(self.dataset_dir, 'malignant', file) for file in os.listdir(os.path.join(self.dataset_dir, 'malignant')) if file.endswith('.png')]
        normal_images = [os.path.join(self.dataset_dir, 'normal', file) for file in os.listdir(os.path.join(self.dataset_dir, 'normal')) if file.endswith('.png')]

        for mask in benign_images:
            if "_mask" in mask:
                image = mask.replace('_mask.png', '.png')
                benign.append((0, image, mask))
        for mask in malignant_images:
            if "_mask" in mask:
                image = mask.replace('_mask.png', '.png')
                malignant.append((1, image, mask))
        for mask in normal_images:
            if "_mask" in mask:
                image = mask.replace('_mask.png', '.png')
                normal.append((2, image, mask))

        b_train_set, b_val_set = train_test_split(benign, test_size=0.2, random_state=42)
        m_train_set, m_val_set = train_test_split(malignant, test_size=0.2, random_state=42)
        n_train_set, n_val_set = train_test_split(normal, test_size=0.2, random_state=42)

        train_set = b_train_set + m_train_set + n_train_set
        val_set = b_val_set + m_val_set + n_val_set
        # # without normal class
        # train_set = b_train_set + m_train_set
        # val_set = b_val_set + m_val_set
        return train_set, val_set


