import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class BackgroundDataset(Dataset):
    def __init__(self, root_dir, category="all", transform=None):
        """
        PyTorch Dataset for Background Images.

        Args:
            root_dir (str): 背景图的主目录 (例如 'data/background')
            category (str or list): 背景图类别. 比如 'desert', 'jungle', 'snow', 'sunset',
                                    或者是它们的列表如 ['desert', 'snow']。如果是 'all' 则返回所有。
            transform (callable, optional): 图像的变换操作.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        # 解析需要的类别
        if category == "all":
            self.categories = [
                d
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            ]
        elif isinstance(category, str):
            self.categories = [category]
        else:
            self.categories = category

        # 收集图片路径
        valid_extensions = (".png", ".jpg", ".jpeg", ".bmp")
        for cat in self.categories:
            cat_dir = os.path.join(self.root_dir, cat)
            if not os.path.exists(cat_dir):
                print(f"Warning: Category directory {cat_dir} does not exist.")
                continue
            for file in os.listdir(cat_dir):
                if file.lower().endswith(valid_extensions):
                    self.image_paths.append(os.path.join(cat_dir, file))

        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {root_dir} for categories {category}.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


def get_background_dataloader(
    root_dir, category="all", batch_size=4, img_size=(640, 640), num_workers=2,shaffle=True
):
    """
    获取 dataloader, 通过设置 shuffle=True 能够每次随机返回不同背景图片。
    """
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            # transforms.RandomCrop(
            #     img_size
            # ),  # 或者 transforms.CenterCrop() 视你的需求而定
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 根据需要解开注释
        ]
    )

    dataset = BackgroundDataset(
        root_dir=root_dir, category=category, transform=transform
    )

    # shuffle=True 保证了获取的时候是随机返回的
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shaffle,
        num_workers=num_workers,
        drop_last=True,
    )

    return dataloader


if __name__ == "__main__":
    # 测试代码
    root = os.path.join(os.path.dirname(__file__), "background")

    # 测试获取 desert 类的随机背景
    desert_loader = get_background_dataloader(
        root_dir=root, category="desert", batch_size=1
    )
    print(f"Loaded desert batches: {len(desert_loader)}")

    # 测试获取多个类别的随机背景
    mix_loader = get_background_dataloader(
        root_dir=root, category=["snow"], batch_size=1
    )
    print(f"Loaded mixed batches: {len(mix_loader)}")

    if len(desert_loader) > 0:
        for imgs in desert_loader:
            print(f"Sample batch shape: {imgs.shape}")
            break
