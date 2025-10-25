import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from pycocotools.coco import COCO
from typing import List, Dict, Tuple


class ObjectRemovalDataset(Dataset):
    """
    Dataset cho object removal với bounding boxes và tags

    Dataset structure:
        dataset/
        ├── images/
        │   ├── img001.jpg
        │   ├── img002.jpg
        │   └── ...
        ├── masks/
        │   ├── img001.png
        │   ├── img002.png
        │   └── ...
        └── annotations.json

    annotations.json format:
    {
        "img001.jpg": {
            "style": "human",  # human, art, graphic_design
            "objects": [
                {
                    "bbox": [x1, y1, x2, y2],
                    "class": "person",
                    "class_id": 0,
                    "removed": true
                },
                {
                    "bbox": [x1, y1, x2, y2],
                    "class": "car",
                    "class_id": 2,
                    "removed": false
                }
            ]
        }
    }
    """

    def __init__(
            self,
            img_dir: str,
            mask_dir: str,
            annotation_file: str,
            img_size: int = 512,
            max_objects: int = 20,
            augment: bool = True
    ):
        """
        Args:
            img_dir: Thư mục chứa images
            mask_dir: Thư mục chứa ground truth masks
            annotation_file: File JSON chứa annotations
            img_size: Kích thước resize image
            max_objects: Số lượng objects tối đa (để padding)
            augment: Có dùng data augmentation không
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.max_objects = max_objects
        self.augment = augment

        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

        self.image_files = list(self.annotations.keys())

        # Build class vocabulary
        self.class_to_idx = self._build_class_vocab()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Build style vocabulary
        self.style_to_idx = {
            'human': 0,
            'art': 1,
            'graphic_design': 2
        }
        self.idx_to_style = {v: k for k, v in self.style_to_idx.items()}

        # Transforms
        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        self.mask_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])

        # Augmentation
        if augment:
            self.aug_transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ])
        else:
            self.aug_transform = None

    def _build_class_vocab(self) -> Dict[str, int]:
        """Build class vocabulary từ annotations"""
        classes = set()
        for anno in self.annotations.values():
            for obj in anno['objects']:
                classes.add(obj['class'])

        # Sort để đảm bảo consistent ordering
        class_list = sorted(list(classes))
        return {cls: idx for idx, cls in enumerate(class_list)}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size

        # Load mask
        mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path).convert('L')

        # Get annotations
        anno = self.annotations[img_name]
        objects = anno['objects']
        style_name = anno.get('style', 'human')  # Default to 'human' if not specified
        style_idx = self.style_to_idx.get(style_name, 0)

        # Filter only removed objects
        removed_objects = [obj for obj in objects if obj.get('removed', False)]

        # Prepare bboxes and classes
        bboxes = []
        classes = []

        for obj in removed_objects[:self.max_objects]:
            bbox = obj['bbox']  # [x1, y1, x2, y2]

            # Normalize bbox to [0, 1]
            x1, y1, x2, y2 = bbox
            x1_norm = x1 / orig_w
            y1_norm = y1 / orig_h
            x2_norm = x2 / orig_w
            y2_norm = y2 / orig_h

            bboxes.append([x1_norm, y1_norm, x2_norm, y2_norm])

            # Get class index
            class_name = obj['class']
            class_idx = self.class_to_idx.get(class_name, 0)
            classes.append(class_idx)

        # Padding to max_objects
        num_objects = len(bboxes)
        if num_objects < self.max_objects:
            # Pad với zeros
            bboxes += [[0.0, 0.0, 0.0, 0.0]] * (self.max_objects - num_objects)
            classes += [0] * (self.max_objects - num_objects)

        # Convert to tensors
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        classes = torch.tensor(classes, dtype=torch.long)

        # Apply augmentation (if training)
        if self.aug_transform is not None:
            seed = torch.randint(0, 2 ** 32, (1,)).item()

            torch.manual_seed(seed)
            image = self.aug_transform(image)

            torch.manual_seed(seed)
            mask = self.aug_transform(mask)

        # Apply transforms
        image = self.img_transform(image)
        mask = self.mask_transform(mask)

        # Binarize mask
        mask = (mask > 0.5).float()

        return {
            'image': image,
            'mask': mask,
            'bboxes': bboxes,
            'classes': classes,
            'style_idx': style_idx,
            'num_objects': num_objects,
            'image_name': img_name
        }


class COCOObjectRemovalDataset(Dataset):
    """
    Dataset dựa trên COCO format

    Advantages:
    - Tương thích với COCO pretrained models
    - Chuẩn hóa annotations
    - Dễ mở rộng
    """

    def __init__(
            self,
            img_dir: str,
            annotation_file: str,
            mask_dir: str = None,
            img_size: int = 512,
            max_objects: int = 20,
            augment: bool = True,
            remove_classes: List[str] = None
    ):
        """
        Args:
            img_dir: Thư mục chứa images
            annotation_file: COCO format annotation file
            mask_dir: Thư mục chứa ground truth masks (nếu có)
            img_size: Kích thước resize
            max_objects: Số objects tối đa
            augment: Data augmentation
            remove_classes: List các classes cần remove (None = all classes)
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.max_objects = max_objects
        self.augment = augment

        # Load COCO annotations
        self.coco = COCO(annotation_file)

        # Get image IDs that have annotations
        self.img_ids = list(self.coco.imgs.keys())

        # Filter by remove_classes if specified
        if remove_classes is not None:
            cat_ids = self.coco.getCatIds(catNms=remove_classes)
            self.cat_ids = set(cat_ids)
        else:
            self.cat_ids = None

        # Build class mapping
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(self.categories)}

        # Transforms
        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        self.mask_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        # Load image info
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Filter annotations
        if self.cat_ids is not None:
            anns = [ann for ann in anns if ann['category_id'] in self.cat_ids]

        # Prepare bboxes and classes
        bboxes = []
        classes = []
        masks_list = []

        for ann in anns[:self.max_objects]:
            # Get bbox [x, y, w, h] -> convert to [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            x1, y1, x2, y2 = x, y, x + w, y + h

            # Normalize
            x1_norm = x1 / orig_w
            y1_norm = y1 / orig_h
            x2_norm = x2 / orig_w
            y2_norm = y2 / orig_h

            bboxes.append([x1_norm, y1_norm, x2_norm, y2_norm])

            # Get class index
            cat_id = ann['category_id']
            class_idx = self.cat_id_to_idx[cat_id]
            classes.append(class_idx)

            # Get segmentation mask
            mask = self.coco.annToMask(ann)
            masks_list.append(mask)

        # Combine masks
        if len(masks_list) > 0:
            combined_mask = np.max(np.stack(masks_list, axis=0), axis=0)
            mask = Image.fromarray((combined_mask * 255).astype(np.uint8))
        else:
            mask = Image.new('L', (orig_w, orig_h), 0)

        # Padding
        num_objects = len(bboxes)
        if num_objects < self.max_objects:
            bboxes += [[0.0, 0.0, 0.0, 0.0]] * (self.max_objects - num_objects)
            classes += [0] * (self.max_objects - num_objects)

        # Convert to tensors
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        classes = torch.tensor(classes, dtype=torch.long)

        # Apply transforms
        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()

        return {
            'image': image,
            'mask': mask,
            'bboxes': bboxes,
            'classes': classes,
            'num_objects': num_objects,
            'image_id': img_id
        }


# ==================== UTILITY FUNCTIONS ====================

def collate_fn(batch):
    """
    Custom collate function để handle variable number of objects
    """
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    bboxes = torch.stack([item['bboxes'] for item in batch])
    classes = torch.stack([item['classes'] for item in batch])
    style_idx = torch.tensor([item['style_idx'] for item in batch], dtype=torch.long)
    num_objects = torch.tensor([item['num_objects'] for item in batch])

    return {
        'image': images,
        'mask': masks,
        'bboxes': bboxes,
        'classes': classes,
        'style_idx': style_idx,
        'num_objects': num_objects
    }


def create_sample_dataset(output_dir: str, num_samples: int = 10):
    """
    Tạo sample dataset để test
    """
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/masks", exist_ok=True)

    annotations = {}

    for i in range(num_samples):
        img_name = f"img{i:03d}.jpg"

        # Create random image
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        Image.fromarray(img).save(f"{output_dir}/images/{img_name}")

        # Create random mask
        mask = np.random.randint(0, 2, (512, 512), dtype=np.uint8) * 255
        Image.fromarray(mask).save(f"{output_dir}/masks/{img_name.replace('.jpg', '.png')}")

        # Create annotations
        num_objects = np.random.randint(1, 4)
        objects = []

        for j in range(num_objects):
            x1 = np.random.randint(0, 400)
            y1 = np.random.randint(0, 400)
            x2 = x1 + np.random.randint(50, 100)
            y2 = y1 + np.random.randint(50, 100)

            objects.append({
                'bbox': [x1, y1, x2, y2],
                'class': np.random.choice(['person', 'car', 'dog']),
                'class_id': np.random.randint(0, 3),
                'removed': bool(np.random.randint(0, 2))
            })

        # Random style
        style = np.random.choice(['human', 'art', 'graphic_design'])

        annotations[img_name] = {
            'style': style,
            'objects': objects
        }

    # Save annotations
    with open(f"{output_dir}/annotations.json", 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f"✅ Sample dataset created at {output_dir}")
    print(f"   - {num_samples} images")
    print(f"   - {num_samples} masks")
    print(f"   - 1 annotation file")


# ==================== TESTING ====================
if __name__ == "__main__":
    # Create sample dataset
    print("Creating sample dataset...")
    create_sample_dataset("./sample_dataset", num_samples=10)

    # Test dataset
    print("\nTesting dataset...")
    dataset = ObjectRemovalDataset(
        img_dir="./sample_dataset/images",
        mask_dir="./sample_dataset/masks",
        annotation_file="./sample_dataset/annotations.json",
        img_size=512,
        max_objects=20,
        augment=True
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Class vocabulary: {dataset.class_to_idx}")

    # Test dataloader
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    print("\nTesting dataloader...")
    for batch in dataloader:
        print(f"Batch shapes:")
        print(f"  - image: {batch['image'].shape}")
        print(f"  - mask: {batch['mask'].shape}")
        print(f"  - bboxes: {batch['bboxes'].shape}")
        print(f"  - classes: {batch['classes'].shape}")
        print(f"  - num_objects: {batch['num_objects']}")
        break

    print("\n✅ Dataset test successful!")