import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import math
import torch.nn as nn

IMAGE_HEIGHT = 540
IMAGE_WIDTH = 960
CHANNELS = 3
PATCH_HEIGHT = 18
PATCH_WIDTH = 16
NUM_PATCH_ROWS = IMAGE_HEIGHT // PATCH_HEIGHT
NUM_PATCH_COLS = IMAGE_WIDTH // PATCH_WIDTH
NUM_PATCHES = NUM_PATCH_ROWS * NUM_PATCH_COLS
PATCH_SIZE = PATCH_HEIGHT * PATCH_WIDTH * CHANNELS
print("Number of patch rows :", NUM_PATCH_ROWS)
print("Number of patch cols :", NUM_PATCH_COLS)
print("Number of patches    :", NUM_PATCHES)
print("Patch size           :", PATCH_SIZE)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_data_image = os.path.join(
    BASE_DIR,
    "Dataset",
    "VisDrone2019-DET-train",
    "images"
)
train_data_label = os.path.join(
    BASE_DIR,
    "Dataset",
    "VisDrone2019-DET-train",
    "annotations"
)
val_data_image = os.path.join(
    BASE_DIR,
    "Dataset",
    "VisDrone2019-DET-val",
    "images"
)
val_data_label = os.path.join(
    BASE_DIR,
    "Dataset",
    "VisDrone2019-DET-val",
    "annotations"
)
print("Train images:", train_data_image)
print("Train labels:", train_data_label)
print("Val images  :", val_data_image)
print("Val labels  :", val_data_label)

class VisDroneDataset(Dataset):
    def __init__(self, image_dir, annotation_dir):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_files = sorted([f for f in os.listdir(image_dir)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ])
        print("Number of images:", len(self.image_files))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):

        if index >= len(self.image_files):
            raise RuntimeError(
                f"Wrong index: {index}"
            )
        image_name = self.image_files[index]
        image_path = os.path.join(
            self.image_dir,
            image_name
        )
        annotation_name = (
            os.path.splitext(image_name)[0]
            + ".txt"
        )
        annotation_path = os.path.join(
            self.annotation_dir,
            annotation_name
        )
        image = Image.open(
            image_path
        ).convert("RGB")
        original_width, original_height = (image.size)# ORIGINAL IMAGE SIZE
        # CONVERT TO NUMPY
        img_matrix = np.array(image)
        # LOAD ANNOTATIONS FIRST
        boxes = []
        labels = []
        with open(annotation_path, "r") as f:
            for line in f:
                values = (line.strip().split(","))
                if len(values) < 8:
                    continue
                # VisDrone annotation format
                # x
                # y
                # width
                # height
                # score
                # category
                # truncation
                # occlusion
                x = int(values[0])
                y = int(values[1])
                w = int(values[2])
                h = int(values[3])
                score = int(values[4])
                category = int(values[5])
                if category == 0:
                    continue
                if category == 11:
                    continue
                # Convert xywh -> xyxy
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                boxes.append([x1, y1, x2, y2])
                # Convert class:
                # 1 -> 0
                # 2 -> 1
                # ...
                # 10 -> 9
                labels.append(category - 1)
        # RESIZE IMAGE
        image = image.resize((IMAGE_WIDTH,IMAGE_HEIGHT))
        # CALCULATE BOX SCALING
        scale_x = (IMAGE_WIDTH / original_width)
        scale_y = (IMAGE_HEIGHT / original_height)

        # SCALE BOUNDING BOXES
        scaled_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = x1 * scale_x
            y1 = y1 * scale_y
            x2 = x2 * scale_x
            y2 = y2 * scale_y
            scaled_boxes.append([x1, y1, x2, y2])
        # IMAGE -> NUMPY AFTER RESIZE
        img_matrix = np.array(image)
        # VERIFY IMAGE SIZE
        if img_matrix.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS):
            raise RuntimeError(
                "Unexpected image shape: "
                f"{img_matrix.shape}"
            )
        # PATCH EXTRACTION
        # Image:
        # [540, 960, 3]
        # ->
        # [30, 18, 60, 16, 3]
        # ->
        # [30, 60, 18, 16, 3]
        # ->
        # [1800, 864]
        patches = img_matrix.reshape(NUM_PATCH_ROWS,PATCH_HEIGHT,NUM_PATCH_COLS,PATCH_WIDTH,CHANNELS)
        print(patches.shape)
        patches = patches.transpose(0, 2, 1, 3,4)
        patch_matrix = patches.reshape(NUM_PATCHES, PATCH_SIZE)
        # CONVERT IMAGE TO TORCH
        image_tensor = torch.tensor(patch_matrix, dtype=torch.float32)
        # OPTIONAL NORMALIZATION
        # Current values:
        # 0 -> 255
        # After /255:
        # 0 -> 1
        image_tensor = (image_tensor / 255.0)

        if len(scaled_boxes) > 0:
            boxes_tensor = torch.tensor(scaled_boxes,dtype=torch.float32)
        else:
            boxes_tensor = torch.zeros(
                (0, 4),
                dtype=torch.float32
            )
        # LABELS TO TORCH
        if len(labels) > 0:
            labels_tensor = torch.tensor(labels,dtype=torch.long)

        else:

            labels_tensor = torch.zeros((0,), dtype=torch.long)
        # TARGET
        target = {"boxes": boxes_tensor, "labels": labels_tensor, "image_name": image_name}
        # RETURN
        return image_tensor, target

# CUSTOM COLLATE FUNCTION
# Different images can have different numbers of objects.
# Example:
# image 1 -> 80 boxes
# image 2 -> 15 boxes
# image 3 -> 42 boxes
# Therefore boxes cannot be stacked into:
# [batch, number_of_boxes, 4]
# Instead targets remain a list.
def detection_collate_fn(batch):

    images = []

    targets = []


    for image, target in batch:

        images.append(image)

        targets.append(target)


    # Images CAN be stacked because every image
    # has exactly 1800 × 864 after preprocessing.

    images = torch.stack(
        images,
        dim=0
    )
    return images, targets

# CREATE DATASETS
train_dataset = VisDroneDataset( train_data_image, train_data_label)
val_dataset = VisDroneDataset(val_data_image, val_data_label)
# CREATE DATALOADERS
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=detection_collate_fn)
val_loader = DataLoader( val_dataset, batch_size=32, shuffle=False, collate_fn=detection_collate_fn)
# DATASET SIZE
print("Train dataset:", len(train_dataset))
print("Validation dataset:", len(val_dataset))
# TEST ONE SAMPLE
image, target = train_dataset[0]
print("SINGLE SAMPLE")
print("Image shape:",image.shape)
print("Image dtype:",image.dtype)
print("Boxes shape:", target["boxes"].shape)
print("Labels shape:", target["labels"].shape)
print("Image name:", target["image_name"])
# TEST DATALOADER
print("TEST DATALOADER")
for train_image, data in train_loader:

    print(
        "Batch image shape:",
        train_image.shape
    )

    print("Batch image dtype:", train_image.dtype)
    print("Number of targets:", len(data))
    print("First image boxes:", data[0]["boxes"].shape)
    print("First image labels:", data[0]["labels"].shape)
    print("First image name:",data[0]["image_name"])
    break

def sinusoidal_2d_position_encoding(
    height_patches,
    width_patches,
    embed_dim
):
    """
    Returns:
        PE shape = [height_patches * width_patches, embed_dim]

    Example:
        height_patches = 30
        width_patches  = 60
        embed_dim      = 256

    Output:
        [1800, 256]
    """
    if embed_dim % 4 != 0:
        raise ValueError(
            "embed_dim must be divisible by 4"
        )
    half_dim = embed_dim // 2
    row_dim = half_dim
    col_dim = half_dim
    # Row encoding
    row_pe = torch.zeros(height_patches, row_dim)
    row_position = torch.arange(height_patches,dtype=torch.float32).unsqueeze(1)
    row_div_term = torch.exp(torch.arange(0, row_dim, 2, dtype=torch.float32)*(-math.log(10000.0) / row_dim))
    row_pe[:, 0::2] = torch.sin(row_position * row_div_term)
    row_pe[:, 1::2] = torch.cos(row_position * row_div_term)
    # Column encoding
    col_pe = torch.zeros(width_patches, col_dim)
    col_position = torch.arange(width_patches,dtype=torch.float32).unsqueeze(1)
    col_div_term = torch.exp(torch.arange(0, col_dim,2,dtype=torch.float32)*(-math.log(10000.0) / col_dim))
    col_pe[:, 0::2] = torch.sin(col_position * col_div_term)
    col_pe[:, 1::2] = torch.cos(col_position * col_div_term)
    # Combine row + column
    pe = torch.zeros(height_patches, width_patches, embed_dim)
    for row in range(height_patches):
        for col in range(width_patches):
            pe[row, col] = torch.cat(
                [row_pe[row], col_pe[col]]
            )
    pe = pe.reshape(height_patches * width_patches,embed_dim)
    return pe

def input_to_transformer_end():
    pass

class input_to_transformer_end(nn.Module):
    def __init__(self, image, height_patch, width_patch, embd_dim):
        super().__init__()
        self.a = nn.Linear(image.shape[1], embd_dim)
        self.poe = sinusoidal_2d_position_encoding(height_patch,width_patch, embd_dim)
        self.register_buffer("poe", self.poe)

    def forward(self, x):
        inter = self.a(x)
        return inter + self.poe

