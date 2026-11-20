import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import math
import torch.nn as nn
import torch.optim as optim
from Layers import *
from Hungarian_loss import *
import time
import xml.etree.ElementTree as ET

IMAGE_HEIGHT = 600
IMAGE_WIDTH = 600
CHANNELS = 3
PATCH_HEIGHT = 20
PATCH_WIDTH = 20
NUM_PATCH_ROWS = IMAGE_HEIGHT // PATCH_HEIGHT
NUM_PATCH_COLS = IMAGE_WIDTH // PATCH_WIDTH
NUM_PATCHES = NUM_PATCH_ROWS * NUM_PATCH_COLS
PATCH_SIZE = PATCH_HEIGHT * PATCH_WIDTH * CHANNELS

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        embd = 600
        heightofimage = 600
        widthofimage = 600
        Patchesalongheight = 30
        patchesalongwidth = 30
        heightofonepatch = 20
        widthofonepatch = 20
        self.query=  nn.Parameter(torch.randn(50, embd) * 0.02)
        self.input = input_to_transformer_end(600,Patchesalongheight ,patchesalongwidth , embd)
        self.encoder1 = Transformer_encode(embd)
        self.decoder1 = Transformer_decoder(embd)
        self.classi = classification_head(embd, 20)
        self.boxhe = BoxHead(embd)

    def forward(self,x):
        #x0 = self.query(x)
        print(x.shape)
        x1 = self.input(x)
        x2 = self.encoder1(x1)
        print(x2.shape)
        x3 = self.decoder1(self.query, x2)
        xco = self.classi(x3)
        xbo = self.boxhe(x3)
        return xco,xbo


class VocTrainDataset(Dataset):
    def __init__(self, image_dir, annotation_dir):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        # Pascal VOC 20 object classes
        self.class_names = [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor"
        ]

        # Convert class name -> class index
        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(self.class_names)
        }

        # Find all image files
        all_image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        # Keep only images that have a corresponding XML annotation
        self.image_files = []

        for image_name in all_image_files:

            base_name = os.path.splitext(image_name)[0]

            annotation_name = base_name + ".xml"

            annotation_path = os.path.join(
                self.annotation_dir,
                annotation_name
            )

            if os.path.exists(annotation_path):
                self.image_files.append(image_name)

        print("Number of images:", len(self.image_files))

    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, index):
        if index >= len(self.image_files):
            raise RuntimeError(f"Wrong index: {index}")
        image_name = self.image_files[index]
        image_path = os.path.join(self.image_dir, image_name)
        base_name = os.path.splitext(image_name)[0]
        annotation_name = base_name + ".xml"
        annotation_path = os.path.join(self.annotation_dir, annotation_name)
        # 2. LOAD IMAGE
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size
        # 3. LOAD XML ANNOTATION
        boxes = []
        labels = []
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in self.class_to_idx:
                continue
            label = self.class_to_idx[class_name]
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # Ensure coordinates are valid
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(original_width, xmax)
            ymax = min(original_height, ymax)
            # Skip invalid boxes
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin,ymin,xmax,ymax])
            labels.append(label)
        # 4. RESIZE IMAGE
        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        scale_x = (IMAGE_WIDTH / original_width)
        scale_y = (IMAGE_HEIGHT / original_height)
        # 6. SCALE BOUNDING BOXES
        scaled_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x1 = x1 * scale_x
            y1 = y1 * scale_y
            x2 = x2 * scale_x
            y2 = y2 * scale_y
            scaled_boxes.append([x1, y1, x2, y2])
        # 7. IMAGE -> NUMPY
        img_matrix = np.array(image)
        # Verify image dimensions
        expected_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)
        if img_matrix.shape != expected_shape:
            raise RuntimeError("Unexpected image shape: " f"{img_matrix.shape}, " f"Expected: {expected_shape}")
        # 8. PATCH EXTRACTION
        # Example:
        # Image:
        # [540, 960, 3]
        # ->
        # [NUM_PATCH_ROWS, PATCH_HEIGHT,
        #  NUM_PATCH_COLS, PATCH_WIDTH, 3]
        # ->
        # [NUM_PATCH_ROWS, NUM_PATCH_COLS,
        #  PATCH_HEIGHT, PATCH_WIDTH, 3]
        # ->
        # [NUM_PATCHES, PATCH_SIZE]
        patches = img_matrix.reshape(NUM_PATCH_ROWS, PATCH_HEIGHT, NUM_PATCH_COLS, PATCH_WIDTH, CHANNELS)
        patches = patches.transpose(0, 2, 1, 3, 4)
        patch_matrix = patches.reshape(NUM_PATCHES, PATCH_SIZE)
        # 9. CONVERT PATCHES TO TORCH TENSOR
        image_tensor = torch.tensor(patch_matrix, dtype=torch.float32)
        # Normalize pixel values: 0-255 -> 0-1
        image_tensor = image_tensor / 255.0
        # 10. CONVERT BOXES TO TENSOR
        if len(scaled_boxes) > 0:
            boxes_tensor = torch.tensor(scaled_boxes, dtype=torch.float32)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        # 11. CONVERT LABELS TO TENSOR
        if len(labels) > 0:
            labels_tensor = torch.tensor(labels,dtype=torch.long)
        else:
            labels_tensor = torch.zeros((0,),dtype=torch.long)
        # 12. CREATE TARGET DICTIONARY
        target = {"boxes": boxes_tensor, "labels": labels_tensor, "image_name": image_name}
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
    images = torch.stack(images,dim=0)
    print(images.shape)
    return images, targets
