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

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        embd = 864
        heightofimage = 540
        widthofimage = 960
        Patchesalongheight = 30
        patchesalongwigth = 60
        heightofonepatch = 18
        widthofonepatch = 16
        self.query = nn.Parameter(torch.ones(100,embd, dtype=torch.float32))
        self.input = input_to_transformer_end(864,Patchesalongheight ,patchesalongwigth , embd)
        self.encoder1 = Transformer_encode(embd)
        self.decoder1 = Transformer_decoder(embd)
        self.classi = classification_head(embd, 10)
        self.boxhe = BoxHead(embd)

    def forward(self,x):
        x1 = self.input(x)
        x2 = self.encoder1(x1)
        x3 = self.decoder1(self.query, x2)
        xco = self.classi(x3)
        xbo = self.boxhe(x3)
        return xco,xbo

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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_data_image = os.path.join(BASE_DIR, "Dataset", "VisDrone2019-DET-train", "images")
train_data_label = os.path.join(BASE_DIR, "Dataset", "VisDrone2019-DET-train", "annotations")
val_data_image = os.path.join(BASE_DIR,"Dataset", "VisDrone2019-DET-val","images")
val_data_label = os.path.join(BASE_DIR, "Dataset", "VisDrone2019-DET-val", "annotations")
test_data_image = os.path.join(BASE_DIR, "Dataset", "VisDrone2019-DET-test-dev", "images")
test_data_label = os.path.join(BASE_DIR, "Dataset", "VisDrone2019-DET-test-dev", "annotations")
model = Model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = Model().to(device)
# CREATE DATASETS
train_dataset = VisDroneDataset( train_data_image, train_data_label)
val_dataset = VisDroneDataset(val_data_image, val_data_label)
test_dataset = VisDroneDataset(test_data_image,test_data_label)
# CREATE DATALOADERS
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, collate_fn=detection_collate_fn)
val_loader = DataLoader( val_dataset, batch_size=12, shuffle=False, collate_fn=detection_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False, collate_fn=detection_collate_fn)
print("*"*50)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
total_params = trainable_params + non_trainable_params
trainable_memory = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
non_trainable_memory = sum(p.numel() * p.element_size() for p in model.parameters() if not p.requires_grad)
total_memory = trainable_memory + non_trainable_memory
print("*" * 50)
print(f"Trainable parameters     : {trainable_params:,}")
print(f"Non-trainable parameters : {non_trainable_params:,}")
print(f"Total parameters         : {total_params:,}")
print()
print(f"Trainable memory         : {trainable_memory / (1024**2):.2f} MB")
print(f"Non-trainable memory     : {non_trainable_memory / (1024**2):.2f} MB")
print(f"Total parameter memory   : {total_memory / (1024**2):.2f} MB")
print("*"*50)
print()
print("#"*50)
print("Dataset info")
# DATASET SIZE
print("Train dataset:", len(train_dataset))
print("Validation dataset:", len(val_dataset))
print("#"*50)

#loss function
criterion = HungarianDETRLoss(
    num_classes=10,
    no_object_weight=0.1,
    class_cost_weight=1.0,
    bbox_cost_weight=5.0,
    giou_cost_weight=2.0,
    bbox_loss_weight=5.0,
    giou_loss_weight=2.0,
    image_width=960,
    image_height=540
).to(device)
checkpoint = torch.load(
    "checkpoint.pth",
    map_location=device
)
model.load_state_dict(checkpoint["model_state_dict"])
#optimizer
learning_rate = 1e-4
epochs = 10
total_loss_g = checkpoint["best_val_loss"]
optimizer = optim.AdamW(model.parameters(), lr = learning_rate, betas=(0.9,0.999), eps = 1e-8, weight_decay=0.01)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
print("ok till here")
for i in range(epochs):
    st = time.time()
    model.train()
    for batch_features, batch_labels in train_loader:
        #print(len(batch_features))
        #print(len(batch_features[0][0]))
        batch_features = batch_features.to(device)
        for target in batch_labels:
            target["boxes"] = target["boxes"].to(device)
            target["labels"] = target["labels"].to(device)
        #print(len(batch_labels[0]))
        #print(batch_labels[0]["labels"])
        clas, outbox = model(batch_features)
        #print("forward pass suceesful")
        #print(clas.shape)# output of print is torch.size([32,100,11])
        #print(outbox.shape)# output of print is torch.size([32,100,4])
        #print(batch_labels[0])
        #btach_labels is a list of dicitionary no of dicitionay equals to no of object in batch each dictionary have three 
        #keys boxes, labels, image_name
        losses = criterion(clas, outbox, batch_labels)
        loss = losses["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    st2 = time.time()
    print(f"time to train:{st2-st}")
    print("starting validation")
    total_loss = 0
    num_val_batches = 0
    model.eval()
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features = batch_features.to(device)
            for target in batch_labels:
                target["boxes"] = target["boxes"].to(device)
                target["labels"] = target["labels"].to(device)
            clas, outbox = model(batch_features)
            losses = criterion(
            clas,
            outbox,
            batch_labels
            )
            total_loss += losses["loss"].item()
            num_val_batches += 1
    avg_loss = total_loss / num_val_batches
    if avg_loss<total_loss_g:
        total_loss_g = avg_loss
        torch.save({
        "epoch": i + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": avg_loss}, "best_model.pth")
    print(f"Epoch {i+1}/{epochs} ",
          f"Avg_val_loss {avg_loss}")
torch.save({
    "epoch": i + 1,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_val_loss": total_loss_g,
    "val_loss": avg_loss,
}, "checkpoint.pth")
'''checkpoint = torch.load(
    "checkpoint.pth",
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

start_epoch = checkpoint["epoch"]
total_loss_g = checkpoint["best_val_loss"]

print("Resuming from epoch:", start_epoch)
print("Best validation loss:", total_loss_g)'''