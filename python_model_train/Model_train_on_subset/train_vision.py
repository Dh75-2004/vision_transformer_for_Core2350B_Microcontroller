import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from useful_classes import VocTrainDataset, detection_collate_fn,Model
from Hungarian_loss import HungarianDETRLoss

# Configuration
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
NUM_IMAGES = 1000
NUM_CLASSES = 20
SAVE_DIRECTORY = "training_results3"
os.makedirs(SAVE_DIRECTORY, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# Train one epoch
def train_one_epoch(model,dataloader,criterion,optimizer,device,):
    model.train()
    running_total_loss = 0.0
    running_class_loss = 0.0
    running_bbox_loss = 0.0
    running_giou_loss = 0.0
    total_batches = len(dataloader)
    for batch_index, (images, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True,)
        optimizer.zero_grad()
        # Forward pass
        class_logits, predicted_boxes = model(images)
        '''outputs = {
            "pred_logits": class_logits,
            "pred_boxes": predicted_boxes,
        }'''
        # Hungarian loss
        loss_dictionary = criterion(class_logits, predicted_boxes, targets,)
        total_loss = loss_dictionary["loss"]
        # Backpropagation
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=0.1,) 
        optimizer.step()
        # Statistics
        running_total_loss += total_loss.item()
        running_class_loss += (loss_dictionary["classification_loss"].item())
        running_bbox_loss += (loss_dictionary["bbox_loss"].item())
        running_giou_loss += (loss_dictionary["giou_loss"].item())
        if (batch_index + 1) % 10 == 0:
            print(f"Batch "
                f"[{batch_index + 1}/{total_batches}] | "
                f"Loss: {total_loss.item():.4f} | "
                f"Class: "
                f"{loss_dictionary['classification_loss'].item():.4f} | "
                f"BBox: "
                f"{loss_dictionary['bbox_loss'].item():.4f} | "
                f"GIoU: "
                f"{loss_dictionary['giou_loss'].item():.4f}")
    return {"loss": running_total_loss / total_batches,
        "classification_loss": (running_class_loss / total_batches),
        "bbox_loss": (running_bbox_loss / total_batches),
        "giou_loss": (running_giou_loss / total_batches),}
# Validation
@torch.no_grad()
def validate(model, dataloader, criterion, device,):
    model.eval()
    running_total_loss = 0.0
    running_class_loss = 0.0
    running_bbox_loss = 0.0
    running_giou_loss = 0.0
    total_batches = len(dataloader)
    for images, targets in dataloader:
        images = images.to(device,non_blocking=True,)
        class_logits, predicted_boxes = model(images)
        outputs = {"pred_logits": class_logits, "pred_boxes": predicted_boxes,}
        loss_dictionary = criterion(class_logits, predicted_boxes, targets,)
        running_total_loss += ( loss_dictionary["loss"].item())
        running_class_loss += ( loss_dictionary["classification_loss"].item())
        running_bbox_loss += ( loss_dictionary["bbox_loss"].item())
        running_giou_loss += (loss_dictionary["giou_loss"].item())
    return {"loss": running_total_loss / total_batches, "classification_loss": (running_class_loss / total_batches),
        "bbox_loss": (running_bbox_loss / total_batches),
        "giou_loss": (running_giou_loss / total_batches),
    }
# Plot training curves
def plot_training_history(history, save_directory):
    epochs = range(1, len(history["train_loss"]) + 1,)
    # Total loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], label="Training Loss",)
    plt.plot(epochs, history["validation_loss"], label="Validation Loss",)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_directory, "total_loss.png",))
    plt.close()
    # Individual losses
    loss_names = ["classification", "bbox", "giou",]
    for loss_name in loss_names:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history[f"train_{loss_name}"], label=f"Training {loss_name}",)
        plt.plot(epochs, history[f"validation_{loss_name}"], label=f"Validation {loss_name}",)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{loss_name.upper()} Loss")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_directory, f"{loss_name}_loss.png",))
        plt.close()

# Main training function
def main():
    print("=" * 70)
    print("DETR HUNGARIAN LOSS TRAINING")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    # Load datasets
    train_dataset = VocTrainDataset(image_dir="data\Vocdevkit\VOC2007\JPEGImages2",annotation_dir="data\Vocdevkit\VOC2007\Annotations")
    validation_dataset = VocTrainDataset(image_dir="data\Vocdevkit\VOC2007\JPEGImages2",annotation_dir="data\Vocdevkit\VOC2007\Annotations")
    print(f"Original training images: " f"{len(train_dataset)}")
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,pin_memory=True, collate_fn=detection_collate_fn,)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True,
        collate_fn=detection_collate_fn,)
    # Create model
    model = Model().to(DEVICE)
    print(f"Trainable parameters: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    # Loss function
    criterion = HungarianDETRLoss(num_classes=NUM_CLASSES, no_object_weight=0.1, class_cost_weight=1.0, bbox_cost_weight=5.0,
        giou_cost_weight=2.0, bbox_loss_weight=5.0, giou_loss_weight=2.0,image_width=600,image_height=600).to(DEVICE)
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8, weight_decay=WEIGHT_DECAY,)
    # History
    history = {"train_loss": [],
        "validation_loss": [],
        "train_classification": [],
        "validation_classification": [],
        "train_bbox": [],
        "validation_bbox": [],
        "train_giou": [],
        "validation_giou": [],}
    best_validation_loss = float("inf")
    # Training loop
    for epoch in range(NUM_EPOCHS):
        checkpoint = torch.load("training_results3/best_model.pth",map_location=DEVICE,)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("\n" + "=" * 70)
        print(f"Epoch " f"{epoch + 1}/{NUM_EPOCHS}")
        print("=" * 70)
        #model.load_state_dict("training_result/best_model.pth")
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE,)
        validation_metrics = validate( model, validation_loader, criterion, DEVICE,)
        # Save history
        history["train_loss"].append(train_metrics["loss"])
        history["validation_loss"].append(validation_metrics["loss"])
        history["train_classification"].append(train_metrics["classification_loss"])
        history["validation_classification"].append(validation_metrics["classification_loss"])
        history["train_bbox"].append(train_metrics["bbox_loss"])
        history["validation_bbox"].append(validation_metrics["bbox_loss"])
        history["train_giou"].append(train_metrics["giou_loss"])
        history["validation_giou"].append(validation_metrics["giou_loss"])
        # Print epoch results
        print("\nEpoch Results")
        print(f"Train Loss:      "
            f"{train_metrics['loss']:.6f}")
        print(f"Validation Loss: "
            f"{validation_metrics['loss']:.6f}")
        # Save best model
        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = (validation_metrics["loss"])
            best_model_path = os.path.join(SAVE_DIRECTORY, "best_model.pth",)
            torch.save({"epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": best_validation_loss,},best_model_path,)
            print(f"Best model saved: "
                f"{best_model_path}")
        # Save checkpoint
        checkpoint_path = os.path.join(SAVE_DIRECTORY,"checkpoint.pth",)
        torch.save({"epoch": epoch + 1,"model_state_dict": model.state_dict(),"optimizer_state_dict": optimizer.state_dict(),
                "validation_loss": validation_metrics["loss"],},checkpoint_path,)
        # Clear unused GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Save history
    history_path = os.path.join(SAVE_DIRECTORY,"training_history.json",)
    with open(history_path, "w") as file:
        json.dump(history,file,indent=4,)
    # Plot graphs
    plot_training_history(history, SAVE_DIRECTORY,)
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Best validation loss: "
        f"{best_validation_loss:.6f}")

if __name__ == "__main__":
    main()