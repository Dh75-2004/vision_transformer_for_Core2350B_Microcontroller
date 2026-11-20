import os
import json
import numpy as np
import torch

from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from useful_classes import (
    VocTrainDataset,
    detection_collate_fn,
    Model,
)

from Hungarian_loss import HungarianDETRLoss


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 4

NUM_CLASSES = 20

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 600

CHECKPOINT_PATH = "training_results3/best_model.pth"

SAVE_DIRECTORY = "evaluation_results2"

# Detection confidence threshold
CONFIDENCE_THRESHOLD = 0.30

# Optional: maximum number of predictions per image
TOP_K = 100

os.makedirs(SAVE_DIRECTORY, exist_ok=True)


# ============================================================
# Box conversion
# Model output:
# normalized cx, cy, w, h
#
# Required by torchmetrics:
# pixel x1, y1, x2, y2
# ============================================================

def cxcywh_to_xyxy(boxes):

    cx = boxes[:, 0]
    cy = boxes[:, 1]

    w = boxes[:, 2]
    h = boxes[:, 3]

    x1 = cx - (w / 2)
    y1 = cy - (h / 2)

    x2 = cx + (w / 2)
    y2 = cy + (h / 2)

    return torch.stack(
        [x1, y1, x2, y2],
        dim=-1,
    )


def convert_predicted_boxes_to_pixels(boxes):

    # Convert normalized cxcywh -> normalized xyxy
    boxes = cxcywh_to_xyxy(boxes)

    # Convert normalized coordinates -> pixel coordinates
    boxes[:, [0, 2]] *= IMAGE_WIDTH
    boxes[:, [1, 3]] *= IMAGE_HEIGHT

    # Clamp boxes inside image
    boxes[:, 0] = boxes[:, 0].clamp(
        min=0,
        max=IMAGE_WIDTH,
    )

    boxes[:, 2] = boxes[:, 2].clamp(
        min=0,
        max=IMAGE_WIDTH,
    )

    boxes[:, 1] = boxes[:, 1].clamp(
        min=0,
        max=IMAGE_HEIGHT,
    )

    boxes[:, 3] = boxes[:, 3].clamp(
        min=0,
        max=IMAGE_HEIGHT,
    )

    return boxes


# ============================================================
# Calculate precision, recall and F1
#
# These are calculated at IoU = 0.50
# ============================================================

def calculate_detection_statistics(
    predictions,
    targets,
    iou_threshold=0.5,
):

    total_true_positive = 0
    total_false_positive = 0
    total_false_negative = 0

    for prediction, target in zip(
        predictions,
        targets,
    ):

        pred_boxes = prediction["boxes"]
        pred_labels = prediction["labels"]

        target_boxes = target["boxes"]
        target_labels = target["labels"]

        # No predictions
        if len(pred_boxes) == 0:

            total_false_negative += len(
                target_boxes
            )

            continue

        # No ground truth
        if len(target_boxes) == 0:

            total_false_positive += len(
                pred_boxes
            )

            continue

        matched_targets = set()

        for prediction_index in range(
            len(pred_boxes)
        ):

            best_iou = 0.0
            best_target_index = -1

            for target_index in range(
                len(target_boxes)
            ):

                # Class must match
                if (
                    pred_labels[prediction_index]
                    != target_labels[target_index]
                ):

                    continue

                # Calculate IoU
                pred_box = pred_boxes[
                    prediction_index
                ]

                target_box = target_boxes[
                    target_index
                ]

                x1 = max(
                    pred_box[0].item(),
                    target_box[0].item(),
                )

                y1 = max(
                    pred_box[1].item(),
                    target_box[1].item(),
                )

                x2 = min(
                    pred_box[2].item(),
                    target_box[2].item(),
                )

                y2 = min(
                    pred_box[3].item(),
                    target_box[3].item(),
                )

                intersection_width = max(
                    0,
                    x2 - x1,
                )

                intersection_height = max(
                    0,
                    y2 - y1,
                )

                intersection = (
                    intersection_width
                    * intersection_height
                )

                pred_area = max(
                    0,
                    (
                        pred_box[2]
                        - pred_box[0]
                    ).item()
                ) * max(
                    0,
                    (
                        pred_box[3]
                        - pred_box[1]
                    ).item()
                )

                target_area = max(
                    0,
                    (
                        target_box[2]
                        - target_box[0]
                    ).item()
                ) * max(
                    0,
                    (
                        target_box[3]
                        - target_box[1]
                    ).item()
                )

                union = (
                    pred_area
                    + target_area
                    - intersection
                )

                iou = (
                    intersection
                    / (union + 1e-8)
                )

                if iou > best_iou:

                    best_iou = iou

                    best_target_index = (
                        target_index
                    )

            # True positive
            if (
                best_iou >= iou_threshold
                and best_target_index
                not in matched_targets
            ):

                total_true_positive += 1

                matched_targets.add(
                    best_target_index
                )

            else:

                total_false_positive += 1

        # Unmatched ground-truth boxes
        total_false_negative += (
            len(target_boxes)
            - len(matched_targets)
        )

    precision = (
        total_true_positive
        / (
            total_true_positive
            + total_false_positive
            + 1e-8
        )
    )

    recall = (
        total_true_positive
        / (
            total_true_positive
            + total_false_negative
            + 1e-8
        )
    )

    f1_score = (
        2
        * precision
        * recall
        / (
            precision
            + recall
            + 1e-8
        )
    )

    return {
        "true_positives": total_true_positive,
        "false_positives": total_false_positive,
        "false_negatives": total_false_negative,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    criterion,
    device,
):

    model.eval()

    # --------------------------------------------------------
    # COCO-style mAP metric
    # --------------------------------------------------------

    map_metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
    )

    all_predictions = []
    all_targets = []

    # --------------------------------------------------------
    # Loss statistics
    # --------------------------------------------------------

    total_loss = 0.0
    total_classification_loss = 0.0
    total_bbox_loss = 0.0
    total_giou_loss = 0.0

    total_batches = 0
    total_predictions = 0
    total_ground_truths = 0

    # ========================================================
    # Loop through dataset
    # ========================================================

    for batch_index, (
        images,
        targets,
    ) in enumerate(dataloader):

        images = images.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Model inference
        # ----------------------------------------------------

        class_logits, predicted_boxes = model(
            images
        )

        # ----------------------------------------------------
        # Calculate validation loss
        # ----------------------------------------------------

        loss_dictionary = criterion(
            class_logits,
            predicted_boxes,
            targets,
        )

        total_loss += (
            loss_dictionary["loss"].item()
        )

        total_classification_loss += (
            loss_dictionary[
                "classification_loss"
            ].item()
        )

        total_bbox_loss += (
            loss_dictionary[
                "bbox_loss"
            ].item()
        )

        total_giou_loss += (
            loss_dictionary[
                "giou_loss"
            ].item()
        )

        total_batches += 1

        # ----------------------------------------------------
        # Convert logits to probabilities
        # ----------------------------------------------------

        probabilities = torch.softmax(
            class_logits,
            dim=-1,
        )

        # Object classes only
        #
        # Last class is no-object:
        # index 20 when NUM_CLASSES = 20
        object_probabilities = probabilities[
            :,
            :,
            :NUM_CLASSES
        ]

        scores, labels = torch.max(
            object_probabilities,
            dim=-1,
        )

        # ====================================================
        # Process every image in the batch
        # ====================================================

        for image_index in range(
            images.shape[0]
        ):

            image_scores = scores[
                image_index
            ]

            image_labels = labels[
                image_index
            ]

            image_boxes = predicted_boxes[
                image_index
            ]

            # ------------------------------------------------
            # Confidence filtering
            # ------------------------------------------------

            keep = (
                image_scores
                >= CONFIDENCE_THRESHOLD
            )

            image_scores = image_scores[keep]
            image_labels = image_labels[keep]
            image_boxes = image_boxes[keep]

            # ------------------------------------------------
            # Keep only top K predictions
            # ------------------------------------------------

            if (
                len(image_scores)
                > TOP_K
            ):

                top_scores, top_indices = (
                    torch.topk(
                        image_scores,
                        TOP_K,
                    )
                )

                image_scores = top_scores

                image_labels = image_labels[
                    top_indices
                ]

                image_boxes = image_boxes[
                    top_indices
                ]

            # ------------------------------------------------
            # Convert predicted boxes
            # ------------------------------------------------

            image_boxes = (
                convert_predicted_boxes_to_pixels(
                    image_boxes.clone()
                )
            )

            # ------------------------------------------------
            # Remove invalid boxes
            # ------------------------------------------------

            valid_boxes = (
                (image_boxes[:, 2]
                 > image_boxes[:, 0])
                &
                (image_boxes[:, 3]
                 > image_boxes[:, 1])
            )

            image_boxes = image_boxes[
                valid_boxes
            ]

            image_scores = image_scores[
                valid_boxes
            ]

            image_labels = image_labels[
                valid_boxes
            ]

            # ------------------------------------------------
            # Prediction dictionary
            # ------------------------------------------------

            prediction = {
                "boxes": (
                    image_boxes
                    .detach()
                    .cpu()
                ),
                "scores": (
                    image_scores
                    .detach()
                    .cpu()
                ),
                "labels": (
                    image_labels
                    .detach()
                    .cpu()
                    .long()
                ),
            }

            # ------------------------------------------------
            # Ground truth dictionary
            #
            # Dataset already provides pixel xyxy boxes
            # ------------------------------------------------

            target = {
                "boxes": (
                    targets[image_index]["boxes"]
                    .detach()
                    .cpu()
                ),
                "labels": (
                    targets[image_index]["labels"]
                    .detach()
                    .cpu()
                    .long()
                ),
            }

            # ------------------------------------------------
            # Remove invalid GT boxes
            # ------------------------------------------------

            if len(target["boxes"]) > 0:

                valid_gt = (
                    (
                        target["boxes"][:, 2]
                        > target["boxes"][:, 0]
                    )
                    &
                    (
                        target["boxes"][:, 3]
                        > target["boxes"][:, 1]
                    )
                )

                target["boxes"] = (
                    target["boxes"][valid_gt]
                )

                target["labels"] = (
                    target["labels"][valid_gt]
                )

            # ------------------------------------------------
            # Store results
            # ------------------------------------------------

            all_predictions.append(
                prediction
            )

            all_targets.append(
                target
            )

            total_predictions += len(
                prediction["boxes"]
            )

            total_ground_truths += len(
                target["boxes"]
            )

            # Update torchmetrics
            map_metric.update(
                [prediction],
                [target],
            )

        # Progress
        if (
            (batch_index + 1) % 10 == 0
        ):

            print(
                f"Evaluation Batch "
                f"[{batch_index + 1}/"
                f"{len(dataloader)}]"
            )

    # ========================================================
    # Compute mAP
    # ========================================================

    map_results = map_metric.compute()

    # ========================================================
    # Calculate Precision / Recall / F1
    # ========================================================

    detection_statistics = (
        calculate_detection_statistics(
            all_predictions,
            all_targets,
            iou_threshold=0.5,
        )
    )

    # ========================================================
    # Convert metrics to Python values
    # ========================================================

    metrics = {

        # ----------------------------------------------------
        # Dataset statistics
        # ----------------------------------------------------

        "images_evaluated": len(
            all_predictions
        ),

        "total_predictions": int(
            total_predictions
        ),

        "total_ground_truth_boxes": int(
            total_ground_truths
        ),

        "confidence_threshold": (
            CONFIDENCE_THRESHOLD
        ),

        # ----------------------------------------------------
        # Losses
        # ----------------------------------------------------

        "validation_loss": (
            total_loss
            / max(total_batches, 1)
        ),

        "classification_loss": (
            total_classification_loss
            / max(total_batches, 1)
        ),

        "bbox_loss": (
            total_bbox_loss
            / max(total_batches, 1)
        ),

        "giou_loss": (
            total_giou_loss
            / max(total_batches, 1)
        ),

        # ----------------------------------------------------
        # COCO mAP metrics
        # ----------------------------------------------------

        "map": float(
            map_results["map"].item()
        ),

        "map_50": float(
            map_results["map_50"].item()
        ),

        "map_75": float(
            map_results["map_75"].item()
        ),

        "map_small": float(
            map_results["map_small"].item()
        ),

        "map_medium": float(
            map_results["map_medium"].item()
        ),

        "map_large": float(
            map_results["map_large"].item()
        ),

        "mar_1": float(
            map_results["mar_1"].item()
        ),

        "mar_10": float(
            map_results["mar_10"].item()
        ),

        "mar_100": float(
            map_results["mar_100"].item()
        ),

        # ----------------------------------------------------
        # Detection statistics at IoU = 0.50
        # ----------------------------------------------------

        "precision_iou_50": float(
            detection_statistics["precision"]
        ),

        "recall_iou_50": float(
            detection_statistics["recall"]
        ),

        "f1_score_iou_50": float(
            detection_statistics["f1_score"]
        ),

        "true_positives_iou_50": int(
            detection_statistics[
                "true_positives"
            ]
        ),

        "false_positives_iou_50": int(
            detection_statistics[
                "false_positives"
            ]
        ),

        "false_negatives_iou_50": int(
            detection_statistics[
                "false_negatives"
            ]
        ),
    }

    # ========================================================
    # Per-class metrics
    # ========================================================

    if (
        "classes" in map_results
        and "map_per_class" in map_results
    ):

        classes = (
            map_results["classes"]
            .detach()
            .cpu()
            .numpy()
        )

        map_per_class = (
            map_results["map_per_class"]
            .detach()
            .cpu()
            .numpy()
        )

        per_class_map = {}

        for class_id, class_map in zip(
            classes,
            map_per_class,
        ):

            per_class_map[
                str(int(class_id))
            ] = float(class_map)

        metrics["map_per_class"] = (
            per_class_map
        )

    return metrics


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("DETR MODEL EVALUATION")
    print("=" * 70)

    print(f"Device: {DEVICE}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    evaluation_dataset = VocTrainDataset(
        image_dir=(
            "data/Vocdevkit/VOC2007/"
            "JPEGImages2"
        ),
        annotation_dir=(
            "data/Vocdevkit/VOC2007/"
            "Annotations"
        ),
    )

    print(
        f"Evaluation images: "
        f"{len(evaluation_dataset)}"
    )

    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=detection_collate_fn,
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = Model().to(DEVICE)

    trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

    print(
    f"Trainable parameters: "
    f"{trainable_parameters:,}"
)

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Checkpoint loaded from: "
        f"{CHECKPOINT_PATH}"
    )

    if "epoch" in checkpoint:

        print(
            f"Checkpoint epoch: "
            f"{checkpoint['epoch']}"
        )

    if "validation_loss" in checkpoint:

        print(
            f"Checkpoint validation loss: "
            f"{checkpoint['validation_loss']:.6f}"
        )

    # --------------------------------------------------------
    # Loss function
    # --------------------------------------------------------

    criterion = HungarianDETRLoss(
        num_classes=NUM_CLASSES,
        no_object_weight=0.1,
        class_cost_weight=1.0,
        bbox_cost_weight=5.0,
        giou_cost_weight=2.0,
        bbox_loss_weight=5.0,
        giou_loss_weight=2.0,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
    ).to(DEVICE)

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = evaluate_model(
        model,
        evaluation_loader,
        criterion,
        DEVICE,
    )

    # ========================================================
    # Print results
    # ========================================================

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"Images evaluated: "
        f"{metrics['images_evaluated']}"
    )

    print(
        f"Total predictions: "
        f"{metrics['total_predictions']}"
    )

    print(
        f"Ground truth boxes: "
        f"{metrics['total_ground_truth_boxes']}"
    )

    print("\nLOSS METRICS")

    print(
        f"Validation Loss: "
        f"{metrics['validation_loss']:.6f}"
    )

    print(
        f"Classification Loss: "
        f"{metrics['classification_loss']:.6f}"
    )

    print(
        f"BBox Loss: "
        f"{metrics['bbox_loss']:.6f}"
    )

    print(
        f"GIoU Loss: "
        f"{metrics['giou_loss']:.6f}"
    )

    print("\nDETECTION METRICS")

    print(
        f"mAP@[0.50:0.95]: "
        f"{metrics['map']:.6f}"
    )

    print(
        f"mAP@0.50: "
        f"{metrics['map_50']:.6f}"
    )

    print(
        f"mAP@0.75: "
        f"{metrics['map_75']:.6f}"
    )

    print(
        f"Precision@IoU=0.50: "
        f"{metrics['precision_iou_50']:.6f}"
    )

    print(
        f"Recall@IoU=0.50: "
        f"{metrics['recall_iou_50']:.6f}"
    )

    print(
        f"F1 Score@IoU=0.50: "
        f"{metrics['f1_score_iou_50']:.6f}"
    )

    # --------------------------------------------------------
    # Print per-class mAP
    # --------------------------------------------------------

    if "map_per_class" in metrics:

        print("\nPER-CLASS mAP")

        for class_id, class_map in (
            metrics["map_per_class"].items()
        ):

            print(
                f"Class {class_id}: "
                f"{class_map:.6f}"
            )

    # ========================================================
    # Save JSON
    # ========================================================

    json_path = os.path.join(
        SAVE_DIRECTORY,
        "evaluation_metrics.json",
    )

    with open(
        json_path,
        "w",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
        )

    print("\n" + "=" * 70)

    print(
        f"Results saved to: "
        f"{json_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()