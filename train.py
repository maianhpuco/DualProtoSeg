import argparse
import datetime
import os
import numpy as np

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from src.model import ClsNetwork
from src.conch_adapter import ConchAdapter

from utils.hierarchical_utils import merge_to_parent_predictions
from utils.optimizer import PolyWarmupAdamW
from utils.pyutils import set_seed
from utils.trainutils import get_cls_dataset
from utils.validate import generate_cam, validate_valid, validate_test

from conch.open_clip_custom import create_model_from_pretrained
from torch.cuda.amp import GradScaler, autocast


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint")
    return parser.parse_args()


def get_device(gpu_id: int):
    return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")


def build_clip_model(cfg, device):
    clip_cfg = getattr(cfg, "clip", None)
    clip_cfg = OmegaConf.to_container(clip_cfg, resolve=True) if clip_cfg is not None else {}

    model_name = clip_cfg.get("model_name", "conch_ViT-B-16")
    checkpoint_path = clip_cfg.get("checkpoint_path")
    hf_hub = clip_cfg.get("hf_hub", "MahmoodLab/conch")

    # DIRECT CONCH LOADING
    model_conch, preprocess = create_model_from_pretrained(
        model_cfg=model_name,
        checkpoint_path=checkpoint_path,
        device=device,
        force_image_size=224,
        cache_dir="",
        hf_auth_token=None,
    )
    model_conch.eval()
    print("CONCH Loaded!")

    # 4 CLASSES — Detailed conceptual prompts (10 prompts per class → 40 prototypes total)
    bcss_conceptual_prompts = {
        0: [  # tumor
            "invasive carcinoma",
            "malignant epithelium",
            "pleomorphic nuclei",
            "solid tumor nests",
            "irregular glands",
            "high nuclear grade",
            "mitotic figures",
            "desmoplastic reaction",
            "angiogenesis",
            "tumor budding"
        ],
        1: [  # stroma
            "fibrous stroma",
            "collagen bundles",
            "spindle cells",
            "hyalinized stroma",
            "desmoplasia",
            "pink collagen",
            "fibroblasts",
            "loose connective tissue",
            "myxoid stroma",
            "scirrhous reaction"
        ],
        2: [  # inflammatory
            "lymphocytic infiltrate",
            "TILs",
            "plasma cells",
            "lymphoid aggregates",
            "peritumoral inflammation",
            "immune hotspot",
            "macrophages",
            "granulocytes",
            "tertiary lymphoid structure",
            "chronic inflammation"
        ],
        3: [  # necrosis
            "tumor necrosis",
            "necrotic debris",
            "ghost cells",
            "comedo necrosis",
            "karyorrhexis",
            "pink acellular areas",
            "central necrosis",
            "apoptotic bodies",
            "nuclear dust",
            "geographic necrosis"
        ]
    }
    class_prompts = bcss_conceptual_prompts

    # Adapter with learned prompts
    clip_adapter = ConchAdapter(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        device=device,
        class_prompts=class_prompts,
        prompt_n_ctx=16,
        prompt_position="end",
        freeze_conch=True,
        hf_hub=hf_hub,
    )
    clip_adapter.to(device)
    return clip_adapter


def build_dataloaders(cfg, num_workers):
    train_dataset, val_dataset = get_cls_dataset(cfg, split="valid")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.samples_per_gpu,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.samples_per_gpu,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def build_model(cfg, device, clip_adapter=None):
    # Detailed conceptual prompts dictionary
    bcss_conceptual_prompts = {
        0: [  # tumor
            "invasive carcinoma",
            "malignant epithelium",
            "pleomorphic nuclei",
            "solid tumor nests",
            "irregular glands",
            "high nuclear grade",
            "mitotic figures",
            "desmoplastic reaction",
            "angiogenesis",
            "tumor budding"
        ],
        1: [  # stroma
            "fibrous stroma",
            "collagen bundles",
            "spindle cells",
            "hyalinized stroma",
            "desmoplasia",
            "pink collagen",
            "fibroblasts",
            "loose connective tissue",
            "myxoid stroma",
            "scirrhous reaction"
        ],
        2: [  # inflammatory
            "lymphocytic infiltrate",
            "TILs",
            "plasma cells",
            "lymphoid aggregates",
            "peritumoral inflammation",
            "immune hotspot",
            "macrophages",
            "granulocytes",
            "tertiary lymphoid structure",
            "chronic inflammation"
        ],
        3: [  # necrosis
            "tumor necrosis",
            "necrotic debris",
            "ghost cells",
            "comedo necrosis",
            "karyorrhexis",
            "pink acellular areas",
            "central necrosis",
            "apoptotic bodies",
            "nuclear dust",
            "geographic necrosis"
        ]
    }
    class_prompts = bcss_conceptual_prompts
    model = ClsNetwork(
        backbone=cfg.model.backbone.config,
        stride=cfg.model.backbone.stride,
        cls_num_classes=cfg.dataset.cls_num_classes,
        clip_adapter=clip_adapter,
        pretrained=cfg.train.pretrained,
        enable_text_fusion=getattr(cfg.model, "enable_text_fusion", True),
        text_prompts=class_prompts,
        fusion_dim=getattr(cfg.model, "fusion_dim", None),
        # Enable spatial aggregation on all CAM scales (cam1–cam4) for v8
        spatial_agg_all_scales=True,
    )
    return model.to(device)


def build_optimizer(cfg, model, trainable_params):
    return PolyWarmupAdamW(
        params=trainable_params,
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        warmup_iter=cfg.scheduler.warmup_iter,
        max_iter=cfg.train.max_iters,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        power=cfg.scheduler.power,
    )


def resume(args, model, optimizer, device):
    start_epoch = 0
    best_metric = 0.0

    if args.resume is None:
        print("\nStarting training from scratch.")
        return start_epoch, best_metric

    if not os.path.exists(args.resume):
        print(f"WARNING: Checkpoint file not found at {args.resume}. Starting from scratch.")
        return start_epoch, best_metric

    print(f"\nResuming training from checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)

    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("Optimizer state loaded successfully.")
    else:
        print("WARNING: Optimizer state not found in checkpoint. Starting with a fresh optimizer.")

    if "epoch" in checkpoint:
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch: {start_epoch}")
    elif "iter" in checkpoint:
        # Fallback: compute epoch from iteration
        iters_per_epoch = checkpoint.get("iters_per_epoch", 1)
        start_epoch = (checkpoint["iter"] + 1) // iters_per_epoch
        print(f"Resuming from epoch: {start_epoch} (computed from iteration)")
    else:
        print("WARNING: Epoch number not found in checkpoint. Starting from epoch 0.")

    if "best_mIoU" in checkpoint:
        best_metric = checkpoint["best_mIoU"]
        print(f"Loaded previous best mIoU: {best_metric:.4f}")

    return start_epoch, best_metric


def build_loss_components(cfg, device, clip_adapter):
    """
    Build only the losses actually used in main8_backup_1_v3:
      - Image-level BCE-with-logits classification loss
    """
    loss_function = nn.BCEWithLogitsLoss().to(device)

    warmup_epochs = getattr(cfg.train, "learnable_prototype_warmup_epochs", 3)
    return {
        "cls": loss_function,
        "warmup_epochs": warmup_epochs,
    }


def save_best(model, optimizer, best_metric, cfg, epoch, current_metric, iters_per_epoch):
    if current_metric <= best_metric:
        return best_metric, False
    best_metric = current_metric
    save_path = os.path.join(cfg.work_dir.ckpt_dir, "best_cam.pth")
    torch.save(
        {
            "cfg": cfg,
            "epoch": epoch,
            "iter": epoch * iters_per_epoch,
            "iters_per_epoch": iters_per_epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_mIoU": best_metric,
        },
        save_path,
        _use_new_zipfile_serialization=True,
    )
    return best_metric, True


def train(cfg, args):
    print("\nInitializing training...")
    torch.backends.cudnn.benchmark = True
    set_seed(42)

    device = get_device(args.gpu)
    print(f"Using device: {device}")
    num_workers = min(10, os.cpu_count())

    clip_model = build_clip_model(cfg, device)
    time0 = datetime.datetime.now().replace(microsecond=0)

    print("\nPreparing datasets...")
    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(cfg, num_workers)

    iters_per_epoch = len(train_loader)
    cfg.train.max_iters = cfg.train.epoch * iters_per_epoch
    cfg.train.eval_iters = iters_per_epoch
    cfg.scheduler.warmup_iter = cfg.scheduler.warmup_iter * iters_per_epoch

    model = build_model(cfg, device, clip_adapter=clip_model)

    # Check trainable parameters before building optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    print(
        f"\nModel parameters: {trainable_count:,} trainable / {total_params:,} total "
        f"({100 * trainable_count / total_params:.2f}%)"
    )

    if len(trainable_params) == 0:
        raise ValueError("ERROR: No trainable parameters found! Check if model is properly initialized.")

    # Build optimizer with ONLY trainable parameters
    optimizer = build_optimizer(cfg, model, trainable_params)

    start_epoch, best_miou = resume(args, model, optimizer, device)

    losses = build_loss_components(cfg, device, clip_model)
    loss_function = losses["cls"]
    scaler = GradScaler()
    model.train()

    print("\nStarting training...")

    for epoch in range(start_epoch, cfg.train.epoch):
        phase_str = f"[TRAINING {epoch + 1}/{cfg.train.epoch}]"

        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{cfg.train.epoch} {phase_str}")
        print(f"{'='*80}")
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_batch_count = 0
        all_train_preds = []
        all_train_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", ncols=120, leave=False)

        for batch_idx, (_, inputs, cls_labels, _) in enumerate(pbar):
            n_iter = epoch * iters_per_epoch + batch_idx

            inputs = inputs.to(device).float()
            cls_labels = cls_labels.to(device).float()

            with autocast():
                # COMMENTED OUT: Not using contrastive loss, so don't pass labels
                # model(inputs, labels=cls_labels) -> model(inputs)
                (
                    cls1,
                    cam1,
                    cls2,
                    cam2,
                    cls3,
                    cam3,
                    cls4,
                    cam4,
                    l_fea,
                    k_list,
                    feature_map_for_diversity,
                    cam_weights,
                    projected_prototypes,
                    text_features_out,
                    contrastive_loss,  # Will be None when labels not passed
                ) = model(inputs)  # COMMENTED OUT: labels=cls_labels

                cls1_merge = merge_to_parent_predictions(cls1, k_list, method=cfg.train.merge_train)
                cls2_merge = merge_to_parent_predictions(cls2, k_list, method=cfg.train.merge_train)
                cls3_merge = merge_to_parent_predictions(cls3, k_list, method=cfg.train.merge_train)
                cls4_merge = merge_to_parent_predictions(cls4, k_list, method=cfg.train.merge_train)

                loss1 = loss_function(cls1_merge, cls_labels)
                loss2 = loss_function(cls2_merge, cls_labels)
                loss3 = loss_function(cls3_merge, cls_labels)
                loss4 = loss_function(cls4_merge, cls_labels)

                cls_loss = (
                    cfg.train.l1 * loss1
                    + cfg.train.l2 * loss2
                    + cfg.train.l3 * loss3
                    + cfg.train.l4 * loss4
                )

                loss = cls_loss
                
            if (
                loss is None
                or not torch.is_tensor(loss)
                or not loss.requires_grad
                or not torch.isfinite(loss)
            ):
                if loss is not None and torch.is_tensor(loss) and not torch.isfinite(loss):
                    print(f"WARNING: loss is not finite at epoch {epoch+1} batch {batch_idx}, skipping")
                continue

            optimizer.zero_grad(set_to_none=True)
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            cls_pred4 = (torch.sigmoid(cls4_merge) > 0.5).float()
            all_cls_acc4 = (cls_pred4 == cls_labels).all(dim=1).float().mean() * 100
            train_acc_sum += all_cls_acc4.item()
            train_batch_count += 1

            all_train_preds.append(torch.sigmoid(cls4_merge).detach().cpu().numpy())
            all_train_labels.append(cls_labels.detach().cpu().numpy())

            cur_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{cur_lr:.2e}",
                    "acc": f"{all_cls_acc4:.1f}%",
                }
            )

        avg_train_loss = train_loss_sum / train_batch_count if train_batch_count > 0 else 0.0
        avg_train_acc = train_acc_sum / train_batch_count if train_batch_count > 0 else 0.0

        if len(all_train_preds) > 0:
            train_preds = np.concatenate(all_train_preds, axis=0)
            train_labels = np.concatenate(all_train_labels, axis=0)
            try:
                train_auc = roc_auc_score(train_labels, train_preds, average="macro")
            except Exception:
                train_auc = 0.0
        else:
            train_auc = 0.0

        # Validation without CRF
        val_mIoU, val_mean_dice, val_fw_iu, val_iu_per_class, val_dice_per_class = validate_valid(
            model=model,
            data_loader=val_loader,
            cfg=cfg,
            cls_loss_func=loss_function,
        )

        # Compute validation loss, accuracy, and AUC
        model.eval()
        val_preds_list = []
        val_labels_list = []
        val_loss_sum = 0.0
        val_batch_count = 0
        val_correct_all = 0
        val_total_samples = 0

        with torch.no_grad():
            for _, inputs, cls_labels, _ in val_loader:
                inputs = inputs.to(device).float()
                cls_labels = cls_labels.to(device).float()

                with autocast():
                    outputs = model(inputs)
                    cls1 = outputs[0]
                    cls4 = outputs[6]
                    k_list = outputs[9]
                    cls1_merge = merge_to_parent_predictions(cls1, k_list, method=cfg.train.merge_test)
                    cls4_merge = merge_to_parent_predictions(cls4, k_list, method=cfg.train.merge_test)
                    val_cls_loss = loss_function(cls4_merge, cls_labels)
                    val_loss_sum += val_cls_loss.item()
                    val_batch_count += 1

                    cls_pred4 = (torch.sigmoid(cls4_merge) > 0.5).float()
                    all_correct = (cls_pred4 == cls_labels).all(dim=1).float()
                    val_correct_all += all_correct.sum().item()
                    val_total_samples += all_correct.shape[0]

                    val_preds_list.append(torch.sigmoid(cls4_merge).cpu().numpy())
                    val_labels_list.append(cls_labels.cpu().numpy())

        model.train()

        avg_val_loss = val_loss_sum / val_batch_count if val_batch_count > 0 else 0.0
        val_acc = (val_correct_all / val_total_samples * 100) if val_total_samples > 0 else 0.0

        if len(val_preds_list) > 0:
            val_preds = np.concatenate(val_preds_list, axis=0)
            val_labels = np.concatenate(val_labels_list, axis=0)
            try:
                val_auc = roc_auc_score(val_labels, val_preds, average="macro")
            except Exception:
                val_auc = 0.0
        else:
            val_auc = 0.0

        # Compact training metrics (1 line)
        print(f"\nTraining Metrics: Loss={avg_train_loss:.4f} | Acc={avg_train_acc:.2f}% | AUC={train_auc:.4f}")

        # Compact validation metrics (1 line)
        print(f"Validation Metrics: Loss={avg_val_loss:.4f} | Acc={val_acc:.2f}% | AUC={val_auc:.4f}")

        # Segmentation metrics (compact format)
        print(f"\nSegmentation Metrics: >>  mIoU: {val_mIoU:.4f}")
        print(f"mDice: {val_mean_dice:.4f}, FwIU: {val_fw_iu:.4f}")

        val_iu_list = val_iu_per_class.cpu().numpy() if hasattr(val_iu_per_class, "cpu") else val_iu_per_class
        val_dice_list = val_dice_per_class.cpu().numpy() if hasattr(val_dice_per_class, "cpu") else val_dice_per_class

        print(f"\n  {'Per-Class Dice:':<20}", end="")
        for i in range(len(val_dice_list)):
            label = f"C{i}" if i < len(val_dice_list) - 1 else "BG"
            print(f"{label}: {val_dice_list[i]*100:.2f}%", end="  ")
        print()

        print(f"  {'Per-Class IoU:':<20}", end="")
        for i in range(len(val_iu_list)):
            label = f"C{i}" if i < len(val_iu_list) - 1 else "BG"
            print(f"{label}: {val_iu_list[i]*100:.2f}%", end="  ")
        print()

        current_miou = val_mIoU
        best_miou, saved = save_best(model, optimizer, best_miou, cfg, epoch, current_miou, iters_per_epoch)

        # Compact checkpoint status
        if saved:
            save_path = os.path.join(cfg.work_dir.ckpt_dir, "best_cam.pth")
            print(f"\n✓ Checkpoint SAVED: mIoU={current_miou:.4f} (improved from {best_miou:.4f})")
        else:
            print(f"\n○ Checkpoint not saved: mIoU={current_miou:.4f} (best: {best_miou:.4f})")

    torch.cuda.empty_cache()
    end_time = datetime.datetime.now()
    total_training_time = end_time - time0
    print(f"\nTotal training time: {total_training_time}")

    final_evaluation(cfg, device, num_workers, model, loss_function, train_dataset)


def final_evaluation(cfg, device, num_workers, model, loss_function, train_dataset):
    print("\n" + "=" * 80)
    print("POST-TRAINING EVALUATION AND CAM GENERATION")
    print("=" * 80)

    print("\nPreparing test dataset...")
    _, test_dataset = get_cls_dataset(cfg, split="test", enable_rotation=False, p=0.0)
    print(f"Test dataset loaded: {len(test_dataset)} samples")

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.samples_per_gpu,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    print("\n1. Testing on test dataset...")
    print("-" * 50)

    test_mIoU, test_mean_dice, test_fw_iu, test_iu_per_class, test_dice_per_class = validate_test(
        model=model, data_loader=test_loader, cfg=cfg, cls_loss_func=loss_function
    )

    model.eval()
    test_correct_all = 0
    test_total_samples = 0
    with torch.no_grad():
        for _, inputs, cls_labels, _ in test_loader:
            inputs = inputs.to(device).float()
            cls_labels = cls_labels.to(device).float()
            with autocast():
                outputs = model(inputs)
                cls4 = outputs[6]
                k_list = outputs[9]
                cls4_merge = merge_to_parent_predictions(cls4, k_list, method=cfg.train.merge_test)
                cls_pred4 = (torch.sigmoid(cls4_merge) > 0.5).float()
                all_correct = (cls_pred4 == cls_labels).all(dim=1).float()
                test_correct_all += all_correct.sum().item()
                test_total_samples += all_correct.shape[0]
    test_acc = (test_correct_all / test_total_samples * 100) if test_total_samples > 0 else 0.0
    model.train()

    print("Testing results:")
    print(f"Test mIoU: {test_mIoU:.4f}")
    print(f"Test Mean Dice: {test_mean_dice:.4f}")
    print(f"Test FwIU: {test_fw_iu:.4f}")
    print(f"Test Acc: {test_acc:.2f}%")

    print("\nPer-class IoU scores (FG classes + BG):")
    for i, score in enumerate(test_iu_per_class):
        label = f"Class {i}" if i < len(test_iu_per_class) - 1 else "Background"
        print(f"  {label}: {score*100:.4f}")

    print("\nPer-class Dice scores (FG classes + BG):")
    for i, score in enumerate(test_dice_per_class):
        label = f"Class {i}" if i < len(test_dice_per_class) - 1 else "Background"
        print(f"  {label}: {score*100:.4f}")

    print("\n2. Generating CAMs for complete training dataset...")
    print("-" * 50)

    train_cam_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    print(f"Generating CAMs for all {len(train_dataset)} training samples...")
    print(f"Output directory: {cfg.work_dir.pred_dir}")

    best_model_path = os.path.join(cfg.work_dir.ckpt_dir, "best_cam.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        best_epoch = checkpoint.get("epoch", "unknown")
        print(f"Best model loaded successfully! (Saved at epoch: {best_epoch})")
    else:
        print("Warning: Best model checkpoint not found, using current model state")
        print(f"Expected path: {best_model_path}")

    generate_cam(model=model, data_loader=train_cam_loader, cfg=cfg)

    print("\nFiles generated:")
    print(f"  Training CAM visualizations: {cfg.work_dir.pred_dir}/*.png")
    print(f"  Model checkpoint: {cfg.work_dir.ckpt_dir}/best_cam.pth")
    print("=" * 80)


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    # If the config path has no directory (e.g., 'config.yaml'), os.path.dirname returns ''
    # Use a sensible default directory to store outputs. Prefer a `runs` folder in CWD.
    config_dir = os.path.dirname(args.config)
    if config_dir == "":
        config_dir = os.path.join(os.getcwd(), "runs")
    cfg.work_dir.dir = config_dir
    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    # Create timestamped directories for both checkpoints and predictions
    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir, timestamp)

    os.makedirs(cfg.work_dir.dir, exist_ok=True)
    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)

    print("\nArgs:", args)
    print("\nConfigs:", cfg)

    train(cfg=cfg, args=args)


if __name__ == "__main__":
    main()


