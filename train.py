import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from tqdm import tqdm
import numpy as np

from model import ISNetTransformerBBox
from dataset import ObjectRemovalDataset, COCOObjectRemovalDataset, collate_fn


class Trainer:
    """
    Trainer class cho ISNet-Transformer
    """

    def __init__(
            self,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            device,
            save_dir='./checkpoints',
            log_dir='./logs'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.log_dir = log_dir

        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Tensorboard writer
        self.writer = SummaryWriter(log_dir)

        # Best model tracking
        self.best_val_loss = float('inf')
        self.global_step = 0

    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()

        epoch_loss = 0
        epoch_loss0 = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            bboxes = batch['bboxes'].to(self.device)
            classes = batch['classes'].to(self.device)
            style_idx = batch['style_idx'].to(self.device)

            # Forward pass
            outputs = self.model(images, bboxes, classes, style_idx)

            # Compute loss
            loss0, loss = self.model.compute_loss(outputs, masks)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            epoch_loss0 += loss0.item()

            # Log to tensorboard
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/Loss0', loss0.item(), self.global_step)

                # Log learning rate
                lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Train/LearningRate', lr, self.global_step)

            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'loss0': f'{loss0.item():.4f}'
            })

        # Average loss
        avg_loss = epoch_loss / len(self.train_loader)
        avg_loss0 = epoch_loss0 / len(self.train_loader)

        return avg_loss, avg_loss0

    @torch.no_grad()
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()

        val_loss = 0
        val_loss0 = 0

        pbar = tqdm(self.val_loader, desc='Validation')

        for batch in pbar:
            # Move to device
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            bboxes = batch['bboxes'].to(self.device)
            classes = batch['classes'].to(self.device)
            style_idx = batch['style_idx'].to(self.device)

            # Forward pass
            outputs = self.model(images, bboxes, classes, style_idx)

            # Compute loss
            loss0, loss = self.model.compute_loss(outputs, masks)

            val_loss += loss.item()
            val_loss0 += loss0.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'loss0': f'{loss0.item():.4f}'
            })

        # Average loss
        avg_val_loss = val_loss / len(self.val_loader)
        avg_val_loss0 = val_loss0 / len(self.val_loader)

        # Log to tensorboard
        self.writer.add_scalar('Val/Loss', avg_val_loss, epoch)
        self.writer.add_scalar('Val/Loss0', avg_val_loss0, epoch)

        # Log sample predictions
        if epoch % 5 == 0:
            self._log_predictions(images, masks, outputs, epoch)

        return avg_val_loss, avg_val_loss0

    def _log_predictions(self, images, masks, outputs, epoch):
        """Log sample predictions to tensorboard"""
        # Take first image in batch
        img = images[0].cpu()
        gt_mask = masks[0].cpu()
        pred_mask = outputs[0][0].cpu()

        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)

        # Log to tensorboard
        self.writer.add_image('Val/Image', img, epoch)
        self.writer.add_image('Val/GroundTruth', gt_mask, epoch)
        self.writer.add_image('Val/Prediction', pred_mask, epoch)

    def save_checkpoint(self, epoch, avg_val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': avg_val_loss,
            'global_step': self.global_step
        }

        # Save latest checkpoint
        checkpoint_path = os.path.join(self.save_dir, 'latest.pth')
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f'✅ Saved best model with val_loss: {avg_val_loss:.4f}')

        # Save periodic checkpoint
        if epoch % 10 == 0:
            epoch_path = os.path.join(self.save_dir, f'epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)

    def train(self, num_epochs):
        """Main training loop"""
        print(f"🚀 Starting training for {num_epochs} epochs...")
        print(f"   Device: {self.device}")
        print(f"   Train samples: {len(self.train_loader.dataset)}")
        print(f"   Val samples: {len(self.val_loader.dataset)}")

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'=' * 60}")

            # Train
            train_loss, train_loss0 = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f}, Loss0: {train_loss0:.4f}")

            # Validate
            val_loss, val_loss0 = self.validate(epoch)
            print(f"Val Loss: {val_loss:.4f}, Loss0: {val_loss0:.4f}")

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.save_checkpoint(epoch, val_loss, is_best=is_best)

        print(f"\n✅ Training completed!")
        print(f"   Best val loss: {self.best_val_loss:.4f}")
        self.writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train ISNet-Transformer for object removal')

    # Dataset
    parser.add_argument('--train_img_dir', type=str, required=True, help='Training images directory')
    parser.add_argument('--train_mask_dir', type=str, required=True, help='Training masks directory')
    parser.add_argument('--train_ann_file', type=str, required=True, help='Training annotation file')
    parser.add_argument('--val_img_dir', type=str, required=True, help='Validation images directory')
    parser.add_argument('--val_mask_dir', type=str, required=True, help='Validation masks directory')
    parser.add_argument('--val_ann_file', type=str, required=True, help='Validation annotation file')
    parser.add_argument('--dataset_type', type=str, default='custom', choices=['custom', 'coco'],
                        help='Dataset type: custom or coco format')

    # Model
    parser.add_argument('--num_classes', type=int, default=80, help='Number of object classes')
    parser.add_argument('--num_styles', type=int, default=3, help='Number of image styles')
    parser.add_argument('--img_size', type=int, default=512, help='Image size')
    parser.add_argument('--max_objects', type=int, default=20, help='Maximum number of objects per image')

    # Training
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')

    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Checkpoint save directory')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Tensorboard log directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    return parser.parse_args()


def main():
    args = get_args()

    print("=" * 60)
    print("ISNet-Transformer Training")
    print("=" * 60)
    print(f"Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("=" * 60)

    # Set device
    device = torch.device(args.device)

    # Create datasets
    print("\n📂 Loading datasets...")

    if args.dataset_type == 'custom':
        train_dataset = ObjectRemovalDataset(
            img_dir=args.train_img_dir,
            mask_dir=args.train_mask_dir,
            annotation_file=args.train_ann_file,
            img_size=args.img_size,
            max_objects=args.max_objects,
            augment=True
        )

        val_dataset = ObjectRemovalDataset(
            img_dir=args.val_img_dir,
            mask_dir=args.val_mask_dir,
            annotation_file=args.val_ann_file,
            img_size=args.img_size,
            max_objects=args.max_objects,
            augment=False
        )
    else:  # coco
        train_dataset = COCOObjectRemovalDataset(
            img_dir=args.train_img_dir,
            annotation_file=args.train_ann_file,
            mask_dir=args.train_mask_dir,
            img_size=args.img_size,
            max_objects=args.max_objects,
            augment=True
        )

        val_dataset = COCOObjectRemovalDataset(
            img_dir=args.val_img_dir,
            annotation_file=args.val_ann_file,
            mask_dir=args.val_mask_dir,
            img_size=args.img_size,
            max_objects=args.max_objects,
            augment=False
        )

    print(f"✅ Train dataset: {len(train_dataset)} samples")
    print(f"✅ Val dataset: {len(val_dataset)} samples")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Create model
    print("\n🏗️  Creating model...")
    model = ISNetTransformerBBox(
        in_ch=3,
        out_ch=1,
        num_classes=args.num_classes,
        num_styles=args.num_styles
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created with {num_params:,} parameters")

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\n📥 Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"✅ Resumed from epoch {start_epoch}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )

    # Start training
    trainer.train(num_epochs=args.num_epochs)


if __name__ == '__main__':
    main()