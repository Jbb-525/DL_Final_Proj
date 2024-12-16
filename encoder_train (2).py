import argparse
from tqdm import tqdm 

import torch
from torchvision.transforms import v2

from dataset import create_wall_dataloader
from models import BarlowTwins, ViTBackbone
from normalizer import StateNormalizer


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Barlow Twins Encoder')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--warmup_epochs', type=float, default=3, help='Number of warmup epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--repr_dim', type=int, default=256, help='Dimensionality of the representation')
    parser.add_argument('--vit_blocks', type=int, default=2, help='Number of transformer blocks in backbone')
    parser.add_argument('--dropout', type=int, default=0.1, help='ViT Dropout')
    parser.add_argument('--base_lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--proj_lyrs', type=int, default=3, help='Number of Projection Layers for Decoder')
    parser.add_argument('--lambd', type=float, default=5e-3, help='Lambda parameter for loss')

    return parser.parse_args()

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

def load_train_data(device, batch_size):
    path = '/scratch/DL24FA/train'

    train_ds = create_wall_dataloader(
        data_path=path,
        device=device,
        probing=False,
        train=True,
        batch_size=batch_size
    )
    return train_ds


def augment_data(imgs):
    
    transforms = v2.Compose([
        v2.RandomRotation(10),
        v2.RandomVerticalFlip(0.5),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomCrop(60),
        v2.RandomApply([
            v2.GaussianBlur(kernel_size=3,sigma=(0.1, 1))
        ], p=0.5),
        v2.Resize(65)
    ])
    return torch.stack([transforms(img) for img in imgs])

def train(model, data, device, epochs, warmup_epochs, base_lr, checkpoint_path=None):
    """
    Encoder Pre-Training Loop
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1, total_iters=warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-warmup_epochs)
    
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [warmup_scheduler, cosine_scheduler],
        [warmup_epochs]
    )
    
    start_epoch = 0 
    best_loss = float('inf')
    losses_off_diag = []
    normalizer = StateNormalizer()

    if checkpoint_path:
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch'] + 1  # Resume from the next 
        
        # Advance scheduler to correct epoch
        for _ in range(start_epoch):
            lr_scheduler.step()
        
        if 'normalizer_state' in checkpoint and checkpoint['normalizer_state'] is not None:
            normalizer.load_state_dict(checkpoint['normalizer_state'])

    for epoch in tqdm(range(epochs), initial=start_epoch, total=epochs):
        epoch_diag_loss = 0
        epoch_off_diag_loss = 0
        num_batches = 0

        for batch in tqdm(data, desc=f'Epoch {epoch}'):
            
            states = batch.states.to(device)
            states = normalizer.normalize_state(states)
            
            # Un-augmented Frames
            Y_a = states
            batch_size, num_frames, channels, height, width = Y_a.shape
            Y_a = Y_a.view(batch_size * num_frames, channels, height, width)
            
            # Augmented Frame for Loss
            Y_b = augment_data(Y_a).to(device)

            # Forward pass
            loss_diag, loss_off_diag = model(Y_a, Y_b)
            loss = loss_diag + loss_off_diag
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_diag_loss += loss_diag.item()
            epoch_off_diag_loss += loss_off_diag.item()
            num_batches += 1
        
        avg_diag_loss = epoch_diag_loss / num_batches
        avg_off_diag_loss = epoch_off_diag_loss / num_batches
        
        losses_off_diag.append(avg_off_diag_loss)
        
        print(f'\nEpoch {epoch} Avg Diagonal Loss: {avg_diag_loss:.3f}, Avg Off Diagonal Loss: {avg_off_diag_loss:.3f}\n')
        lr_scheduler.step()

        if avg_off_diag_loss < best_loss:
            best_loss = avg_off_diag_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'normalizer_state': normalizer.state_dict() if hasattr(normalizer, 'state_dict') else None
            }, '/home/ad3254/checkpoints/best.pth')

    return model


def main():
    args = parse_args()
    device = get_device()
    data = load_train_data(device, batch_size=args.batch_size)
    
    vit_backbone = ViTBackbone(
            image_size=65,
            patch_size=5,
            in_channels=2,
            embed_dim=args.repr_dim,
            num_heads=args.repr_dim // 64,
            mlp_dim=args.repr_dim*4,
            num_layers=args.vit_blocks,
            dropout=args.dropout,
        )
    
    enc = BarlowTwins(vit_backbone, args.repr_dim, args.batch_size * 17, args.proj_lyrs, args.lambd)
    encoder = train(enc, data, device, args.epochs, args.warmup_epochs, args.base_lr)
    torch.save(encoder.state_dict(), '/home/ad3254/encoder.pth')
    

if __name__ == "__main__":
    main()