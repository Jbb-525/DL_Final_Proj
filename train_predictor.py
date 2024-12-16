import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import create_wall_dataloader
from tqdm import tqdm
import matplotlib.pyplot as plt
from normalizer import StateNormalizer, ActionNormalizer
from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import ViTBackbone, BarlowTwins, JEPA
import glob
import random
    
def compute_loss(predictions, true_states, init_encoder,tuning_encoder=False):
    if tuning_encoder==False:
        return F.mse_loss(predictions, init_encoder[:, 1:])
    else:
        total_loss = F.mse_loss(predictions, true_states[:, 1:]) + F.mse_loss(init_encoder,true_states)
        return total_loss, F.mse_loss(predictions, true_states[:, 1:]),F.mse_loss(init_encoder,true_states)

def train_step_teacher(model,batch,optimizer,epoch,num_epochs):
    obs_sequence = batch.states
    actions_sequence = batch.actions  # [B, T-1, 2]
    
    predictions, targets,init_encoder = model(obs_sequence, actions_sequence,epoch, num_epochs)
    
    loss = compute_loss(predictions, targets,init_encoder,tuning_encoder=False)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def train_step_schedule(model,batch,optimizer,epoch,num_epochs):
    obs_sequence = batch.states
    actions_sequence = batch.actions  # [B, T-1, 2]
    
    predictions, targets,init_encoder = model(obs_sequence, actions_sequence,epoch, num_epochs)
    
    loss,pred,reg = compute_loss(predictions, targets,init_encoder,tuning_encoder=True)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss,pred,reg

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_jepa = JEPA(repr_dim=256).to(device)

    num_epochs = 80
    warmup_epochs = 30
    schedule_epochs = 40
    batch_size = 256

    train_loader = create_wall_dataloader(
        data_path='/scratch/DL24FA/train',
        batch_size=batch_size,
        device = device
    )

    for param in model_jepa.encoder.parameters():
        param.requires_grad = False 

    for param in model_jepa.encoder_mlp.parameters():
        param.requires_grad = False 
        
    optimizer_teacher = optim.AdamW(
        model_jepa.predictor.parameters(),
        lr=5e-4
    )
    scheduler_teacher = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_teacher,
        T_max=warmup_epochs,
        eta_min=1e-5)

    optimizer_schedule = optim.Adam(model_jepa.predictor.parameters(), lr = 1e-5)

    optimizer_tuning = optim.AdamW([
        {'params': model_jepa.predictor.parameters(), 'lr': 1e-5},
        {'params': model_jepa.encoder_mlp.parameters()} 
    ],lr=1e-5)


    best_loss = 1e+4
    best_model= 100


    for epoch in range(0,num_epochs):
        pred_loss = 0
        mse_loss = 0
        dis_loss = 0
        
        print("="*100)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', position=0, leave=True)
        
        if epoch < warmup_epochs:
            optimizer = optimizer_teacher
            for batch_idx, batch in enumerate(pbar):
                    states = batch.states
                    losses = train_step_teacher(model_jepa, batch, optimizer,epoch,num_epochs)
                    pred_loss += losses
                    if batch_idx % 10 == 0:
                        pbar.set_postfix({'pred_loss': f"{pred_loss/(batch_idx+1):.4f}"})
        
            avg_pred_loss = pred_loss / len(train_loader)
            print(f"\nEpoch {epoch+1} Average Losses:{avg_pred_loss:.4f}")
                    
        elif epoch < warmup_epochs + schedule_epochs:
            optimizer = optimizer_schedule
            for batch_idx, batch in enumerate(pbar):
                    states = batch.states
                    losses = train_step_teacher(model_jepa, batch, optimizer,epoch,num_epochs)
                    pred_loss += losses
                    if batch_idx % 10 == 0:
                        pbar.set_postfix({'pred_loss': f"{pred_loss/(batch_idx+1):.4f}"})
        
            avg_pred_loss = pred_loss / len(train_loader)
            print(f"\nEpoch {epoch+1} Average Losses:{avg_pred_loss:.4f}")
                    
        else:
            for param in model_jepa.encoder_mlp.parameters():
                param.requires_grad = True 
            optimizer = optimizer_tuning
            for batch_idx, batch in enumerate(pbar):
                    states = batch.states  
                    
                    losses,mse,dis = train_step_schedule(model_jepa, batch, optimizer,epoch,num_epochs)
                    pred_loss += losses
                    mse_loss += mse
                    dis_loss += dis
        
                    if batch_idx % 10 == 0:
                        pbar.set_postfix({'pred_loss': f"{pred_loss/(batch_idx+1):.4f}",
                                        'mse_loss': f"{mse_loss/(batch_idx+1):.4f}",
                                        'dis_loss': f"{dis_loss/(batch_idx+1):.4f}"})
        
            avg_pred_loss = pred_loss / len(train_loader)
            print(f"\nEpoch {epoch+1} Average Losses:{avg_pred_loss:.4f}")
            
            if avg_pred_loss < best_loss:
                torch.save(model_jepa.state_dict(), "JEPA.pth")
                best_loss = avg_pred_loss 

if __name__ == "__main__":    
    main()