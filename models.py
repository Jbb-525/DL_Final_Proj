from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
import math
from normalizer import StateNormalizer, ActionNormalizer


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)

class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output

####################################################################################
#                                      ENCODER                                     #
####################################################################################

class CNNSelfAttention(nn.Module):
    
    def __init__(self, n_channels, n_heads):
        super(SelfAttention, self).__init__()
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.q_conv = nn.Conv2d(in_channels=n_channels, out_channels = n_channels // n_heads, kernel_size=1)
        self.k_conv = nn.Conv2d(in_channels=n_channels, out_channels = n_channels // n_heads, kernel_size=1)
        self.v_conv = nn.Conv2d(in_channels=n_channels, out_channels = n_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
      
    def forward(self, x):
        
        m_batchsize, C, width , height = x.size()
        proj_query  = self.q_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.k_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check

        # Scale Attn Weights by Channel Depth (embedding dimension)
        attention = self.softmax(energy / torch.sqrt(torch.tensor(C))) # BX (N) X (N) 
        proj_value = self.v_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out

class ResidualLayer(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.ConvBlock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        # 1 x 1 Convolution for Residual whenever downsmapling (stride > 1)
        self.downsample_residual = None
        if stride != 1 or in_channels != out_channels:
            self.downsample_residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        if self.downsample_residual is not None:
            residual = self.downsample_residual(x)
        x = self.ConvBlock(x)
        return (F.relu(x + residual))


class CNNBackbone(nn.Module):

    def __init__(self, n_kernels, repr_dim, dropout=0.1):
        super().__init__()
        self.n_kernels = n_kernels
        self.repr_dim = repr_dim
        self.dropout = dropout
        # 2 x 64 x 64 --> n_kernels x 64 x 64
        self.ConvLayer1 = nn.Sequential(
            nn.Conv2d(2, self.n_kernels, kernel_size=3), 
            nn.BatchNorm2d(self.n_kernels),
            nn.ReLU(inplace=True)
        )        
        
        # n_kernels x 64 x 64 --> n_kernels * 4 x 16 x 16
        self.ResBlock1 = nn.Sequential(
            ResidualLayer(self.n_kernels, self.n_kernels*2, stride=2),
            ResidualLayer(self.n_kernels*2, self.n_kernels*4, stride=2),
        )
        self.SelfAttn1 = CNNSelfAttention(
            n_channels=self.n_kernels*4,
            n_heads=2 # 1 Head per 32 channels
        )
        self.Bn1 = nn.BatchNorm2d(self.n_kernels*4)
        
        # n_kernels * 2 x 16 x 16 --> n_kernels * 16 x 4 x 4
        self.ResBlock2 = nn.Sequential(
            ResidualLayer(self.n_kernels*4, self.n_kernels*8, stride=2),
            ResidualLayer(self.n_kernels*8, self.n_kernels*16, stride=2),
        )
        self.SelfAttn2 = CNNSelfAttention(
            n_channels=self.n_kernels*16,
            n_heads=2 # 1 Head per 32 channels
        )
        self.Bn2 = nn.BatchNorm2d(self.n_kernels*16)
    
        self.FC1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.n_kernels*16*4*4, out_features=self.repr_dim*2, bias=True),
            nn.Dropout(p=self.dropout),
            nn.ReLU(),
            nn.Linear(in_features=self.repr_dim*2, out_features=self.repr_dim, bias=True)
        )
    
    def forward(self, x):
        x = self.ConvLayer1(x) # 64x64 -> 32x32
        x = self.Bn1(self.SelfAttn1(self.ResBlock1(x))) # 32x32 -> 16x16
        x = self.Bn2(self.SelfAttn2(self.ResBlock2(x))) # 16x16 -> 8x8
        x = self.FC1(x)
        return x # (batch_size, n_kernels * 16)

class PatchEmbedding(nn.Module):
    
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
      super().__init__()
      self.image_size = image_size
      self.patch_size = patch_size
      self.in_channels = in_channels
      self.embed_dim = embed_dim
      
      self.conv_proj = nn.Conv2d(
        in_channels=self.in_channels, 
        out_channels=embed_dim, 
        kernel_size=self.patch_size, 
        stride=self.patch_size
      )

    def forward(self, x):
      bs, c, h, w = x.shape
      n_patches = (h * w) // self.patch_size**2

      x = self.conv_proj(x) # (bs, c, h, w) --> (bs, embed_size, n_h, n_w)
      x = x.reshape(bs, self.embed_dim, n_patches) # (bs, embed_size, n_h, n_w) --> (bs, embed_size, n_patches)
      x = x.permute(0, 2, 1) # (bs, embed_size, n_patches) --> (bs, n_patches, embed_size)
      return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
      super().__init__()
      
      self.embed_dim = embed_dim
      self.num_heads = num_heads
      self.head_dim = embed_dim // num_heads
      self.dropout = dropout

      # Create copies of input for each q, k, v weights
      self.qkv = nn.Linear(embed_dim, embed_dim * 3) 
      self.q_norm = nn.LayerNorm(self.head_dim)
      self.k_norm = nn.LayerNorm(self.head_dim)

      self.projection = nn.Linear(self.embed_dim, self.embed_dim)
      self.projection_dropout = nn.Dropout(self.dropout)


    def forward(self, x):
      bs, n_patches, embed_size = x.shape

      # Copies input embedding 3 times for q, k and v --> (Concat, bs, num_heads, n_patches, head_dim)
      qkv = self.qkv(x).reshape(bs, n_patches, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
      # 3 x copies (bs, num_heads, n_patches, head_dim)
      q, k, v = qkv.unbind(0)

      q, k = self.q_norm(q), self.k_norm(k)

      # Scaled Dot Product Attn (QK^T) / sqrt(d)
      attn = q @ k.transpose(-2, -1) * math.sqrt(self.head_dim)**-1
      attn = F.softmax(attn, dim=-1)

      x = attn @ v
      x = x.transpose(1, 2).reshape(bs, n_patches, embed_size)
      x = self.projection(x)
      x = self.projection_dropout(x)
      return x
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.layer_norm_1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm_2 = nn.LayerNorm(self.embed_dim)
        
        self.attention = MultiHeadSelfAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=self.dropout)

        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, mlp_dim),
            nn.Dropout(self.dropout),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        residual_1 = x
        x = self.attention(x)
        x = self.layer_norm_1(x) + residual_1

        residual_2 = x
        x = self.ffn(x)
        x = self.layer_norm_2(x) + residual_2

        return x  
    
class ViTBackbone(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, num_heads, mlp_dim, num_layers, dropout=0.1):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.patch_embedding = PatchEmbedding(self.image_size, self.patch_size, self.in_channels, self.embed_dim)
        n_patches = (self.image_size // self.patch_size)**2

        # Learnable Class Token
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) 
      
        # Learnable Position Embeddings 
        self.position_encoding = nn.Parameter(torch.empty(1, n_patches + 1, self.embed_dim))
        nn.init.trunc_normal_(self.position_encoding, std=0.02)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.embed_dim, self.num_heads, self.mlp_dim, self.dropout)
            for _ in range(self.num_layers)
        ])

    def forward(self, x):
        bs = x.shape[0]
        x = self.patch_embedding(x)

        # Expan class tokens along batch dimension
        class_token = self.class_token.expand(bs, -1, -1)
        
        # Concatenat class token to each embedding
        x = torch.cat((class_token, x), dim=1)

        # Add positional encoding
        x += self.position_encoding

        for block in self.transformer_blocks:
            x = block(x)

        # Return class tokens as dense representation
        embedding = x[:, 0, :]
        return embedding


class BarlowTwins(nn.Module):

    def __init__(self, backbone, repr_dim, batch_size=None, projection_layers=3, lambd=5E-3):
        super().__init__()

        self.batch_size = batch_size
        self.repr_dim = repr_dim
        self.projection_layers = projection_layers
        self.lambd = lambd
        self.backbone = backbone

        layer_sizes = [self.repr_dim] + [(self.repr_dim * 4) for _ in range(self.projection_layers)]
        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=False))
            layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
       
        self.batch_norm = nn.BatchNorm1d(layer_sizes[-1], affine=False)
    
    def _off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m, 'Square Matrix Expected, Rectangular Instead'
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, Y_a, Y_b=None):

        if not self.training:
            return self.backbone(Y_a)
        
        Z_a = self.projector(self.backbone(Y_a))
        Z_b = self.projector(self.backbone(Y_b))

        # Cross-Correlation Matrix
        cc_mat = self.batch_norm(Z_a).T @ self.batch_norm(Z_b)
        cc_mat.div_(self.batch_size)

        diag = torch.diagonal(cc_mat).add_(-1).pow_(2).sum()
        off_diag = self._off_diagonal(cc_mat).pow_(2).sum()

        return diag, self.lambd * off_diag

####################################################################################
#                                    PREDICTOR                                     #
####################################################################################

class Predictor(nn.Module):
  def __init__(self, hidden_dim):
    super().__init__()
    self.action_encoder = nn.Sequential(
        nn.Linear(2,hidden_dim//2),
        nn.ReLU(),
        nn.Linear(hidden_dim//2,hidden_dim)
    )
        # Cross Attention between agent and wall features
    self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

    self.norm1 = nn.LayerNorm(hidden_dim)
    self.norm2 = nn.LayerNorm(hidden_dim)
      
    self.GRU = nn.GRU(hidden_dim, hidden_dim*2, 1, batch_first = False)
    self.mlp = nn.Linear(hidden_dim*2, hidden_dim)

    self.dropout = nn.Dropout(0.1)

  def forward(self,state,action):
      # state [B, T-1, hidden_dim]
    state_=[]
    for t in range(state.size(1)):
        
        action_embed = self.action_encoder(action[:, t])
        state_t = state[:, t].unsqueeze(1) # [B, 1, H]
        action_embed = action_embed.unsqueeze(1) # [B, 1, H]
        
        x = state_t
        q = self.norm1(x)
        k = v = self.norm2(action_embed)
        attended, _ = self.cross_attention(q, k, v)
        x = x + attended  # residual
        state_.append(x.squeeze(1))
    
    state_ = torch.stack(state_,dim=1) # [B, T-1, hidden_dim]
    out,_ = self.GRU(state_) # [B, T-1,H*2]
    
    states = self.dropout(self.mlp(out)) # [B, T-1,H]
    
    return states, states[:,-1]

def init_identity_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.eye_(m.weight)
        nn.init.zeros_(m.bias)

class JEPA(nn.Module):
  def __init__(self, repr_dim):
    super().__init__()
    enc_path = r"best_encoder.pth"
    state_dict = torch.load(enc_path, weights_only=False)
    backbone_state_dict = {k.replace("backbone.", ""): v for k, v in state_dict["model_state_dict"].items() if "backbone." in k}
    # Define the ViT Backbone
    backbone = ViTBackbone(
        image_size=65,
        patch_size=5,
        in_channels=2,
        embed_dim=256,
        num_heads=4,
        mlp_dim=1024,
        num_layers=2,
        dropout=0.1,
    )
    backbone.load_state_dict(backbone_state_dict, strict=True)
    self.encoder = backbone
    self.encoder.eval()
    self.repr_dim = repr_dim
    self.predictor = Predictor(repr_dim)

    self.encoder_mlp = nn.Linear(repr_dim,repr_dim)
    # self.encoder_mlp = nn.Identity()
    
    self.encoder_mlp.apply(init_identity_weights)

    self.warmup_epochs = 30
    self.decay_rate = 0.2
      
  def forward(self, states, actions, current_epoch=None, total_epochs=None):
      
    statenormalizer = StateNormalizer()
    obs_sequence = statenormalizer.normalize_state(states)

    actionnormalizer = ActionNormalizer()
    actions_sequence = actionnormalizer.normalize_actions(actions)
    
    if not self.training:
        with torch.no_grad():
            B, T = actions_sequence.size(0), actions_sequence.size(1)

            state_0 = self.encoder_mlp(self.encoder(obs_sequence[:, 0])).unsqueeze(1)
            state_t = state_0.clone()
            states=[]
                
            for t in range(T):
                _,last_hidden = self.predictor(state_t, actions_sequence[:,t:t+1])
                states.append(last_hidden.reshape(B, -1)) #[B,hidden_dim*4]
                state_t = last_hidden.reshape(B, 1,-1)
            states =  torch.stack(states,dim=1)
            pred_states, _ = self.predictor(states, actions_sequence)
            
            pred_states = torch.cat((state_0,pred_states),dim=1)
            return pred_states
            
    if self.training:      
        # Batch size and sequence length
        B, T = obs_sequence.size(0), obs_sequence.size(1)
        # Generate true states
        true_states  = []
        init_encoder = []
        for t in range(T):
            true_state  = self.encoder(obs_sequence[:, t])
            init_encoder.append(true_state)
            true_states.append(self.encoder_mlp(true_state))
        true_states = torch.stack(true_states, dim=1)  # [B, T, hidden_dim] 
        init_encoder = torch.stack(init_encoder, dim=1)  # [B, T, hidden_dim] 

        if current_epoch < self.warmup_epochs:
            # Use full Teacher Forcing during warm-up
            states, _ = self.predictor(init_encoder[:, 0:T-1], actions_sequence)
        else:
            teacher_forcing_ratio = max(0.0, math.exp(-self.decay_rate * (current_epoch - self.warmup_epochs)))
            # Initialize state_t with the first time step's encoded state
            state_0 = true_states[:, 0].unsqueeze(1)  # [B, 1, hidden_dim]
            state_t = state_0.clone()
            states = []

            # Iterate over time steps
            for t in range(T - 1):
                    # Predict next state
                _, last_hidden = self.predictor(state_t, actions_sequence[:, t:t + 1])  # [B, 1, hidden_dim]
                states.append(last_hidden.reshape(B, -1))  # Append predicted state

                # Decide next input based on teacher forcing ratio
                if random.random() < teacher_forcing_ratio:
                    state_t = true_states[:, t + 1:t + 2, :]  # Use ground truth state
                else:
                    state_t = last_hidden.reshape(B, 1, -1)  # Use predicted state

                # Stack predicted states
            states = torch.stack(states, dim=1)  # [B, T-1, hidden_dim]

        return states, true_states, init_encoder
    
  def compute_loss(self,predictions, true_states, init_encoder,tuning_encoder=False):
    if tuning_encoder==False:
        return F.mse_loss(predictions, init_encoder[:, 1:])
    else:
        total_loss = F.mse_loss(predictions, true_states[:, 1:]) + F.mse_loss(init_encoder,true_states)
        return total_loss, F.mse_loss(predictions, true_states[:, 1:]),F.mse_loss(init_encoder,true_states)





 
        

