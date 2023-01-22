import torch
from torch import nn
from audio_mae import AudioMaskedAutoencoderViT
from functools import partial


audio_mels = torch.ones([2, 1, 64, 128])

# Paper recommended archs
model  = AudioMaskedAutoencoderViT(
        num_mels=128, mel_len=64, in_chans=1,
        patch_size=16, embed_dim=768, encoder_depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        
loss, pred, mask = model(audio_mels)