# import numpy as np

# array = np.load('/home/di49map/FactFormer/data/smoke_0.npz')
# print(array.files)
# print(array['fluid_field'].shape)
# print(array['velocity'].shape)

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from libs.factorization_module import FABlock3D

dim = 128
dim_head = 64
latent_dim = 128
heads = 4
dim_out = 4
kernel_multiplier = 4
use_rope = True
scaling_factor = 1
scaling_factor = 3

fa_layer = FABlock3D(dim,                   # input dimension
                     dim_head,              # dimension in each attention head, will be expanded by the kernel_multiplier when computing kernel: d = dim_head * kernel_multiplier
                     latent_dim,            # the output dimension of the projection operator
                     heads,                 # attention heads
                     dim_out,               # output dimension
                     kernel_multiplier,     # use more function bases to computer kernel: k(x_i, x_j)=\sum_{c}^dq_c(x_i)k_c(x_j)    
                     use_rope,              # use rotary positional encoding or not, by default True
                     scaling_factor         # use scaling factor to modulate the kernel, an example is 1/ sqrt(d) like scaled-dot product attention, by default is: 1
                    )
# random input
z = torch.randn((1, 66, 66, 66, dim))
# axial coords
pos_x = torch.linspace(0, 1, 66).unsqueeze(-1)       # leave a channel  dimension   
pos_y = torch.linspace(0, 1, 66).unsqueeze(-1)
pos_z = torch.linspace(0, 1, 66).unsqueeze(-1)
lst = [pos_x, pos_y, pos_z]
z = fa_layer(z, [pos_x, pos_y, pos_z]) 
print(z.shape)