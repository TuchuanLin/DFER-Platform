


import torch
from ptflops import get_model_complexity_info
from options import Options

from models.M3D import M3DFEL

args = Options().parse()

def create_model(args):
    """create model according to args

    Args:
        args
    """
    model = M3DFEL(args)

    return model

# 假设 `model` 是你的模型实例
model = create_model(args)

# 输入张量的维度
input_shape =  (16, 3, 112, 112) 

# 计算FLOPs和参数数量
flops, params = get_model_complexity_info(model, input_shape, as_strings=True, print_per_layer_stat=True)

print(f'FLOPs: {flops}')
print(f'Params: {params}')


# import torch
# from torch import nn
# from torchvision.models.video import r3d_18, R3D_18_Weights
# from einops import rearrange
# import torch.nn.functional as F
# from utils import *
# #from .PyxLSTM.xLSTM import *



# # 定义 M3DFEL 类
# class M3DFEL(nn.Module):
#     def __init__(self, args):
#         super(M3DFEL, self).__init__()

#         self.args = args
#         self.device = torch.device('cuda:%d' % args.gpu_ids[0] if args.gpu_ids else 'cpu')
#         self.bag_size = self.args.num_frames // self.args.instance_length
#         self.instance_length = self.args.instance_length
#         self.window_size = 4
#         self.step_size = 2

#         model = r3d_18(weights=R3D_18_Weights.DEFAULT)
#         self.features = nn.Sequential(*list(model.children())[:-1])
#         self.BiIlstm = BiImprovedLSTM(input_dim=512, hidden_dim=512, num_layers=2)
        
#         #nn.LSTM(input_size=512, hidden_size=512,
#         #                    num_layers=2, batch_first=True, bidirectional=True)
        
#         # self.slstm = xLSTM(input_size=512, hidden_size=512, num_layers=2, num_blocks=2, dropout=0.3, lstm_type="slstm")
#         # self.mlstm = xLSTM(input_size=1024, hidden_size=1024, num_layers=1, num_blocks=2, dropout=0.3, lstm_type="mlstm")

#         self.heads = 8
#         self.dim_head = 1024 // self.heads
#         self.scale = self.dim_head ** -0.5
#         self.attend = nn.Softmax(dim=-1)
#         self.to_qkv = nn.Linear(1024, (self.dim_head * self.heads) * 3, bias=False)

#         self.norm = DMIN(num_features=1024)
#         self.pwconv = nn.Conv1d(7, 1, 3, 1, 1)

#         self.fc = nn.Linear(1024, self.args.num_classes)
#         self.Softmax = nn.Softmax(dim=-1)

#     def MIL(self, x):
#         """The Multi Instance Learning Aggregation of instances

#         Inputs:
#             x: [batch, 7, 512]
#         """
#         # self.lstm.flatten_parameters()
#         # x_ = x
#         #x, _ = self.slstm(x)
        
        
#         # First xLSTM layer (mLSTM)
#         x= self.BiIlstm(x)
#         # [batch, bag_size, 512]
#         #x2, _ = self.slstm(x_)
#         # Second xLSTM layer (sLSTM)
#         #x, _ = self.xlstm2(x)
#         # [batch, bag_size, 512]
#        # x_ = torch.cat((x1, x2), dim=-1) 
#         # ori_x = x

#         # # Multi-Head Self Attention
#         # qkv = self.to_qkv(x).chunk(3, dim=-1)
#         # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
#         # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#         # attn = self.attend(dots)
#         # x = torch.matmul(attn, v)
#         # x = rearrange(x, 'b h n d -> b n (h d)')
#         # x = self.norm(x)
#         # x = torch.sigmoid(x)

#         return x


#     def forward(self, x):
#         """Forward pass through the model
        
#         Inputs:
#             x: [batch, 16, 3, 112, 112]
#         """
#        # Initial input shape: [batch, 16, 3, 112, 112]
#         x = rearrange(x, 'b t c h w -> b c t h w')

#         # After rearrange: [batch, 3, 16, 112, 112]

#         x = create_sliding_windows(x, window_size=self.window_size, step_size=self.step_size)
#         # After create_sliding_windows: [batch, num_windows, 3, window_size, 112, 112]
#         # Example shape: [batch, 7, 3, 4, 112, 112]

#         x = rearrange(x, 'b w c t h d -> (b w) c t h d')
#         # After rearrange: [(batch * num_windows), 3, window_size, 112, 112]
#         # Example shape: [batch * 7, 3, 4, 112, 112]
        
#         x = self.features(x).squeeze()
#         # After features extraction and squeeze: [(batch * num_windows), 512]
#         # Example shape: [batch * 7, 512]
        
#         x = rearrange(x, '(b w) c -> b w c', w=7)
#         # After rearrange: [batch, num_windows, 512]
#         # Example shape: [batch, 7, 512]
        
#         x = self.MIL(x)
#         # After MIL: [batch, num_windows, 1024]
#         # Example shape: [batch, 7, 1024]

#         x = self.pwconv(x).squeeze()
#         # After pwconv and squeeze: [batch, 1024]
#         ##features before the FC layer are used to form t-SNE
#         ##temporal avg pooling
#         x_FER =x
#         out = self.fc(x)
#         # After fc: [batch, num_classes]

#         return out, x_FER




# # 定义 create_sliding_windows 方法
# def create_sliding_windows(x, window_size, step_size):
#     b, c, t, h, w = x.shape
#     windows = []
#     for i in range(0, t - window_size + 1, step_size):
#         windows.append(x[:, :, i:i + window_size, :, :])
#     return torch.stack(windows, dim=1)





# class ImprovedLSTMCell(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(ImprovedLSTMCell, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim

#         self.W_i = nn.Linear(input_dim, hidden_dim)
#         self.U_i = nn.Linear(hidden_dim, hidden_dim)
#         self.W_f = nn.Linear(input_dim, hidden_dim)
#         self.U_f = nn.Linear(hidden_dim, hidden_dim)
#         self.W_o = nn.Linear(input_dim, hidden_dim)
#         self.U_o = nn.Linear(hidden_dim, hidden_dim)
#         self.W_c = nn.Linear(input_dim, hidden_dim)
#         self.U_c = nn.Linear(hidden_dim, hidden_dim)
        
#         # 使用多层感知器的弱信号门
#         self.W_s = nn.Linear(input_dim, hidden_dim)
#         self.U_s = nn.Linear(hidden_dim, hidden_dim)
#         self.MLP = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )

#     def forward(self, x, hidden_state):
#         h_prev, c_prev = hidden_state

#         i_t = torch.sigmoid(self.W_i(x) + self.U_i(h_prev))
#         f_t = torch.sigmoid(self.W_f(x) + self.U_f(h_prev))
#         o_t = torch.sigmoid(self.W_o(x) + self.U_o(h_prev))
#         c_hat_t = torch.tanh(self.W_c(x) + self.U_c(h_prev))
        
#         # 弱信号门
#         s_t = torch.sigmoid(self.W_s(x) + self.U_s(h_prev))
#         s_t = self.MLP(s_t)
        
#         c_t = f_t * c_prev + i_t * c_hat_t * s_t
#         h_t = o_t * torch.tanh(c_t)

#         return h_t, c_t

# class BiImprovedLSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers=1):
#         super(BiImprovedLSTM, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers

#         self.fwd_layers = nn.ModuleList([ImprovedLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
#         self.bwd_layers = nn.ModuleList([ImprovedLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])

#     def forward(self, x):
#         batch_size, seq_len, _ = x.size()

#         h_fwd = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
#         c_fwd = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        
#         h_bwd = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
#         c_bwd = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]

#         output_fwd = []
#         for t in range(seq_len):
#             x_t = x[:, t, :]
#             for layer in range(self.num_layers):
#                 h_fwd[layer], c_fwd[layer] = self.fwd_layers[layer](x_t, (h_fwd[layer], c_fwd[layer]))
#                 x_t = h_fwd[layer]
#             output_fwd.append(h_fwd[-1])

#         output_bwd = []
#         for t in range(seq_len-1, -1, -1):
#             x_t = x[:, t, :]
#             for layer in range(self.num_layers):
#                 h_bwd[layer], c_bwd[layer] = self.bwd_layers[layer](x_t, (h_bwd[layer], c_bwd[layer]))
#                 x_t = h_bwd[layer]
#             output_bwd.append(h_bwd[-1])

#         output_bwd.reverse()
#         output = torch.cat([torch.stack(output_fwd, dim=1), torch.stack(output_bwd, dim=1)], dim=-1)

#         return output


