import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Beta
import math
from math import sqrt
from torchvision.models.video import r3d_18, R3D_18_Weights
from einops import rearrange

from ..utils.utils import *
#from models.dpmil import *

class M3DFEL(nn.Module):
    """The proposed M3DFEL framework

    Args:
        args
    """

    def __init__(self, args):
        super(M3DFEL, self).__init__()

        self.args = args
        self.device = torch.device(
            'cuda:%d' % args.gpu_ids[0] if args.gpu_ids else 'cpu')
        self.bag_size = self.args.num_frames // self.args.instance_length
        self.instance_length = self.args.instance_length

        # backbone networks
        model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        self.features = nn.Sequential(
            *list(model.children())[:-1])  # after avgpool 512x1
      
        self.graph_block = EnhancedGraphBlock(c_out=512, d_model=512, seq_len=4, gcn_depth=1, dropout=0.3, propalpha=0.7, node_dim=32)
        self.BiIlstm = BiImprovedLSTM(input_dim=512, hidden_dim=512, num_layers=4)
      
        # multi head self attention
        self.heads = 8
        self.dim_head = 1024 // self.heads
        self.scale = self.dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(
            1024, (self.dim_head * self.heads) * 3, bias=False)

        self.norm = DMIN(num_features=1024)
        self.pwconv = nn.Conv1d(4, 1, 3, 1, 1)

        # classifier
        self.fc = nn.Linear(1024, self.args.num_classes)
        #self.fc1 = nn.Linear(1024, 2048)
        #self.fc2 = nn.Linear(2048, self.args.num_classes)
        self.Softmax = nn.Softmax(dim=-1)

    def MIL(self, x):
        """The Multi Instance Learning Agregation of instances

        Inputs:
            x: [batch, bag_size, 512]
        """

        #self.lstm.flatten_parameters()
        #x, _ = self.lstm(x)
       # x = self.adp_lstm(x)
        x = self.graph_block(x)
        # 假设 GraphBlock 输出: graph_output [batch, seq_len, d_model]
        graph_weights = F.softmax(x, dim=-1)

        # 将 graph_output 作为 BiImprovedLSTM 的输入
      
        x = self.BiIlstm(x, graph_weights)
        # [batch, bag_size, 1024]
        ori_x = x


        #dp_logits = self.dp_instance_cluster.infer(x)

        # MHSA
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        x = torch.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.norm(x)
        x = torch.sigmoid(x)

        x = ori_x * x

        return x 

    def forward(self, x):
        try:
            # [batch, 16, 3, 112, 112]
            batch_size = x.size(0)
            
            # 确保输入维度正确
            if x.dim() != 5:
                raise ValueError(f"Expected 5D input tensor, got {x.dim()}D")
            
            x = rearrange(x, 'b (t1 t2) c h w -> (b t1) c t2 h w',
                          t1=4, t2=4)
            # [batch*bag_size, 3, il, 112, 112]

            x = self.features(x)
            # 安全的squeeze操作
            if x.dim() > 2:
                # 如果有多余的空间维度，移除它们
                while x.dim() > 2 and x.size(-1) == 1:
                    x = x.squeeze(-1)
                while x.dim() > 2 and x.size(-2) == 1:
                    x = x.squeeze(-2)
            
            # 确保最终是2维的 [batch*bag_size, 512]
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            
            # [batch*bag_size, 512]
            x = rearrange(x, '(b t) c -> b t c', t=4, b=batch_size)

            # [batch, bag_size, 512]
            x = self.MIL(x)
            # [batch, bag_size, 1024]

            # 安全的卷积和squeeze操作
            x = self.pwconv(x)
            if x.dim() > 2:
                x = x.squeeze(-1)  # 只squeeze最后一个维度
            
            x_FER = x
            # [batch, 1024]
            
            out = self.fc(x)
            # [batch, 7]

            return out, x_FER
            
        except Exception as e:
            # 如果出现错误，返回默认的tensor
            batch_size = x.size(0) if x.dim() > 0 else 1
            device = x.device if hasattr(x, 'device') else torch.device('cpu')
            
            # 返回默认输出
            default_out = torch.zeros(batch_size, self.args.num_classes, device=device)
            default_features = torch.zeros(batch_size, 1024, device=device)
            
            print(f"M3D forward error: {e}")
            return default_out, default_features


########################################################################################



class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        # x = torch.einsum('ncwl,wv->nclv',(x,A)
        return x.contiguous()


class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)


class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho


class EnhancedGraphBlock(nn.Module):
    def __init__(self, c_out, d_model, seq_len, conv_channel=32, skip_channel=32,
                 gcn_depth=32, dropout=0.1, propalpha=0.1, node_dim=10):
        super(EnhancedGraphBlock, self).__init__()

        # 初始化自适应图卷积的节点参数
        self.nodevec1 = nn.Parameter(torch.randn(c_out, node_dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(node_dim, c_out), requires_grad=True)

        # 初始卷积: 用于时间和空间维度特征提取
        self.start_conv = nn.Conv2d(1, conv_channel, (d_model - c_out + 1, 1))

        # 基于自适应邻接矩阵的图卷积
        self.gconv1 = mixprop(conv_channel, skip_channel, gcn_depth, dropout, propalpha)

        # 激活函数
        self.gelu = nn.GELU()

        # 结束卷积：结合时空信息，将 skip_channel 还原至 seq_len
        self.end_conv = nn.Conv2d(skip_channel, seq_len, (1, seq_len))

        # 线性层，确保输出维度与 d_model 一致
        self.linear = nn.Linear(c_out, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # 自适应邻接矩阵的生成
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

        # 增加通道维度
        out = x.unsqueeze(1).transpose(2, 3)  # [B, 1, seq_len, d_model]

        # 时间-空间特征提取
        out = self.start_conv(out)  # [B, conv_channel, T, 1]

        # 图卷积，学习节点之间的复杂关系
        out = self.gelu(self.gconv1(out, adp))  # [B, skip_channel, T, 1]

        # 最终卷积，将 skip_channel 降维到 seq_len
        out = self.end_conv(out).squeeze(-1)  # [B, seq_len, c_out]

        # 线性变换和归一化
        out = self.linear(out)  # [B, seq_len, d_model]
        return self.norm(x + out)  # 残差连接
    

    ####################################################################


class ImprovedLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ImprovedLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim)
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim)
        self.W_c = nn.Linear(input_dim, hidden_dim)
        self.U_c = nn.Linear(hidden_dim, hidden_dim)
        
        # 动态权重门
        self.W_s = nn.Linear(input_dim, hidden_dim)
        self.U_s = nn.Linear(hidden_dim, hidden_dim)
        self.dynamic_weight = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x, hidden_state, dynamic_weight):
        h_prev, c_prev = hidden_state

        i_t = torch.sigmoid(self.W_i(x) + self.U_i(h_prev))
        f_t = torch.sigmoid(self.W_f(x) + self.U_f(h_prev))
        o_t = torch.sigmoid(self.W_o(x) + self.U_o(h_prev))
        c_hat_t = torch.tanh(self.W_c(x) + self.U_c(h_prev))
        
        # 动态权重
        s_t = torch.sigmoid(self.W_s(x) + self.U_s(h_prev)) * dynamic_weight
        c_t = f_t * c_prev + i_t * c_hat_t * s_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

class BiImprovedLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(BiImprovedLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fwd_layers = nn.ModuleList([ImprovedLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        self.bwd_layers = nn.ModuleList([ImprovedLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        
        # 增强注意力模块
        self.attention = nn.MultiheadAttention(embed_dim=2*hidden_dim, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(2 * hidden_dim)

    def forward(self, x, graph_weights):
        batch_size, seq_len, _ = x.size()

        h_fwd = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        c_fwd = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        
        h_bwd = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]
        c_bwd = [torch.zeros(batch_size, self.hidden_dim, device=x.device) for _ in range(self.num_layers)]

        # 前向
        output_fwd = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer in range(self.num_layers):
                dynamic_weight = graph_weights[:, t, :]  # 动态权重来自GraphBlock
                h_fwd[layer], c_fwd[layer] = self.fwd_layers[layer](x_t, (h_fwd[layer], c_fwd[layer]), dynamic_weight)
                x_t = h_fwd[layer]
            output_fwd.append(h_fwd[-1])

        # 后向
        output_bwd = []
        for t in range(seq_len-1, -1, -1):
            x_t = x[:, t, :]
            for layer in range(self.num_layers):
                dynamic_weight = graph_weights[:, t, :]  # 动态权重来自GraphBlock
                h_bwd[layer], c_bwd[layer] = self.bwd_layers[layer](x_t, (h_bwd[layer], c_bwd[layer]), dynamic_weight)
                x_t = h_bwd[layer]
            output_bwd.append(h_bwd[-1])

        output_bwd.reverse()
        output = torch.cat([torch.stack(output_fwd, dim=1), torch.stack(output_bwd, dim=1)], dim=-1)

        # 融合注意力
        output, _ = self.attention(output, output, output)
        output = self.layer_norm(output)

        return output