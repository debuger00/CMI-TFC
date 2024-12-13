import torch.nn as nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Config:
    def __init__(self):
        # 时间序列对齐后的长度 (输入的序列长度)
        self.TSlength_aligned = 200  # 根据数据集中的时间序列长度设置
        self.dim_feedforward = 400  # Transformer 编码器中的前馈网络隐藏层大小
        self.num_heads = 2  # Transformer 编码器的多头注意力头数
        self.num_layers = 2  # Transformer 编码器的层数
        self.num_features = 6  # 每个时间步的特征维度
        #self.num_classes = 6  # 分类任务的类别数


class CaNet_TFC(nn.Module):
    def __init__(self, configs=Config(), num_classes=6):
        super().__init__()
        
        # 使用 TFC 中的 Transformer 编码器来替换原有的卷积块
        self.tfc_encoder_t = TFC(configs).transformer_encoder_t
        self.tfc_encoder_f = TFC(configs).transformer_encoder_f
        
        self.CA_C_s = BasicBlock()
        
        # 替换后的 projection 层，用于对时间域和频率域特征进行降维
        # self.projector_t = nn.Sequential(
        #     nn.Linear(configs.TSlength_aligned * configs.num_features, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256, 64)  # 降维到 64 以匹配后续的 concat 操作
        # )

        # self.projector_f = nn.Sequential(
        #     nn.Linear(configs.TSlength_aligned * configs.num_features, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256, 64)
        # )
        self.projector_t = nn.Sequential(
            nn.Linear(configs.TSlength_aligned * 3, 256),  # 200 * 3 = 600
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )

        self.projector_f = nn.Sequential(
            nn.Linear(configs.TSlength_aligned * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64)
)


        # Fully connected layer for final classification
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        xa = x[:, :, :, 0:3]
        xg = x[:, :, :, 3:6]
        
        # 去掉多余的通道维度
        xa = xa.squeeze(1)  # 形状变为 [B, L, 3]
        xg = xg.squeeze(1)  # 形状变为 [B, L, 3]

        # xa = xa.permute(0, 2, 3, 1)  # 重排维度以适配 Transformer，变为 [batch_size, sequence_length, num_features]
        # xg = xg.permute(0, 2, 3, 1)
        
        # 使用 TFC 的 Transformer 编码器进行特征提取
        xa_encoded = self.tfc_encoder_t(xa)
        xg_encoded = self.tfc_encoder_f(xg)
        
        # 展平 Transformer 编码器的输出
        xa_flat = xa_encoded.reshape(xa_encoded.shape[0], -1)
        xg_flat = xg_encoded.reshape(xg_encoded.shape[0], -1)
        
        # 使用投影层降维
        xa_proj = self.projector_t(xa_flat).unsqueeze(-1).unsqueeze(-1)
        xg_proj = self.projector_f(xg_flat).unsqueeze(-1).unsqueeze(-1)
        
        # 交叉注意力机制
        output_x, output_y, att_map_acc, att_map_gyr = self.CA_C_s(xa_proj, xg_proj)
        
        # 拼接加速度计和陀螺仪的特征
        output_cat = torch.cat((output_x, output_y), 1)
        
        # 自适应平均池化和全连接层用于最终的分类
        output_cat = self.avg_pool(output_cat)  # [batch_size, num_filters, 1, 1]
        output_cat = output_cat.view(output_cat.size(0), -1)  # [batch_size, num_filters]
        output = self.fc(output_cat)
        
        return output


class BasicBlock(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.FM_A = Feature_map_att(1, 16)

    def forward(self, x_acc, x_gyr):
        out_acc, out_gyr, att_map_acc, att_map_gyr = self.FM_A(x_acc, x_gyr)
        acc_output = out_acc + x_acc
        gyr_output = out_gyr + x_gyr
        
        return acc_output, gyr_output, att_map_acc, att_map_gyr


class Feature_map_att(nn.Module):  # CMIM 模型
    def __init__(self, input_channel=1, middle_channel=16):
        super().__init__()
        
        self.conv_combination1 = nn.Sequential(
            nn.Conv2d(2 * input_channel, middle_channel, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(inplace=True))
        self.conv_acc = nn.Sequential(
            nn.Conv2d(middle_channel, input_channel, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(input_channel),
            nn.Sigmoid())
        self.conv_gyr = nn.Sequential(
            nn.Conv2d(middle_channel, input_channel, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.BatchNorm2d(input_channel),
            nn.Sigmoid())
        
    def forward(self, f_acc, f_gyr):
        b, c, _, w = f_acc.size()
        squeeze_array = []
        for tensor in [f_acc, f_gyr]:
            tview = torch.mean(tensor, dim=1, keepdim=True, out=None)
            squeeze_array.append(tview)
        squeeze = torch.cat(squeeze_array, 1)

        excitation = self.conv_combination1(squeeze)
        acc_out = self.conv_acc(excitation)
        gyr_out = self.conv_gyr(excitation)
      
        return f_acc * acc_out.expand_as(f_acc), f_gyr * gyr_out.expand_as(f_gyr), acc_out, gyr_out


class TFC(nn.Module):
    def __init__(self, configs):
        super(TFC, self).__init__()

        # encoder_layers_t = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=2 * configs.TSlength_aligned, nhead=configs.num_heads)
        # self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, configs.num_layers)

        # encoder_layers_f = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=2 * configs.TSlength_aligned, nhead=configs.num_heads)
        # self.transformer_encoder_f = TransformerEncoder(encoder_layers_f, configs.num_layers)
        encoder_layers_t = TransformerEncoderLayer(
            d_model=3,   # 特征维度为3，因为xa是[B,200,3]
            dim_feedforward=6,  # 一般是2-4倍d_model，可以根据需要调整，这里用2*3=6
            nhead=1,
            batch_first=True
        )
        self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, configs.num_layers)

        encoder_layers_f = TransformerEncoderLayer(
            d_model=3, 
            dim_feedforward=6, 
            nhead=1,
            batch_first=True
        )
        self.transformer_encoder_f = TransformerEncoder(encoder_layers_f, configs.num_layers)


# 通过调用这个函数来创建新的 CaNet_TFC 模型
def canet_tfc():
    return CaNet_TFC()
