import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(Attention, self).__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes,in_planes // ratio, bias = False, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, bias = False, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.att(x) * x


class softmax_Attention(nn.Module):
    def __init__(self, in_planes, ratio, K, temperature):
        super(softmax_Attention, self).__init__()
        self.dy_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, int(in_planes / ratio), bias = False, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(in_planes / ratio), K, bias = False, kernel_size=1),
            # nn.AvgPool2d(5, stride=2, padding=2),
            nn.Softmax(dim=1)
        )
        self.temperature = temperature

    def forward(self, x):
        # c = self.dy_att(x) # debug att map
        return self.dy_att(x)


class ConvELU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvELU, self).__init__()
        self.convelu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ELU()
        )

    def forward(self, x):
        return self.convelu(x)


class ChannelWiseAttention(nn.Module):
    def __init__(self, hf_channels, lf_channels, out_channel, scale=2):
        super(ChannelWiseAttention, self).__init__()
        self.in_channels = hf_channels + lf_channels
        self.out_channels = out_channel
        self.ca = Attention(self.in_channels)
        self.conv_se = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.upsample = nn.Upsample(scale_factor=scale, mode="nearest")

    def forward(self, high_features, low_features):
        features = torch.cat([self.upsample(high_features), low_features], dim=1)
        features = self.ca(features)
        return self.conv_se(features)


class DCAM(nn.Module):
    def __init__(self, q_planes, k_planes, v_planes, mid_planes=32,
                 down_scale_q=1, down_scale_k=1, down_scale_v=1):
        super(DCAM, self).__init__()
        if down_scale_q > 1:
            self.avg_pool_q = nn.AvgPool2d(down_scale_q, stride=down_scale_q, padding=0)
        else:
            self.avg_pool_q = nn.Identity()
        if down_scale_k > 1:
            self.avg_pool_k = nn.AvgPool2d(down_scale_k, stride=down_scale_k, padding=0)
        else:
            self.avg_pool_k = nn.Identity()
        self.avg_pool_v = nn.AvgPool2d(down_scale_v, stride=down_scale_v, padding=0)
        self.query_conv = nn.Conv2d(q_planes, mid_planes, bias = False, kernel_size=1)
        self.key_conv = nn.Conv2d(k_planes, mid_planes, bias = False, kernel_size=1)
        self.value_conv = nn.Conv2d(v_planes, v_planes, bias = False, kernel_size=1)
        self.up_sample = nn.Upsample(scale_factor=down_scale_v, mode="nearest")

    def forward(self, feature_q, feature_k, feature_v):
        q = self.avg_pool_q(feature_q)
        k = self.avg_pool_k(feature_k)
        v = self.avg_pool_v(feature_v)
        B, C, H, W = v.size()
        q = self.query_conv(q).view(B,-1,H*W).permute(0,2,1)  # B, H*W, 32
        k = self.key_conv(k).view(B,-1,H*W)  # B, 32, H*W
        v = self.value_conv(v).view(B,-1,H*W)  # B,C,H*W

        return self.up_sample(torch.bmm(v, torch.softmax(torch.bmm(q, k), dim=-1).permute(0,2,1)).view(B, C, H, W)) + feature_v


# cross_attention_v2
class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc):
        super(DepthDecoder, self).__init__()
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = [16, 32, 64, 128, 256]

        self.convs_up_x9_0 = ConvELU(self.num_ch_dec[1],self.num_ch_dec[0])
        self.convs_up_x9_1 = ConvELU(self.num_ch_dec[0],self.num_ch_dec[0])
        self.deconvs_72 = ChannelWiseAttention(self.num_ch_enc[4]  , self.num_ch_enc[3] * 2, 256)
        self.deconvs_36 = ChannelWiseAttention(256, self.num_ch_enc[2] * 3, 128)
        self.deconvs_18 = ChannelWiseAttention(128, self.num_ch_enc[1] * 3 + 64 , 64)
        self.deconvs_9 = ChannelWiseAttention(64, 64, 32, scale=2)

        # self.qkv = Dynamic_QKV(32)

        self.disp_head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.dropout = nn.Dropout2d(0.1)

        self.attention = softmax_Attention(144, ratio=9, K=4, temperature=4)

        self.QKV18 = DCAM(q_planes=64, k_planes=64, v_planes=32,
                                down_scale_q=2, down_scale_k=2, down_scale_v=4)
        self.QKV36 = DCAM(q_planes=128, k_planes=128, v_planes=32,
                                down_scale_q=1, down_scale_k=1, down_scale_v=4)
        self.QKV72 = DCAM(q_planes=256, k_planes=256, v_planes=32,
                                down_scale_q=1, down_scale_k=1, down_scale_v=8)

        self.convs_up_x9_0 = ConvELU(self.num_ch_dec[1],self.num_ch_dec[0])
        self.convs_up_x9_1 = ConvELU(self.num_ch_dec[0],self.num_ch_dec[0])

    def forward(self, input_features):
        self.outputs = {}
        feature144 = input_features[4]
        feature72 = input_features[3]
        feature36 = input_features[2]
        feature18 = input_features[1]
        feature64 = input_features[0]

        x72 = self.deconvs_72(feature144, feature72)
        x36 = self.deconvs_36(x72 , feature36)
        x18 = self.deconvs_18(x36 , feature18)
        x9 = self.deconvs_9(x18, feature64)
        x9 = self.dropout(x9)

        weight_map = self.attention(feature144).unsqueeze(2)
        qkv_all = torch.stack((x9,
                               self.QKV18(x18, x18, x9),
                               self.QKV36(x36, x36, x9),
                               self.QKV72(x72, x72, x9)), dim=1)

        qkv_all = qkv_all * weight_map
        qkv_all = torch.sum(qkv_all, dim=1)

        x6 = self.convs_up_x9_1(self.upsample(self.convs_up_x9_0(qkv_all)))

        self.outputs[("disp", 0, 0)] = self.disp_head(x6)
        return self.outputs

# # depth_decoder without Dynamic Cross att 
# class DepthDecoder(nn.Module):
#     def __init__(self, num_ch_enc):
#         super(DepthDecoder, self).__init__()
#         self.num_ch_enc = num_ch_enc
#         self.num_ch_dec = [16, 32, 64, 128, 256]

#         self.convs_up_x9_0 = ConvELU(self.num_ch_dec[1],self.num_ch_dec[0])
#         self.convs_up_x9_1 = ConvELU(self.num_ch_dec[0],self.num_ch_dec[0])
#         self.deconvs_72 = ChannelWiseAttention(self.num_ch_enc[4]  , self.num_ch_enc[3] * 2, 256)
#         self.deconvs_36 = ChannelWiseAttention(256, self.num_ch_enc[2] * 3, 128)
#         self.deconvs_18 = ChannelWiseAttention(128, self.num_ch_enc[1] * 3 + 64 , 64)
#         self.deconvs_9 = ChannelWiseAttention(64, 64, 32)

#         self.dispConvScale0 = nn.Sequential(
#             nn.Conv2d(self.num_ch_dec[0], 1, kernel_size=3, padding=1, bias=True),
#             nn.Sigmoid()
#         )
#         # self.dispConvScale1 = nn.Sequential(
#         #     nn.Conv2d(self.num_ch_dec[1], 1, kernel_size=3, padding=1, bias=True),
#         #     nn.Sigmoid()
#         # )
#         # self.dispConvScale2 = nn.Sequential(
#         #     nn.Conv2d(self.num_ch_dec[2], 1, kernel_size=3, padding=1, bias=True),
#         #     nn.Sigmoid()
#         # )
#         # self.dispConvScale3 = nn.Sequential(
#         #     nn.Conv2d(self.num_ch_dec[3], 1, kernel_size=3, padding=1, bias=True),
#         #     nn.Sigmoid()
#         # )
#         self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

#     def forward(self, input_features):
#         self.outputs = {}
#         feature144 = input_features[4]
#         feature72 = input_features[3]
#         feature36 = input_features[2]
#         feature18 = input_features[1]
#         feature64 = input_features[0]
#         x72 = self.deconvs_72(feature144, feature72)
#         x36 = self.deconvs_36(x72 , feature36)
#         x18 = self.deconvs_18(x36 , feature18)
#         x9 = self.deconvs_9(x18, feature64)
#         x6 = self.convs_up_x9_1(self.upsample(self.convs_up_x9_0(x9)))

#         self.outputs[("disp", 0, 0)] = self.dispConvScale0(x6)
#         # self.outputs[("disp", 0, 1)] = self.dispConvScale1(x9)
#         # self.outputs[("disp", 0, 2)] = self.dispConvScale2(x18)
#         # self.outputs[("disp", 0, 3)] = self.dispConvScale3(x36)
#         return self.outputs