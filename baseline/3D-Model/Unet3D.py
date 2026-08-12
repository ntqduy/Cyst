from torch.nn import Module, Sequential
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, Dropout3d
from torch.nn import ReLU
import torch


def _match_spatial_3d(source, reference):
    if source.shape[-3:] == reference.shape[-3:]:
        return source
    return torch.nn.functional.interpolate(source, size=reference.shape[-3:], mode="trilinear", align_corners=False)


class UNet(Module):
    # __                            __
    #  1|__   ________________   __|1
    #     2|__  ____________  __|2
    #        3|__  ______  __|3
    #           4|__ __ __|4
    # The convolution operations on either side are residual subject to 1*1 Convolution for channel homogeneity

    def __init__(self, in_dim=1, out_dim=2, feat_channels=(64, 256, 256, 512, 1024), residual='conv'):
        # residual: conv for residual input x through 1*1 conv across every layer for downsampling, None for removal of residuals

        super(UNet, self).__init__()
        self.model_name = "unet"
        self.backbone_name = "unet3d_encoder"
        self.architecture_config = {
            "in_dim": int(in_dim),
            "out_dim": int(out_dim),
            "feat_channels": list(feat_channels),
            "residual": residual,
        }

        # Encoder downsamplers
        self.pool1 = MaxPool3d((2, 2, 2))
        self.pool2 = MaxPool3d((2, 2, 2))
        self.pool3 = MaxPool3d((2, 2, 2))
        self.pool4 = MaxPool3d((2, 2, 2))

        # Encoder convolutions
        self.conv_blk1 = Conv3D_Block(in_dim, feat_channels[0], residual=residual)
        self.conv_blk2 = Conv3D_Block(feat_channels[0], feat_channels[1], residual=residual)
        self.conv_blk3 = Conv3D_Block(feat_channels[1], feat_channels[2], residual=residual)
        self.conv_blk4 = Conv3D_Block(feat_channels[2], feat_channels[3], residual=residual)
        self.conv_blk5 = Conv3D_Block(feat_channels[3], feat_channels[4], residual=residual)

        # Decoder convolutions
        self.dec_conv_blk4 = Conv3D_Block(2 * feat_channels[3], feat_channels[3], residual=residual)
        self.dec_conv_blk3 = Conv3D_Block(2 * feat_channels[2], feat_channels[2], residual=residual)
        self.dec_conv_blk2 = Conv3D_Block(2 * feat_channels[1], feat_channels[1], residual=residual)
        self.dec_conv_blk1 = Conv3D_Block(2 * feat_channels[0], feat_channels[0], residual=residual)

        # Decoder upsamplers
        self.deconv_blk4 = Deconv3D_Block(feat_channels[4], feat_channels[3])
        self.deconv_blk3 = Deconv3D_Block(feat_channels[3], feat_channels[2])
        self.deconv_blk2 = Deconv3D_Block(feat_channels[2], feat_channels[1])
        self.deconv_blk1 = Deconv3D_Block(feat_channels[1], feat_channels[0])

        # Final 1*1 Conv Segmentation map
        self.one_conv = Conv3d(feat_channels[0], out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.dropout = Dropout3d(p=0.5)

    def forward(self, x, return_features: bool = False):
        # Encoder part

        x1 = self.conv_blk1(x)

        x_low1 = self.pool1(x1)
        x2 = self.conv_blk2(x_low1)

        x_low2 = self.pool2(x2)
        x3 = self.conv_blk3(x_low2)

        x_low3 = self.pool3(x3)
        x4 = self.conv_blk4(x_low3)

        x_low4 = self.pool4(x4)
        base = self.conv_blk5(x_low4)

        # Decoder part

        d4_up = _match_spatial_3d(self.deconv_blk4(base), x4)
        d4 = torch.cat([d4_up, x4], dim=1)
        d_high4 = self.dec_conv_blk4(d4)

        d3_up = _match_spatial_3d(self.deconv_blk3(d_high4), x3)
        d3 = torch.cat([d3_up, x3], dim=1)
        d_high3 = self.dec_conv_blk3(d3)
        d_high3 = self.dropout(d_high3)

        d2_up = _match_spatial_3d(self.deconv_blk2(d_high3), x2)
        d2 = torch.cat([d2_up, x2], dim=1)
        d_high2 = self.dec_conv_blk2(d2)
        d_high2 = self.dropout(d_high2)

        d1_up = _match_spatial_3d(self.deconv_blk1(d_high2), x1)
        d1 = torch.cat([d1_up, x1], dim=1)
        d_high1 = self.dec_conv_blk1(d1)

        seg = self.one_conv(d_high1)

        if return_features:
            features = {
                "encoder": [x1, x2, x3, x4, base],
                "bottleneck": base,
                "decoder": {
                    "up4": d_high4,
                    "up3": d_high3,
                    "up2": d_high2,
                    "up1": d_high1,
                    "final": d_high1,
                },
            }
            return seg, features

        return seg



class Conv3D_Block(Module):

    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=None):

        super(Conv3D_Block, self).__init__()

        self.conv1 = Sequential(
            Conv3d(inp_feat, out_feat, kernel_size=kernel,
                   stride=stride, padding=padding, bias=True),
            BatchNorm3d(out_feat),
            ReLU())

        self.conv2 = Sequential(
            Conv3d(out_feat, out_feat, kernel_size=kernel,
                   stride=stride, padding=padding, bias=True),
            BatchNorm3d(out_feat),
            ReLU())

        self.residual = residual

        if self.residual is not None:
            self.residual_upsampler = Conv3d(inp_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):

        res = x

        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)


class Deconv3D_Block(Module):

    def __init__(self, inp_feat, out_feat, kernel=3, stride=2, padding=1):
        super(Deconv3D_Block, self).__init__()

        self.deconv = Sequential(
            ConvTranspose3d(inp_feat, out_feat, kernel_size=(kernel, kernel, kernel),
                            stride=(stride, stride, stride), padding=(padding, padding, padding), output_padding=1, bias=True),
            ReLU())

    def forward(self, x):
        return self.deconv(x)
