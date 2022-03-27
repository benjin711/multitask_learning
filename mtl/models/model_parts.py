import torch
import torch.nn.functional as F
import torchvision.models.resnet as resnet

# ---------
# ENCODER
# ---------

class BasicBlockWithDilation(torch.nn.Module):
    """Workaround for prohibited dilation in BasicBlock in 0.4.0"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockWithDilation, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = resnet.conv3x3(inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = torch.nn.ReLU()
        self.conv2 = resnet.conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

_basic_block_layers = {
    'resnet18': (2, 2, 2, 2),
    'resnet34': (3, 4, 6, 3),
}

def get_encoder_channel_counts(encoder_name):
    is_basic_block = encoder_name in _basic_block_layers
    ch_out_encoder_bottleneck = 512 if is_basic_block else 2048
    ch_out_encoder_4x = 64 if is_basic_block else 256
    ch_out_encoder_2x = 64
    ch_out_encoder_1x = 3
    return ch_out_encoder_bottleneck, ch_out_encoder_4x, ch_out_encoder_2x, ch_out_encoder_1x

class Encoder(torch.nn.Module):
    def __init__(self, name, **encoder_kwargs):
        super().__init__()
        encoder = self._create(name, **encoder_kwargs)
        del encoder.avgpool
        del encoder.fc
        self.encoder = encoder

    def _create(self, name, **encoder_kwargs):
        if name not in _basic_block_layers.keys():
            fn_name = getattr(resnet, name)
            model = fn_name(**encoder_kwargs)
        else:
            # special case due to prohibited dilation in the original BasicBlock
            pretrained = encoder_kwargs.pop('pretrained', False)
            progress = encoder_kwargs.pop('progress', True)
            model = resnet._resnet(
                name, BasicBlockWithDilation, _basic_block_layers[name], pretrained, progress, **encoder_kwargs
            )
        replace_stride_with_dilation = encoder_kwargs.get(
            'replace_stride_with_dilation', (False, False, False))
        assert len(replace_stride_with_dilation) == 3
        if replace_stride_with_dilation[0]:
            model.layer2[0].conv2.padding = (2, 2)
            model.layer2[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[1]:
            model.layer3[0].conv2.padding = (2, 2)
            model.layer3[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[2]:
            model.layer4[0].conv2.padding = (2, 2)
            model.layer4[0].conv2.dilation = (2, 2)
        return model

    def update_skip_dict(self, skips, x, sz_in):
        rem, scale = sz_in % x.shape[3], sz_in // x.shape[3]
        assert rem == 0
        skips[scale] = x

    def forward(self, x):
        """
        DeepLabV3+ style encoder
        :param x: RGB input of reference scale (1x)
        :return: dict(int->Tensor) feature pyramid mapping downscale factor to a tensor of features
        """
        out = {1: x}
        sz_in = x.shape[3]

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.maxpool(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer1(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer2(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer3(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer4(x)
        self.update_skip_dict(out, x, sz_in)

        return out

# --------------
# BASIC DECODER
# --------------

class DecoderPart(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                            stride, padding, dilation, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )

class DecoderDeeplabV3p(torch.nn.Module):
    def __init__(self, enable_decoder, bottleneck_ch, skip_4x_ch, ch_out_skip_conv, num_out_ch):
        super(DecoderDeeplabV3p, self).__init__()

        self.enable_decoder = enable_decoder

        # For compatibility with naive model
        if not self.enable_decoder:
            self.features_to_predictions = torch.nn.Conv2d(
                bottleneck_ch, num_out_ch, kernel_size=1, stride=1)
        else:
            self.skip_conv = DecoderPart(
                skip_4x_ch, ch_out_skip_conv, kernel_size=1, stride=1, padding=0, dilation=1)

            self.conv3x3_final = torch.nn.Conv2d(
                ch_out_skip_conv + bottleneck_ch, num_out_ch, kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, features_bottleneck, features_skip_4x):
        """
        DeepLabV3+ style decoder
        :param features_bottleneck: bottleneck features of scale > 4
        :param features_skip_4x: features of encoder of scale == 4
        :return: features with 256 channels and the final tensor of predictions
        """
        # For compatibility with naive model
        if not self.enable_decoder:
            features_4x = F.interpolate(
                features_bottleneck, size=features_skip_4x.shape[2:], mode='bilinear', align_corners=False
            )
            predictions_4x = self.features_to_predictions(features_4x)
            return predictions_4x, features_4x

        # SKIP CONNECTION: LOW-LEVEL FEATURES
        low_lvl_features = self.skip_conv(features_skip_4x)

        # UPSAMPLING
        features_4x = F.interpolate(
            features_bottleneck, size=features_skip_4x.shape[2:], mode='bilinear', align_corners=False)

        # CONCATENATION
        conc = torch.cat((low_lvl_features, features_4x), axis=1)

        # FUSION
        output = self.conv3x3_final(conc)

        return output, conc

# -------------------
# U-NET-LIKE DECODER
# -------------------

class DecoderDeeplabV3p_Unet(torch.nn.Module):
    def __init__(self, enable_decoder, bottleneck_ch_dict, feat_skip_shape, ch_out_skip_dict, num_out_ch):
        super(DecoderDeeplabV3p_Unet, self).__init__()

        self.enable_decoder = enable_decoder

        # For compatibility with naive model
        if not self.enable_decoder:
            self.features_to_predictions = torch.nn.Conv2d(
                bottleneck_ch_dict[4], num_out_ch, kernel_size=1, stride=1)
        else:
            self.skip_4x_conv = DecoderPart(
                feat_skip_shape[4], ch_out_skip_dict[4], kernel_size=1, stride=1, padding=0, dilation=1)

            self.conv3x3_4x = DecoderPart(
                ch_out_skip_dict[4] + bottleneck_ch_dict[4],
                bottleneck_ch_dict[2], kernel_size=3, stride=1, padding=1, dilation=1)

            self.skip_2x_conv = DecoderPart(
                feat_skip_shape[2], ch_out_skip_dict[2], kernel_size=1, stride=1, padding=0, dilation=1)

            self.conv3x3_2x = DecoderPart(
                ch_out_skip_dict[2] + bottleneck_ch_dict[2],
                bottleneck_ch_dict[1], kernel_size=3, stride=1, padding=1, dilation=1)

            self.skip_1x_conv = DecoderPart(
                feat_skip_shape[1], ch_out_skip_dict[1], kernel_size=1, stride=1, padding=0, dilation=1)

            self.conv3x3_final = torch.nn.Conv2d(
                ch_out_skip_dict[1] + bottleneck_ch_dict[1],
                num_out_ch, kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, features_bottleneck, features_skip_4x, features_skip_2x, features_skip_1x):

        # For compatibility with naive model
        if not self.enable_decoder:
            features_4x = F.interpolate(
                features_bottleneck, size=features_skip_4x.shape[2:], mode='bilinear', align_corners=False
            )
            predictions_4x = self.features_to_predictions(features_4x)
            return predictions_4x, features_4x

        # x4-LEVEL
        # Low-level feature adaptation
        low_features_4x = self.skip_4x_conv(features_skip_4x)

        bottleneck_4x = F.interpolate(                                  # High-level feature upsampling
            features_bottleneck, size=features_skip_4x.shape[2:], mode='bilinear', align_corners=False)

        conc_4x = torch.cat((low_features_4x, bottleneck_4x),
                            axis=1)   # Concatenation

        features_bottleneck_2x = self.conv3x3_4x(
            conc_4x)               # Fusion

        # x2-LEVEL
        # Low-level feature adaptation
        low_features_2x = self.skip_2x_conv(features_skip_2x)

        bottleneck_2x = F.interpolate(                                  # High-level feature upsampling
            features_bottleneck_2x, size=features_skip_2x.shape[2:], mode='bilinear', align_corners=False)

        conc_2x = torch.cat((low_features_2x, bottleneck_2x),
                            axis=1)   # Concatenation

        features_bottleneck_1x = self.conv3x3_2x(
            conc_2x)               # Fusion

        # x1-LEVEL
        # Low-level feature adaptation
        low_features_1x = self.skip_1x_conv(features_skip_1x)

        bottleneck_1x = F.interpolate(                                  # High-level feature upsampling
            features_bottleneck_1x, size=features_skip_1x.shape[2:], mode='bilinear', align_corners=False)

        conc_1x = torch.cat((low_features_1x, bottleneck_1x),
                            axis=1)   # Concatenation

        output = self.conv3x3_final(
            conc_1x)                            # Fusion

        return output, {'1x': conc_1x, '2x': conc_2x, '4x': conc_4x, '8x': features_bottleneck}


# --------------------
# DECODER FOR PAD-NET
# --------------------

class DecoderDeeplabV3pDist(torch.nn.Module):
    def __init__(self, enable_decoder, decoder_ch, sa_ch, num_out_ch):
        super(DecoderDeeplabV3pDist, self).__init__()

        self.enable_decoder = enable_decoder

        # For compatibility with naive model
        if not self.enable_decoder:
            self.features_to_predictions = torch.nn.Conv2d(
                decoder_ch, num_out_ch, kernel_size=1, stride=1)
        else:
            self.conv3x3_final = torch.nn.Conv2d(
                decoder_ch, num_out_ch, kernel_size=3, stride=1, padding=1, dilation=1)

    def forward(self, features_decoder, features_sa):

        # For compatibility with naive model
        if not self.enable_decoder:
            features = F.interpolate(
                features_decoder, size=features_sa.shape[2:], mode='bilinear', align_corners=False
            )
            predictions = self.features_to_predictions(features)
            return predictions, features

        # ADD SA OUTPUT AND DECODER OUTPUT
        final_features = features_decoder + features_sa

        # FUSION
        output = self.conv3x3_final(final_features)

        return output, final_features

# -----
# ASPP
# -----

class ASPPpart(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                            stride, padding, dilation, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )


class ASPP(torch.nn.Module):
    # in_channels is 512, out_channels is 256
    def __init__(self, enable_aspp, in_channels, pyramid_channels, out_channels, rates):
        super().__init__()

        self.enable_aspp = enable_aspp

        # For compatibility with naive model
        if not self.enable_aspp:
            self.default_conv_out = ASPPpart(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        else:
            self.lvl1_conv = ASPPpart(
                in_channels, pyramid_channels, kernel_size=1, stride=1, padding=0, dilation=1)
            self.lvl2_conv = ASPPpart(
                in_channels, pyramid_channels, kernel_size=3, stride=1, padding=rates[0], dilation=rates[0])
            self.lvl3_conv = ASPPpart(
                in_channels, pyramid_channels, kernel_size=3, stride=1, padding=rates[1], dilation=rates[1])
            self.lvl4_conv = ASPPpart(
                in_channels, pyramid_channels, kernel_size=3, stride=1, padding=rates[2], dilation=rates[2])

            self.conv_out = ASPPpart(
                in_channels + 4 * pyramid_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)

    def forward(self, x):
        # For compatibility with naive model
        if not self.enable_aspp:
            out = self.default_conv_out(x)
            return out

        lvl1_out = self.lvl1_conv(x)
        lvl2_out = self.lvl2_conv(x)
        lvl3_out = self.lvl3_conv(x)
        lvl4_out = self.lvl4_conv(x)

        # GLOBAL AVERAGE POOLING
        global_pooling = torch.nn.AvgPool2d(
            kernel_size=x.shape[2:])
        upsampler = torch.nn.Upsample(
            size=x.shape[2:], mode='bilinear')

        global_features = global_pooling(x)
        global_features = upsampler(global_features)

        # CONCATENATION
        conc = torch.cat((lvl1_out, lvl2_out, lvl3_out,
                          lvl4_out, global_features), axis=1)

        # FINAL CONVOLUTION
        out = self.conv_out(conc)

        return out

# ----------------------
# Self-Attention Module
# ----------------------

class SelfAttention(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.attention = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        with torch.no_grad():
            self.attention.weight.copy_(
                torch.zeros_like(self.attention.weight))

    def forward(self, x):
        features = self.conv(x)
        attention_mask = torch.sigmoid(self.attention(x))
        return features * attention_mask


# --------------------------
# Task Propagation (PAP-Net)
# --------------------------

class CrossTaskPropagationPart(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                            stride, padding, dilation, bias=False),
            torch.nn.BatchNorm2d(out_channels),
        )


class CrossTaskPropagation(torch.nn.Module):
    def __init__(self, in_channel_sizes):
        super().__init__()

        # Convolution for reducing the channel size by a factor of 2
        self.conv_half_channel_size_semseg = CrossTaskPropagationPart(
            in_channel_sizes["semseg"], int(in_channel_sizes["semseg"] / 2), kernel_size=1, stride=1, padding=0, dilation=1)

        self.conv_half_channel_size_depth = CrossTaskPropagationPart(
            in_channel_sizes["depth"], int(in_channel_sizes["depth"] / 2), kernel_size=1, stride=1, padding=0, dilation=1)

        # Adaptive combination
        self.adaptive_combination_semseg = torch.nn.Conv2d(
            in_channels=2, out_channels=1, kernel_size=1, padding=0, dilation=1, groups=1, bias=False)
        self.adaptive_combination_depth = torch.nn.Conv2d(
            in_channels=2, out_channels=1, kernel_size=1, padding=0, dilation=1, groups=1, bias=False)

        self.alpha_1 = torch.nn.Parameter(
            torch.tensor([0.5]), requires_grad=True)
        self.alpha_2 = torch.nn.Parameter(
            torch.tensor([0.5]), requires_grad=True)

    def forward(self, task_specific_features_dict):

        # Split the task specific concatenated semseg and depth features again
        semseg_features = task_specific_features_dict['semseg']['8x']
        depth_features = task_specific_features_dict['depth']['8x']

        # Reduce the task specific feature's channel size to a half
        semseg_features = self.conv_half_channel_size_semseg(semseg_features)
        depth_features = self.conv_half_channel_size_depth(depth_features)

        # Calculate the task specific affinity matrices

        # Permute: BxCxHxW -> BxHxWxC
        semseg_features = semseg_features.permute((0, 2, 3, 1))
        depth_features = depth_features.permute((0, 2, 3, 1))

        # Reshape: BxHxWxC -> BxHWxC
        semseg_features = semseg_features.view(
            semseg_features.shape[0], -1, semseg_features.shape[3])
        depth_features = depth_features.view(
            depth_features.shape[0], -1, depth_features.shape[3])

        # Transpose (aka permute): BxHWxC -> BxCxHW
        semseg_features_T = semseg_features.permute((0, 2, 1))
        depth_features_T = depth_features.permute((0, 2, 1))

        # Affinity matrix-> X·X_T => BxHWxHW
        aff_mat_semseg = torch.bmm(semseg_features, semseg_features_T)
        aff_mat_depth = torch.bmm(depth_features, depth_features_T)

        # Normalize the rows of the affinity matrix
        aff_mat_semseg = aff_mat_semseg.float() / torch.clamp(aff_mat_semseg.norm(p=1,
                                                                                  dim=1, keepdim=True)[0], min=1e-12)
        aff_mat_depth = aff_mat_depth.float() / torch.clamp(aff_mat_depth.norm(p=1,
                                                                               dim=1, keepdim=True)[0], min=1e-12)

        alpha_1 = torch.clamp(self.alpha_1, min=0, max=1)
        alpha_2 = torch.clamp(self.alpha_2, min=0, max=1)

        aff_mat_semseg = alpha_1*aff_mat_semseg + (1-alpha_1)*aff_mat_depth
        aff_mat_depth = alpha_2*aff_mat_depth + (1-alpha_2)*aff_mat_semseg

        return {'semseg': aff_mat_semseg, 'depth': aff_mat_depth}


class TaskSpecificPropagation(torch.nn.Module):
    def __init__(self, n_iterations, diffusion_weight):
        super().__init__()

        self.n_iterations = n_iterations
        self.diffusion_weight = diffusion_weight

    def forward(self, task_specific_features_dict, affinity_matrix_dict):
        # Inputs
        semseg_features_0 = task_specific_features_dict['semseg']['8x']
        depth_features_0 = task_specific_features_dict['depth']['8x']

        aff_mat_semseg = affinity_matrix_dict['semseg']
        aff_mat_depth = affinity_matrix_dict['depth']

        # Permute: BxCxHxW -> BxHxWxC
        semseg_features_diff = semseg_features_0.permute((0, 2, 3, 1))
        depth_features_diff = depth_features_0.permute((0, 2, 3, 1))

        # Reshape: BxHxWxC -> BxHWxC
        semseg_features_diff = semseg_features_diff.view(
            semseg_features_diff.shape[0], -1, semseg_features_diff.shape[3])
        depth_features_diff = depth_features_diff.view(
            depth_features_diff.shape[0], -1, depth_features_diff.shape[3])

        # Start iterative diffusion process
        for t in range(0, self.n_iterations):
            semseg_features_diff = torch.bmm(
                aff_mat_semseg, semseg_features_diff)
            depth_features_diff = torch.bmm(aff_mat_depth, depth_features_diff)

        # Permute: BxHWxC -> BxCxHW
        semseg_features_diff = semseg_features_diff.permute((0, 2, 1))
        depth_features_diff = depth_features_diff.permute((0, 2, 1))

        # Reshaping features back to initial shape: BxCxHW -> BxCxHxW
        semseg_features_diff = semseg_features_diff.view(
            semseg_features_0.shape)
        depth_features_diff = depth_features_diff.view(depth_features_0.shape)

        # Weighted sum of initial and diffused features
        semseg_features = self.diffusion_weight*semseg_features_diff + \
            (1-self.diffusion_weight)*semseg_features_0
        depth_features = self.diffusion_weight*depth_features_diff + \
            (1-self.diffusion_weight)*depth_features_0

        return {'semseg': semseg_features, 'depth': depth_features}


# --------------------------
# Squeeze and Excitation (not used)
# --------------------------

class SqueezeAndExcitation(torch.nn.Module):
    """
    Squeeze and excitation module as explained in https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, channels, r=16):
        super(SqueezeAndExcitation, self).__init__()
        self.transform = torch.nn.Sequential(
            torch.nn.Linear(channels, channels // r),
            torch.nn.ReLU(),
            torch.nn.Linear(channels // r, channels),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        N, C, H, W = x.shape
        squeezed = torch.mean(x, dim=(2, 3)).reshape(N, C)
        squeezed = self.transform(squeezed).reshape(N, C, 1, 1)
        return x * squeezed
