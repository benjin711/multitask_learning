import torch
import torch.nn.functional as F

from mtl.models.model_parts import Encoder, get_encoder_channel_counts, ASPP, DecoderDeeplabV3p


class ModelDeepLabV3PlusBranched(torch.nn.Module):
    def __init__(self, cfg, outputs_desc):
        super().__init__()
        self.outputs_desc = outputs_desc

        self.encoder = Encoder(
            cfg.model_encoder_name,
            pretrained=cfg.use_resnet34_weights,
            zero_init_residual=True,
            replace_stride_with_dilation=(
                cfg.encdr_dltn_one, cfg.encdr_dltn_two, cfg.encdr_dltn_three),
        )

        ch_out_encoder_bottleneck, ch_out_encoder_4x, _, _ = get_encoder_channel_counts(
            cfg.model_encoder_name)

        self.aspp_semseg = ASPP(
            cfg.enable_aspp, ch_out_encoder_bottleneck, cfg.pyr_ch_semseg_aspp, cfg.ch_out_semseg_aspp,
            rates=(cfg.semseg_aspp_dltn_lvl2, cfg.semseg_aspp_dltn_lvl3, cfg.semseg_aspp_dltn_lvl4))

        self.aspp_depth = ASPP(
            cfg.enable_aspp, ch_out_encoder_bottleneck, cfg.pyr_ch_depth_aspp, cfg.ch_out_depth_aspp,
            rates=(cfg.depth_aspp_dltn_lvl2, cfg.depth_aspp_dltn_lvl3, cfg.depth_aspp_dltn_lvl4))

        self.decoder_semseg = DecoderDeeplabV3p(
            cfg.enable_decoder, cfg.ch_out_semseg_aspp, ch_out_encoder_4x, cfg.ch_out_skip_conv_semseg, outputs_desc['semseg'])

        self.decoder_depth = DecoderDeeplabV3p(
            cfg.enable_decoder, cfg.ch_out_depth_aspp, ch_out_encoder_4x, cfg.ch_out_skip_conv_depth, outputs_desc['depth'])

    def forward(self, x):
        input_resolution = (x.shape[2], x.shape[3])

        features = self.encoder(x)

        lowest_scale = max(features.keys())

        features_lowest = features[lowest_scale]

        # SEMSEG BRANCH
        semseg_features_tasks = self.aspp_semseg(features_lowest)

        semseg_predictions_4x, s_ = self.decoder_semseg(
            semseg_features_tasks, features[4])

        # DEPTH BRANCH
        depth_features_tasks = self.aspp_depth(features_lowest)

        depth_predictions_4x, s_ = self.decoder_depth(
            depth_features_tasks, features[4])

        # CONCATENATE AND UPSCALING
        predictions_4x = torch.cat(
            (semseg_predictions_4x, depth_predictions_4x), axis=1)

        predictions_1x = F.interpolate(
            predictions_4x, size=input_resolution, mode='bilinear', align_corners=False)

        out = {}
        offset = 0

        for task, num_ch in self.outputs_desc.items():
            out[task] = predictions_1x[:, offset:offset+num_ch, :, :]
            offset += num_ch

        return out
