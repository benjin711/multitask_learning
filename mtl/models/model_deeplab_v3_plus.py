import torch
import torch.nn.functional as F

from mtl.models.model_parts import Encoder, get_encoder_channel_counts, ASPP, DecoderDeeplabV3p

class ModelDeepLabV3Plus(torch.nn.Module):
    def __init__(self, cfg, outputs_desc):
        super().__init__()
        self.outputs_desc = outputs_desc
        ch_out = sum(outputs_desc.values())

        self.encoder = Encoder(
            cfg.model_encoder_name,
            pretrained=cfg.use_resnet34_weights,
            zero_init_residual=True,
            replace_stride_with_dilation=(
                cfg.encdr_dltn_one, cfg.encdr_dltn_two, cfg.encdr_dltn_three),
        )

        ch_out_encoder_bottleneck, ch_out_encoder_4x,_,_ = get_encoder_channel_counts(
            cfg.model_encoder_name)

        self.aspp = ASPP(
            cfg.enable_aspp, ch_out_encoder_bottleneck, cfg.pyr_ch_shared, cfg.ch_out_shared_aspp,
            rates=(cfg.shared_aspp_dltn_lvl2, cfg.shared_aspp_dltn_lvl3, cfg.shared_aspp_dltn_lvl4))

        self.decoder = DecoderDeeplabV3p(
            cfg.enable_decoder, cfg.ch_out_shared_aspp, ch_out_encoder_4x, cfg.ch_out_skip_conv_shared, ch_out)

    def forward(self, x):
        input_resolution = (x.shape[2], x.shape[3])

        features = self.encoder(x)

        # Uncomment to see the scales of feature pyramid with their respective number of channels.
        # print(", ".join([f"{k}:{v.shape[1]}" for k, v in features.items()]))

        lowest_scale = max(features.keys())

        features_lowest = features[lowest_scale]

        features_tasks = self.aspp(features_lowest)

        predictions_4x, _ = self.decoder(features_tasks, features[4])

        predictions_1x = F.interpolate(
            predictions_4x, size=input_resolution, mode='bilinear', align_corners=False)

        out = {}
        offset = 0

        for task, num_ch in self.outputs_desc.items():
            out[task] = predictions_1x[:, offset:offset+num_ch, :, :]
            offset += num_ch

        return out
