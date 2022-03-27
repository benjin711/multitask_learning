import torch
import torch.nn.functional as F

from mtl.models.model_parts import *


class ModelDeepLabV3PlusTaskDist_Unet(torch.nn.Module):
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

        ch_out_encoder_bottleneck, ch_out_encoder_4x, ch_out_encoder_2x, ch_out_encoder_1x = get_encoder_channel_counts(
            cfg.model_encoder_name)

        self.aspp1 = ASPP(
            cfg.enable_aspp, ch_out_encoder_bottleneck, cfg.pyr_ch_aspp1, cfg.ch_out_aspp1,
            rates=(cfg.aspp1_dltn_lvl2, cfg.aspp1_dltn_lvl3, cfg.aspp1_dltn_lvl4))

        self.aspp2 = ASPP(
            cfg.enable_aspp, ch_out_encoder_bottleneck, cfg.pyr_ch_aspp2, cfg.ch_out_aspp2,
            rates=(cfg.aspp2_dltn_lvl2, cfg.aspp2_dltn_lvl3, cfg.aspp2_dltn_lvl4))

        feat_skip_ch_dict = {
            1: ch_out_encoder_1x,
            2: ch_out_encoder_2x,
            4: ch_out_encoder_4x
        }

        dec1_bottleneck_ch_dict = {
            1:  cfg.dcdr1_btnek_ch_1x,
            2:  cfg.dcdr1_btnek_ch_2x,
            4:  cfg.ch_out_aspp1
        }

        dec1_skip_conv_ch_dict = {
            1:  cfg.dcdr1_skip_conv_ch_1x,
            2:  cfg.dcdr1_skip_conv_ch_2x,
            4:  cfg.dcdr1_skip_conv_ch_4x
        }

        self.decoder1 = DecoderDeeplabV3p_Unet(
            cfg.enable_decoder, dec1_bottleneck_ch_dict, feat_skip_ch_dict, dec1_skip_conv_ch_dict, outputs_desc['semseg'])

        dec2_bottleneck_ch_dict = {
            1:  cfg.dcdr2_btnek_ch_1x,
            2:  cfg.dcdr2_btnek_ch_2x,
            4:  cfg.ch_out_aspp2
        }

        dec2_skip_conv_ch_dict = {
            1:  cfg.dcdr2_skip_conv_ch_1x,
            2:  cfg.dcdr2_skip_conv_ch_2x,
            4:  cfg.dcdr2_skip_conv_ch_4x
        }

        self.decoder2 = DecoderDeeplabV3p_Unet(
            cfg.enable_decoder, dec2_bottleneck_ch_dict, feat_skip_ch_dict, dec2_skip_conv_ch_dict, outputs_desc['depth'])

        self.self_attention1 = SelfAttention(
             cfg.dcdr2_skip_conv_ch_1x+cfg.dcdr2_btnek_ch_1x, cfg.dcdr1_skip_conv_ch_1x+cfg.dcdr1_btnek_ch_1x)

        self.self_attention2 = SelfAttention(
             cfg.dcdr1_skip_conv_ch_1x+cfg.dcdr1_btnek_ch_1x, cfg.dcdr2_skip_conv_ch_1x+cfg.dcdr2_btnek_ch_1x)

        self.decoder3 = DecoderDeeplabV3pDist(
             cfg.enable_decoder, cfg.dcdr1_skip_conv_ch_1x+cfg.dcdr1_btnek_ch_1x, cfg.sa2_ch_out, outputs_desc['semseg'])

        self.decoder4 = DecoderDeeplabV3pDist(
             cfg.enable_decoder, cfg.dcdr2_skip_conv_ch_1x+cfg.dcdr2_btnek_ch_1x, cfg.sa1_ch_out, outputs_desc['depth'])

        
    def forward(self, x):
        input_resolution = (x.shape[2], x.shape[3])

        encdr_ftrs = self.encoder(x)

        lowest_scale = max(encdr_ftrs.keys())

        encdr_ftrs_lwst = encdr_ftrs[lowest_scale]

        encdr_ftrs_1 = encdr_ftrs[1]
        encdr_ftrs_2 = encdr_ftrs[2]
        encdr_ftrs_4 = encdr_ftrs[4]

        ## FIRST HALF ##
        # SEMSEG BRANCH 1ST HALF
        smsg_aspp1_ftrs = self.aspp1(encdr_ftrs_lwst)

        smsg_dcdr1_prdctns, smsg_dcdr1_feats = self.decoder1(
            smsg_aspp1_ftrs, encdr_ftrs_4, encdr_ftrs_2, encdr_ftrs_1)

        # DEPTH BRANCH 1ST HALF
        dpth_aspp2_ftrs = self.aspp2(encdr_ftrs_lwst)

        dpth_dcdr2_prdctns, dpth_dcdr2_feats = self.decoder2(
            dpth_aspp2_ftrs, encdr_ftrs_4, encdr_ftrs_2, encdr_ftrs_1)

        # CONCATENATE -> Output
        prdctn1 = torch.cat(
            (smsg_dcdr1_prdctns, dpth_dcdr2_prdctns), axis=1)

        # SECOND HALF ## (Distillation)
        # SEMSEG BRANCH 2ND HALF
        ftrs_sa2 = self.self_attention2(dpth_dcdr2_feats)

        smsg_dcdr3_prdctns, _ = self.decoder3(smsg_dcdr1_feats, ftrs_sa2)

        # DEPTH BRANCH 2ND HALF
        ftrs_sa1 = self.self_attention1(smsg_dcdr1_feats)

        dpth_dcdr4_prdctns, _ = self.decoder4(dpth_dcdr2_feats, ftrs_sa1)

        # CONCATENATE
        prdctn2 = torch.cat(
            (smsg_dcdr3_prdctns, dpth_dcdr4_prdctns), axis=1)

        ####

        out = {}
        offset = 0

        for task, num_ch in self.outputs_desc.items():
            out[task] = [prdctn1[:, offset:offset+num_ch, :, :],
                         prdctn2[:, offset:offset+num_ch, :, :]]
            offset += num_ch

        return out
