import argparse
import os
import json

from mtl.datasets.definitions import SPLIT_VALID, SPLIT_TEST


def expandpath(path):
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def command_line_parser():
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    
    #### SHARED ARCHITECTURE SPECIAL PARAMS #####
    '''
    ## ENCODER ##
    parser.add_argument(
        '--use_resnet34_weights', type=str2bool, default=True, help='Use resnet34 weights')
    parser.add_argument(
        '--encdr_dltn_one', type=str2bool, default=False, help='Set dilation in layer2 of the encoder')
    parser.add_argument(
        '--encdr_dltn_two', type=str2bool, default=False, help='Set dilation in layer3 of the encoder')
    parser.add_argument(
        '--encdr_dltn_three', type=str2bool, default=True, help='Set dilation in layer4 of the encoder')

    
    ## ASPP ##
    parser.add_argument(
        '--enable_aspp', type=str2bool, default=True, help='Enable costum ASPP module')
    parser.add_argument(
        '--pyr_ch_shared', type=int, default=256, help='Output channel number of the pyramid convolutions')
    parser.add_argument(
        '--shared_aspp_dltn_lvl2', type=int, default=3, help='Set dilation in level 2 of the aspp pyramid')
    parser.add_argument(
        '--shared_aspp_dltn_lvl3', type=int, default=6, help='Set dilation in level 3 of the aspp pyramid')
    parser.add_argument(
        '--shared_aspp_dltn_lvl4', type=int, default=9, help='Set dilation in level 4 of the aspp pyramid')
    parser.add_argument(
        '--ch_out_shared_aspp', type=int, default=256, help='Output channel number of the aspp module')

    ## DECODER ##
    parser.add_argument(
        '--enable_decoder', type=str2bool, default=True, help='Enable costum decoder module')
    parser.add_argument(
        '--ch_out_skip_conv_shared', type=int, default=64, help='Output channel number of skip connection convolution')
    #### SHARED ARCHITECTURE END ####
    # '''

    '''
    #### BRANCHED ARCHITECTURE START ####
    ## ENCODER ##
    parser.add_argument(
        '--use_resnet34_weights', type=str2bool, default=True, help='Use resnet34 weights')
    parser.add_argument(
        '--encdr_dltn_one', type=str2bool, default=False, help='Set dilation in layer2 of the encoder')
    parser.add_argument(
        '--encdr_dltn_two', type=str2bool, default=False, help='Set dilation in layer3 of the encoder')
    parser.add_argument(
        '--encdr_dltn_three', type=str2bool, default=True, help='Set dilation in layer4 of the encoder')

    ## ASPP ##
    parser.add_argument(
        '--enable_aspp', type=str2bool, default=True, help='Enable costum ASPP module')

    # SEMSEG
    parser.add_argument(
        '--pyr_ch_semseg_aspp', type=int, default=256, help='Output channel number of the pyramid convolutions in semseg aspp')
    parser.add_argument(
        '--semseg_aspp_dltn_lvl2', type=int, default=3, help='Set dilation in level 2 of the semseg aspp pyramid')
    parser.add_argument(
        '--semseg_aspp_dltn_lvl3', type=int, default=6, help='Set dilation in level 3 of the semseg aspp pyramid')
    parser.add_argument(
        '--semseg_aspp_dltn_lvl4', type=int, default=9, help='Set dilation in level 4 of the semseg aspp pyramid')
    parser.add_argument(
        '--ch_out_semseg_aspp', type=int, default=256, help='Output channel number of the semseg aspp module')

    # DEPTH
    parser.add_argument(
        '--pyr_ch_depth_aspp', type=int, default=256, help='Output channel number of the pyramid convolutions in depth aspp')
    parser.add_argument(
        '--depth_aspp_dltn_lvl2', type=int, default=3, help='Set dilation in level 2 of the depth aspp pyramid')
    parser.add_argument(
        '--depth_aspp_dltn_lvl3', type=int, default=6, help='Set dilation in level 3 of the depth aspp pyramid')
    parser.add_argument(
        '--depth_aspp_dltn_lvl4', type=int, default=9, help='Set dilation in level 4 of the depth aspp pyramid')
    parser.add_argument(
        '--ch_out_depth_aspp', type=int, default=256, help='Output channel number of the depth aspp module')

    ## DECODER ##
    parser.add_argument(
        '--enable_decoder', type=str2bool, default=True, help='Enable costum decoder module')

    # SEMSEG
    parser.add_argument(
        '--ch_out_skip_conv_semseg', type=int, default=64, help='Output channel number of skip connection convolution in semseg decoder')

    # DEPTH
    parser.add_argument(
        '--ch_out_skip_conv_depth', type=int, default=64, help='Output channel number of skip connection convolution in depth decoder')

    ### BRANCHED ARCHITECTURE END ###
    # '''

    '''
    ### TASK DISTILLATION ARCHITECTURE START ###
    ## ENCODER ##
    parser.add_argument(
        '--use_resnet34_weights', type=str2bool, default=True, help='Use resnet34 weights')
    parser.add_argument(
        '--encdr_dltn_one', type=str2bool, default=False, help='Set dilation in layer2 of the encoder')
    parser.add_argument(
        '--encdr_dltn_two', type=str2bool, default=False, help='Set dilation in layer3 of the encoder')
    parser.add_argument(
        '--encdr_dltn_three', type=str2bool, default=True, help='Set dilation in layer4 of the encoder')

    ## ASPP ##
    parser.add_argument(
        '--enable_aspp', type=str2bool, default=True, help='Enable costum ASPP module')

    # 1
    parser.add_argument(
        '--pyr_ch_aspp1', type=int, default=256, help='Output channel number of the pyramid convolutions in semseg aspp')
    parser.add_argument(
        '--aspp1_dltn_lvl2', type=int, default=3, help='Set dilation in level 2 of the semseg aspp pyramid')
    parser.add_argument(
        '--aspp1_dltn_lvl3', type=int, default=6, help='Set dilation in level 3 of the semseg aspp pyramid')
    parser.add_argument(
        '--aspp1_dltn_lvl4', type=int, default=9, help='Set dilation in level 4 of the semseg aspp pyramid')
    parser.add_argument(
        '--ch_out_aspp1', type=int, default=256, help='Output channel number of the semseg aspp module')

    # 2
    parser.add_argument(
        '--pyr_ch_aspp2', type=int, default=256, help='Output channel number of the pyramid convolutions in depth aspp')
    parser.add_argument(
        '--aspp2_dltn_lvl2', type=int, default=3, help='Set dilation in level 2 of the depth aspp pyramid')
    parser.add_argument(
        '--aspp2_dltn_lvl3', type=int, default=6, help='Set dilation in level 3 of the depth aspp pyramid')
    parser.add_argument(
        '--aspp2_dltn_lvl4', type=int, default=9, help='Set dilation in level 4 of the depth aspp pyramid')
    parser.add_argument(
        '--ch_out_aspp2', type=int, default=256, help='Output channel number of the depth aspp module')

    ## SELF ATTENTION ##
    # 1 (UPPER ONE)
    parser.add_argument(
        '--sa1_ch_out', type=int, default=1, help='Output channel number SA1 module for depth')

    # 2 (BOTTOM ONE)
    parser.add_argument(
        '--sa2_ch_out', type=int, default=19, help='Output channel number SA2 module for semseg')

    ## DECODER ##
    parser.add_argument(
        '--enable_decoder', type=str2bool, default=True, help='Enable costum decoder module')

    # 1
    parser.add_argument(
        '--ch_out_skip_conv_dcdr1', type=int, default=64, help='Output channel number of skip connection convolution in decoder1')

    # 2
    parser.add_argument(
        '--ch_out_skip_conv_dcdr2', type=int, default=64, help='Output channel number of skip connection convolution in decoder2')

    #### TASK DISTILLATION PARAMETERS ####
    # '''

    #'''
    ### TASK DISTILLATION + UNET ARCHITECTURE START ###
    ## ENCODER ##
    parser.add_argument(
        '--use_resnet34_weights', type=str2bool, default=True, help='Use resnet34 weights')
    parser.add_argument(
        '--encdr_dltn_one', type=str2bool, default=False, help='Set dilation in layer2 of the encoder')
    parser.add_argument(
        '--encdr_dltn_two', type=str2bool, default=True, help='Set dilation in layer3 of the encoder')
    parser.add_argument(
        '--encdr_dltn_three', type=str2bool, default=True, help='Set dilation in layer4 of the encoder')

    ## ASPP ##
    parser.add_argument(
        '--enable_aspp', type=str2bool, default=True, help='Enable costum ASPP module')

    # 1
    parser.add_argument(
        '--pyr_ch_aspp1', type=int, default=256, help='Output channel number of the pyramid convolutions in semseg aspp')
    parser.add_argument(
        '--aspp1_dltn_lvl2', type=int, default=3, help='Set dilation in level 2 of the semseg aspp pyramid')
    parser.add_argument(
        '--aspp1_dltn_lvl3', type=int, default=6, help='Set dilation in level 3 of the semseg aspp pyramid')
    parser.add_argument(
        '--aspp1_dltn_lvl4', type=int, default=9, help='Set dilation in level 4 of the semseg aspp pyramid')
    parser.add_argument(
        '--ch_out_aspp1', type=int, default=256, help='Output channel number of the semseg aspp module')

    # 2
    parser.add_argument(
        '--pyr_ch_aspp2', type=int, default=256, help='Output channel number of the pyramid convolutions in depth aspp')
    parser.add_argument(
        '--aspp2_dltn_lvl2', type=int, default=3, help='Set dilation in level 2 of the depth aspp pyramid')
    parser.add_argument(
        '--aspp2_dltn_lvl3', type=int, default=6, help='Set dilation in level 3 of the depth aspp pyramid')
    parser.add_argument(
        '--aspp2_dltn_lvl4', type=int, default=9, help='Set dilation in level 4 of the depth aspp pyramid')
    parser.add_argument(
        '--ch_out_aspp2', type=int, default=256, help='Output channel number of the depth aspp module')

    ## SELF ATTENTION ##
    # 1 (UPPER ONE)
    parser.add_argument(
        '--sa1_ch_out', type=int, default=1, help='Output channel number SA1 module for depth')

    # 2 (BOTTOM ONE)
    parser.add_argument(
        '--sa2_ch_out', type=int, default=19, help='Output channel number SA2 module for semseg')

    ## DECODER ##
    parser.add_argument(
        '--enable_decoder', type=str2bool, default=True, help='Enable costum decoder module')

    # 1
    parser.add_argument(
        '--dcdr1_skip_conv_ch_4x', type=int, default=64, help='Output channel number of skip connection 4x convolution in decoder1')
    parser.add_argument(
        '--dcdr1_skip_conv_ch_2x', type=int, default=16, help='Output channel number of skip connection 2x convolution in decoder1')
    parser.add_argument(
        '--dcdr1_skip_conv_ch_1x', type=int, default=4, help='Output channel number of skip connection 1x convolution in decoder1')

    parser.add_argument(
        '--dcdr1_btnek_ch_2x', type=int, default=64, help='Output channel number of intermediate bottleneck at level 2x in decoder1')
    parser.add_argument(
        '--dcdr1_btnek_ch_1x', type=int, default=16, help='Output channel number of intermediate bottleneck at level 1x in decoder1')

    # 2
    parser.add_argument(
        '--dcdr2_skip_conv_ch_4x', type=int, default=64, help='Output channel number of skip connection 4x convolution in decoder2')
    parser.add_argument(
        '--dcdr2_skip_conv_ch_2x', type=int, default=16, help='Output channel number of skip connection 2x convolution in decoder2')
    parser.add_argument(
        '--dcdr2_skip_conv_ch_1x', type=int, default=4, help='Output channel number of skip connection 1x convolution in decoder2')

    parser.add_argument(
        '--dcdr2_btnek_ch_2x', type=int, default=64, help='Output channel number of intermediate bottleneck at level 2x in decoder2')
    parser.add_argument(
        '--dcdr2_btnek_ch_1x', type=int, default=16, help='Output channel number of intermediate bottleneck at level 1x in decoder2')

    # EXTRA PARAMETERS FOR PAP ARCHITECTURE

    parser.add_argument(
        '--pap_diffusion_iterations', type=int, default=10, help='Insert nice descrpition')

    parser.add_argument(
        '--pap_diffusion_weight', type=float, default=0.05, help='Insert nice descrpition')

    ### TASK DISTILLATION + UNET END ###
    # '''

    #### GENERAL PARAMETERS ####
    parser.add_argument(
        '--log_dir', type=expandpath, required=True, help='Place for artifacts and logs')
    parser.add_argument(
        '--dataset_root', type=expandpath, required=True, help='Path to dataset')

    parser.add_argument(
        '--prepare_submission', type=str2bool, default=False,
        help='Run best model on RGB test data, pack the archive with predictions for the grader')

    parser.add_argument(
        '--num_epochs', type=int, default=16, help='Number of training epochs')
    parser.add_argument(
        '--batch_size', type=int, default=1, help='Number of samples in a batch for training')
    parser.add_argument(
        '--batch_size_validation', type=int, default=1, help='Number of samples in a batch for validation')
    parser.add_argument(
        '--aug_input_crop_size', type=int, default=256, help='Training crop size')
    parser.add_argument(
        '--aug_geom_scale_min', type=float, default=1.0, help='Augmentation: lower bound of scale')
    parser.add_argument(
        '--aug_geom_scale_max', type=float, default=1.0, help='Augmentation: upper bound of scale')
    parser.add_argument(
        '--aug_geom_tilt_max_deg', type=float, default=0.0, help='Augmentation: maximum rotation degree')
    parser.add_argument(
        '--aug_geom_wiggle_max_ratio', type=float, default=0.0,
        help='Augmentation: perspective warping level between 0 and 1')
    parser.add_argument(
        '--aug_geom_reflect', type=str2bool, default=False, help='Augmentation: Random horizontal flips')

    parser.add_argument(
        '--optimizer', type=str, default='adam', choices=['sgd', 'adam'], help='Type of optimizer')
    parser.add_argument(
        '--optimizer_lr', type=float, default=0.0001, help='Learning rate at start of training')
    parser.add_argument(
        '--optimizer_momentum', type=float, default=0.9, help='Optimizer momentum')
    parser.add_argument(
        '--optimizer_weight_decay', type=float, default=0.0001, help='Optimizer weight decay')

    parser.add_argument(
        '--lr_scheduler', type=str, default='poly', choices=['poly'], help='Type of learning rate scheduler')
    parser.add_argument(
        '--lr_scheduler_power', type=float, default=0.9, help='Poly learning rate power')

    parser.add_argument(
        '--dataset', type=str, default='miniscapes', choices=['miniscapes'], help='Dataset name')

    parser.add_argument(
        '--model_name', type=str, default='deeplabv3ppap_unet', choices=['deeplabv3p', 'deeplabv3pbrnchd', 'deeplabv3ptaskdist', 'deeplabv3ptaskdist_unet', 'deeplabv3ppap_unet'], help='CNN architecture')
    parser.add_argument(
        '--model_encoder_name', type=str, default='resnet34', choices=['resnet34'], help='CNN architecture encoder')

    parser.add_argument(
        '--loss_weight_semseg', type=float, default=0.5, help='Weight of semantic segmentation loss')
    parser.add_argument(
        '--loss_weight_depth', type=float, default=0.5, help='Weight of depth estimation loss')

    parser.add_argument(
        '--workers', type=int, default=16, help='Number of worker threads fetching training data')
    parser.add_argument(
        '--workers_validation', type=int, default=4, help='Number of worker threads fetching validation data')

    parser.add_argument(
        '--num_steps_visualization_first', type=int, default=100, help='Visualization: first time step')
    parser.add_argument(
        '--num_steps_visualization_interval', type=int, default=1000, help='Visualization: interval in steps')
    parser.add_argument(
        '--visualize_num_samples_in_batch', type=int, default=8, help='Visualization: max number of samples in batch')
    parser.add_argument(
        '--observe_train_ids', type=json.loads, default='[0,100]', help='Visualization: train IDs')
    parser.add_argument(
        '--observe_valid_ids', type=json.loads, default='[0,100]', help='Visualization: validation IDs')
    parser.add_argument(
        '--tensorboard_img_grid_width', type=int, default=8, help='Visualization: number of samples per row')
    parser.add_argument(
        '--tensorboard_daemon_start', type=str2bool, default=False,
        help='Visualization: run tensorboard alongside training process')
    parser.add_argument(
        '--tensorboard_daemon_port', type=int, default=10000,
        help='Visualization: port to use with tensorboard daemon')
    parser.add_argument(
        '--ngrok_daemon_start', type=str2bool, default=False,
        help='Visualization: run ngrok alongside training process')
    parser.add_argument(
        '--ngrok_auth_token', type=str, default='', help='ngrok authtoken (visit ngrok.com for more information)')

    parser.add_argument(
        '--log_to_console', type=str2bool, default=True, help='Disables progress bar')

    cfg = parser.parse_args()

    #print(json.dumps(cfg.__dict__, indent=4, sort_keys=True))

    return cfg


EXPERIMENT_INVARIANT_KEYS = (
    'log_dir',
    'dataset_root',
    'prepare_submission',
    'batch_size_validation',
    'workers',
    'workers_validation',
    'num_steps_visualization_first',
    'num_steps_visualization_interval',
    'visualize_num_samples_in_batch',
    'observe_train_ids',
    'observe_valid_ids',
    'tensorboard_img_grid_width',
    'tensorboard_daemon_start',
    'tensorboard_daemon_port',
    'ngrok_daemon_start',
    'ngrok_auth_token',
    'log_to_console',
)
