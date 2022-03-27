The model is currently ready for testing the Task Distillation + U-Net-like Decoders architecture with L2 loss.

To test the Joint, Branched or basic Task Distillation models, please go to mtl/utils/config.py and follow the comments to comment/uncomment sections of parameters.
The PAP net can use the same paramerts as the TD+U-Net architecture.

To test the berHu loss, please go to mtl/losses/loss_regression.py, comment the L2 loss and uncomment the berHu loss.