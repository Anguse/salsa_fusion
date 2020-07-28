## _salsa_fusion_:

By Harald Lilja

This repository contains an implementation of an encoder-decoder network for semantic scene segmentation based on 2D-LiDAR and depth data. The network employs sensor fusion on a network level with feature maps from both modalities being concatenated during upsampling in order to learn correlated information from the data inputs. Applying this strategy yields improvement in accuracy while acting independantly from textures in the scene.

<p align="center">
    <img src="./data/network.png" class="center" width=600>
</p>
<p align="center">
    <em>The network design is based on two modified instances of [SalsaNet](https://gitlab.com/aksoyeren/salsanet) with three shared representation layers in the decoder.</em>
</p>

<p align="center">
    <img src="./data/depth.gif" height="200" class="left">
    <img src="./data/laser.gif" height="200"  class="center">
    <img src="./data/preds.gif" height="200"  class="right">
</p>

<p align="center">
    <img src="./data/rgb.gif" height="200" class="left">
</p>
<p align="center">
    <em>Predictions from an unseen circuit after training on a dataset of 1080 datapoints from a different circuit.</em>
</p>
