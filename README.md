# CenterNet-for-Autonomous-Vehicle

* Took 3D visualization code from https://www.kaggle.com/zstusnoopy/visualize-the-location-and-3d-bounding-box-of-car
* CenterNet paper https://arxiv.org/pdf/1904.07850.pdf
* CenterNet repository https://github.com/xingyizhou/CenterNet

# What is this competition about?
1. You are given the images taken from the roof of a car
    * ~4k training images
    * Always the same car and the same camera
2. You are asked to detect other cars on that image
    * There can be many cars
    * You need to predict their positions
![](https://i.ibb.co/7RJ2Wbs/results-33-2.png)

## What is in this notebook?
* Data distributions: 1D, 2D and 3D
* Functions to transform between camera coordinates and road coordinates
* Simple CenterNet baseline

## CenterNet
This architecture predicts centers of objects as a heatmap.  
It predicts sizes of the boxes as a regression task.  
![](https://github.com/xingyizhou/CenterNet/raw/master/readme/fig2.png)

It is also used for pose estimation:
![](https://raw.githubusercontent.com/xingyizhou/CenterNet/master/readme/pose3.png)
*(images from the [original repository](https://github.com/xingyizhou/CenterNet))*  
Coordinates of human joints are also predicted using regression.  

I use this idea to predict `x, y, z` coordinates of the vehicle and also `yaw, pitch_cos, pitch_sin, roll` angles.  
For `pitch` I predict sin and cos, because, as we will see, this angle can be both near 0 and near 3.14.  
These 7 parameters are my regression target variables instead of `shift_x, shift_y, size_x, size_y`.  



# Model Construction and Implementation
## 1. Model Design

In the domain of image recognition for autonomous vehicles,
we have opted to employ EfficientNet as the encoder and UNet as the decoder for our model. 
This model architecture maximizes the robust feature extraction capability of EfficientNet and leverages the advantages of UNet in feature reconstruction and detail recovery.

![圖片](https://github.com/YeeHaoSu/CenterNet-for-Autonomous-Vehicle/assets/90921571/9bc7d493-2485-4ce7-867c-24e560535760)

## 2. Implementation of Encoder Construction

Firstly, let's delve into the implementation process of EfficientNet as the encoder. 
EfficientNet is a deep learning model created using automated machine learning (AutoML) and compound scaling. 
It achieves optimal performance under limited computational resources by simultaneously scaling the network's depth, width, and resolution. In implementing EfficientNet, multiple depthwise separable convolution layers are constructed, incorporating batch normalization and Swish activation functions. 
These depthwise separable convolution layers efficiently extract features from images, significantly reducing computational and parameter overhead compared to traditional convolution layers. The advantages of using EfficientNet as the model encoder are evident in the following aspects:

   - Performance Optimization: EfficientNet is designed to achieve efficient model scaling through AutoML and compound scaling. By scaling the network's depth, width, and resolution simultaneously, EfficientNet enhances performance while maintaining model complexity within a manageable range.

   - Parameter Optimization: Compared to other deep learning models, EfficientNet has fewer parameters. This translates to higher training and inference speeds under equivalent computational resources. The reduced parameter count also contributes to easier training, mitigating the risk of overfitting.

   - Accuracy Improvement: EfficientNet consistently demonstrates high accuracy across various image recognition tasks. Particularly notable is its performance on the ImageNet dataset, surpassing traditional models like ResNet with comparable precision at lower computational costs.

   - Model Scalability: EfficientNet provides a systematic approach to scaling deep learning models, applicable to diverse tasks and devices. Accordingly, different scales of EfficientNet models, such as EfficientNet-B0 to EfficientNet-B7, can be chosen based on practical requirements. For this project, we chose EfficientNet-B0 considering hardware limitations.

![圖片](https://github.com/YeeHaoSu/CenterNet-for-Autonomous-Vehicle/assets/90921571/b5145d13-7c91-40e2-a5b7-5ef146c26eff)

## 3. Implementation of Decoder Construction

Next, let's examine the implementation process of UNet as the decoder.
UNet is a deep learning model with a U-shaped structure, comprising a contracting path (encoder) and an expanding path (decoder). 
In this context, we utilize only the expanding path of UNet as our decoder.
During the implementation of UNet, multiple upsampling layers are constructed, incorporating batch normalization and ReLU activation functions. 
These upsampling layers restore feature maps extracted by EfficientNet to the original image size, achieving feature reconstruction and detail recovery.
The selection of UNet as the decoder offers the following advantages:

   - Unique Structure: UNet's distinctive U-shaped structure includes a contracting path (Encoder) for feature extraction and an expanding path (Decoder) for upsampling and reconstruction. This dual-path design contributes to effective feature extraction and image segmentation.

   - Powerful Capabilities: Due to its U-shaped structure, UNet exhibits excellent performance in image segmentation tasks, especially in medical image segmentation where it has demonstrated superior capabilities.

   - Detail Recovery: The expanding path (Decoder) of UNet, through upsampling and skip connections, facilitates the restoration of images to their original resolution. This allows for feature reconstruction and detail recovery.

   - Model Training: UNet has relatively low data requirements for training, enabling the training of a good model even with limited data. This attribute proves beneficial in scenarios with smaller datasets.

## 4. Coordinate Tracking

Since UNet's original design is intended for segmentation tasks and does not readily accommodate tasks like coordinate tracking, we address this limitation by annotating absolute coordinates on images before training. 
These coordinates are then fused into the images, allowing the model to output vehicle coordinates and center positions after training and post-processing.

## 5. Tensor Computation

In the process of tensor computation, we start with the feature maps output by EfficientNet, represented as a four-dimensional tensor with the shape (batch size, feature map height, feature map width, number of channels).
Subsequently, we pass this tensor through various layers of UNet to transform the pixel values of the feature maps into the output images of upsampling layers.
Throughout this process, numerous tensor multiplication and addition operations, as well as some convolution and interpolation operations, are performed.

In summary, for autonomous vehicle image recognition, we combine EfficientNet as the encoder and UNet as the decoder to achieve efficient feature extraction and detail reconstruction.
Tensor computation plays a crucial role in the model implementation, enabling effective feature extraction and transformation on high-dimensional data for image recognition tasks.

![圖片](https://github.com/YeeHaoSu/CenterNet-for-Autonomous-Vehicle/assets/90921571/f39d8bf4-d2d0-4c54-b746-fc04a66435cd)

## 6. Convergence Trends

The left graph illustrates training loss, while the right graph represents validation loss. 
The trends appear relatively aligned, affirming the model's flexibility and resilience against overfitting. 
Additionally, the loss converges, indicating a balance between model flexibility and convergence on this challenging dataset.

![圖片](https://github.com/YeeHaoSu/CenterNet-for-Autonomous-Vehicle/assets/90921571/6ceda947-0566-42ca-bee7-cdc1463674a2) ![圖片](https://github.com/YeeHaoSu/CenterNet-for-Autonomous-Vehicle/assets/90921571/25e77446-2917-4859-a495-aa74d5b707b3)

# Model Evaluation and Results Presentation

## 1. Results Display
First Outcome, as depicted below, the results showcase the segmentation predictions, where regions with confidence scores surpassing a certain threshold are marked as answers.

* Image

![圖片](https://github.com/YeeHaoSu/CenterNet-for-Autonomous-Vehicle/assets/90921571/bd745c87-312c-42ea-90fb-388aaf16c22d)

* Ground Truth

![圖片](https://github.com/YeeHaoSu/CenterNet-for-Autonomous-Vehicle/assets/90921571/e9be7b9d-be52-4897-a6cc-49245fe5cd47)

* Model prediction

![圖片](https://github.com/YeeHaoSu/CenterNet-for-Autonomous-Vehicle/assets/90921571/b3184b3d-1b0f-4ffe-a709-552e554a0687)

Second Output Type. The output is presented based on the extracted coordinates and the vehicle's center point, facilitating data retrieval and presentation.

![圖片](https://github.com/YeeHaoSu/CenterNet-for-Autonomous-Vehicle/assets/90921571/8bf48d41-61dc-4809-8210-9e502202d2c4)

![圖片](https://github.com/YeeHaoSu/CenterNet-for-Autonomous-Vehicle/assets/90921571/d222ae04-e417-4605-aeca-2499d8c091af)

# Reference

[1] Peking University/Baidu - Autonomous Driving

https://www.kaggle.com/competitions/pku-autonomous-driving/overview

[2] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." in Medical Image Computing and Computer-Assisted Intervention–(MICCAI), 2015

[3] Tan, Mingxing, and Quoc Le. "Efficientnet: Rethinking model scaling for convolutional neural networks." International conference on machine learning. PMLR, 2019.

