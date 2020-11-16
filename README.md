# LearningNote
> A record of my course for deep learning

## 1. Programming Framework
> Contents of the programming framework for machine learning and medical image computing.  

### 1.1 [PyTorch](https://pytorch.org/)  
 An open source machine learning framework that accelerates the path from research prototyping to production deployment.

### 1.2 PyTorch Extension Library
- [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric)
  > A Geometric Deep Learning Extension Library for PyTorch
- [Kaolin](https://github.com/NVIDIAGameWorks/kaolin)
  > PyTorch Library for Accelerating 3D Deep Learning Research
- [Kornia](https://github.com/kornia/kornia)
  > Open Source Differentiable Computer Vision Library for PyTorch
- [Torchmeta](https://github.com/tristandeleu/pytorch-meta)
  > A Collection of Extensions and Data-loaders for Few-shot Learning & Meta-learning in PyTorch
- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
  > The Lightweight PyTorch Wrapper for ML Researchers
- [Tianshou](https://github.com/thu-ml/tianshou)
  > An Elegant, Flexible, and Superfast PyTorch Deep Reinforcement Learning Platform
- [PyTorch Points3D](https://github.com/nicolas-chaulet/torch-points3d)
  > PyTorch Framework for Doing Deep Learning on Point Clouds.
- [Pywick](https://github.com/achaiah/pywick)
  > High-Level Training Framework for PyTorch

  
### 1.3 Python Extension Library
- [ANTsPy](https://github.com/ANTsX/ANTsPy)
  > Advanced Normalization Tools in Python
- [CuPy](https://github.com/cupy/cupy)
  > NumPy-like API Accelerated with CUDA
- [MedPy](https://github.com/loli/medpy)  
  > Medical Image Processing in Python
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
  > An Image Analysis Toolkit
  
## 2. Papers
> Paper list for deep learning in computer vision and medical image computing. 

### 2.1 Summary, Survey and Review
- **Deep Learning**  *Nature* (2015) [[Paper]](https://www.nature.com/articles/nature14539)
- **A Survey on Deep Learning in Medical Image Analysis**  *Medical Image Analysis* (2017) [[Paper]](https://www.sciencedirect.com/science/article/pii/S1361841517301135)
- **Bag of Tricks for Image Classification with Convolutional Neural Networks**  *CVPR* (2019) [[Paper]](https://arxiv.org/pdf/1812.01187v2.pdf)
- **Deep Learning for Cardiac Image Segmentation: A Review**  *Frontiers in Cardiovascular Medicine* (2020) [[Paper]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7066212/)
- **Self-supervised Visual Feature Learning with Deep Neural Networks: A Survey**  *IEEE Transactions on Pattern Analysis and Machine Intelligence* (2020) [[Paper]](https://ieeexplore.ieee.org/abstract/document/9086055)
  
### 2.2 Recognition
- **ImageNet Classification with Deep Convolutional Neural Networks**  *NeurIPS* (2012) [[Paper]](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- **Very Deep Convolutional Networks for Large-Scale Image Recognition**  *ICLR* (2015) [[Paper]](https://arxiv.org/abs/1409.1556)
- **Going Deeper with Convolutions**  *CVPR* (2015) [[Paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)
- **Deep Residual Learning for Image Recognition**  *CVPR* (2016) [[Paper]](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) 
- **Identity Mappings in Deep Residual Networks**  *ECCV* (2016) [[Paper]](https://rd.springer.com/chapter/10.1007/978-3-319-46493-0_38) [[Code]](https://github.com/KaimingHe/resnet-1k-layers)
- **Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning**  *AAAI* (2017) [[Paper]](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14806/14311) 
- **Densely Connected Convolutional Networks**  *CVPR* (2017) [[Paper]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf) [[Code]](https://github.com/liuzhuang13/DenseNet)
- **Aggregated Residual Transformations for Deep Neural Networks**  *CVPR* (2017) [[Paper]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf) [[Code]](https://github.com/facebookresearch/ResNeXt)
- **Dynamic Routing Between Capsules** *NeurIPS* （2017）[[Paper]](https://arxiv.org/pdf/1710.09829.pdf)
- **Squeeze-and-Excitation Networks**  *CVPR* (2018) [[Paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf) [[Code]](https://github.com/hujie-frank/SENet)
- **MixConv: Mixed Depthwise Convolutional Kernels**  *BMVC* (2019) [[Paper]](https://arxiv.org/pdf/1907.09595.pdf) [[Code]](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet)    
- **Res2Net: A New Multi-scale Backbone Architecture**  *IEEE Transactions on Pattern Analysis and Machine Intelligence* (2019) [[Paper]](https://ieeexplore.ieee.org/abstract/document/8821313) [[Code]](https://github.com/Res2Net)

#### 2.2.1 Fine-grained
- **Focus Longer to See Better: Recursively Refined Attention for Fine-grained Image Classification**  *CVPR* (2020) [[Paper]](https://arxiv.org/pdf/2005.10979.pdf) [[Code]](https://github.com/TAMU-VITA/Focus-Longer-to-See-Better)
  
### 2.3 Segmentation
- **Fully Convolutional Networks for Semantic Segmentation**  *CVPR* (2015) [[Paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)
- **SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation**  *IEEE Transactions on Pattern Analysis and Machine Intelligence* (2017) [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7803544) 
- **U-Net: Convolutional Networks for Biomedical Image Segmentation**  *MICCAI* (2015) [[Paper]](https://rd.springer.com/chapter/10.1007/978-3-319-24574-4_28) [[Code]](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net)
- **3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation**  *MICCAI* (2016) [[Paper]](https://rd.springer.com/chapter/10.1007/978-3-319-46723-8_49)
- **V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation**  *3DV* (2016) [[Paper]](https://ieeexplore.ieee.org/abstract/document/7785132)
- **UNet++: A Nested U-Net Architecture for Medical Image Segmentation**  *DLMIA* (2018) [[Paper]](https://rd.springer.com/chapter/10.1007/978-3-030-00889-5_1)
- **DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs**   *IEEE Transactions on Pattern Analysis and Machine Intelligence* (2018) [[Paper]](https://ieeexplore.ieee.org/abstract/document/7913730)
- **Capsules for Biomedical Image Segmentation** *Medical Image Analysis* (2020) [[Paper]](https://www.sciencedirect.com/science/article/pii/S136184152030253X)

#### 2.3.1 High Resolution and Efficient Model
- **ICNet for Real-Time Semantic Segmentation on High-Resolution Images**  *ECCV* (2018) [[Paper]](https://arxiv.org/abs/1704.08545) [[Code]](https://github.com/hszhao/ICNet)
- **Collaborative Global-Local Networks for Memory-Efficient Segmentation of Ultra-High Resolution Images**  *ICCV* (2019) [[Paper]](https://arxiv.org/pdf/1905.06368.pdf) [[Code]](https://github.com/TAMU-VITA/GLNet)

#### 2.3.2 Semi, Weakly, and One-shot
- **LT-Net: Label Transfer by Learning Reversible Voxel-wise Correspondence for One-shot Medical Image Segmentation**  *CVPR* (2020) [[Paper]](https://arxiv.org/abs/2003.07072)

- **Self-supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentation**  *CVPR* (2020) [[Paper]](https://arxiv.org/abs/2004.04581) [[Code]](https://github.com/YudeWang/SEAM)
 
#### 2.3.3 Interaction
- **Iteratively-Refined Interactive 3D Medical Image Segmentation with Multi-Agent Reinforcement Learning**  
  *CVPR* (2020) [[Paper]](https://arxiv.org/abs/1911.10334)  

#### 2.3.4 Uncertainty and Attention
- **Cars Can’t Fly up in the Sky: Improving Urban-Scene Segmentation via Height-driven Attention Networks**  *CVPR* (2020) [[Paper]](https://arxiv.org/pdf/2003.05128.pdf) [[Code]](https://github.com/shachoi/HANet)
- **Attention U-Net: Learning Where to Look for the Pancreas**  *MIDL* (2018) [[Paper]](https://openreview.net/pdf?id=Skft7cijM) [[Code]](https://github.com/ozan-oktay/Attention-Gated-Networks)

#### 2.3.5 Loss Function
- **Learning Active Contour Models for Medical Image Segmentation**  *CVPR* (2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Learning_Active_Contour_Models_for_Medical_Image_Segmentation_CVPR_2019_paper.pdf) [[Code]](https://github.com/xuuuuuuchen/Active-Contour-Loss)
  
### 2.4 Detection
- [**Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation.**](https://openaccess.thecvf.com/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf)  *CVPR,* 2014. 
- [**Fast R-CNN.**](https://openaccess.thecvf.com/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf) *ICCV,* 2015.
- [**Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.**](http://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf)  *NeurIPS,* 2015.
- [**Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition.**](https://ieeexplore.ieee.org/abstract/document/7005506)  *IEEE Transactions on Pattern Analysis and Machine Intelligence,* 2015.
- [**Focal Loss for Dense Object Detection.**](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)  *ICCV* 2017.
- [**Feature Pyramid Networks for Object Detection.**](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf)  *CVPR,* 2017.
- [**Mask R-CNN.**](https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf)  *ICCV,* 2017.

### Registration
- [**An Unsupervised Learning Model for Deformable Medical Image Registration.**](https://arxiv.org/abs/1802.02604)  *CVPR,* 2018.
- [**Learning Conditional Deformable Templates with Convolutional Networks.**](https://arxiv.org/pdf/1908.02738.pdf)  *NeurIPS,* 2019. 
#### Multi-modal registration
- [**Weakly-supervised Convolutional Neural Networks for Multi-modal Image Registration.**](https://www.sciencedirect.com/science/article/pii/S1361841518301051)  *Medical Image Analysis,* 2018.
- [**JSSR: A Joint Synthesis, Segmentation, and Registration System for 3D Multi-Modal Image Alignment of Large-scale Pathological CT Scans.**](https://arxiv.org/pdf/2005.12209.pdf) *arXiv,* 2020.
- [**Adversarial Uni- and Multi-modal Stream Networks for Multimodal Image Registration.**](https://arxiv.org/pdf/2007.02790.pdf) *MICCAI,* 2020.
- [**Unsupervised Multi-Modal Image Registration via Geometry Preserving Image-to-Image Translation.**](https://arxiv.org/abs/2003.08073) *CVPR,* 2020.

### Self-supervised Learning
#### Contrast & Consistent Learning
- [**Robust Learning Through Cross-Task Consistency.**](https://consistency.epfl.ch/Cross_Task_Consistency_CVPR2020.pdf)  *CVPR,* 2020. 
- [**A Simple Framework for Contrastive Learning of Visual Representations.**](https://arxiv.org/pdf/2002.05709.pdf)  *arXiv,* 2020.

### Landmark Detection
- [**Evaluating Reinforcement Learning Agents for Anatomical Landmark Detection.**](https://www.sciencedirect.com/science/article/pii/S1361841518306121) *Medical Image Analysis,* 2019.
- [**Multi-Scale Deep Reinforcement Learning for Real-Time 3D-Landmark Detection in CT Scans.**](https://ieeexplore.ieee.org/abstract/document/8187667) *IEEE Transactions on Pattern Analysis and Machine Intelligence,* 2019.

### View Planning
#### 2D View Planning
- [**SonoNet: Real-Time Detection and Localisation of Fetal Standard Scan Planes in Freehand Ultrasound.**](https://ieeexplore.ieee.org/document/7974824) *IEEE Transctions on Medical Imaging,* 2017.
- [**Ultrasound Video Summarization using Deep Reinforcement Learning.**](https://arxiv.org/pdf/2005.09531.pdf)  *MICCAI,* 2020. 
#### 3D View Planning
- [**Standard Plane Detection in 3D Fetal Ultrasound Using an Iterative Transformation Network.**](https://rd.springer.com/chapter/10.1007/978-3-030-00928-1_45)  *MICCAI,* 2018. 
- [**Automatic View Planning with Multi-scale Deep Reinforcement Learning Agents.**](https://arxiv.org/pdf/1806.03228.pdf)  *MICCAI,* 2018. 

### Others
#### Anomaly Detection:
- [**GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training.**](https://arxiv.org/pdf/1805.06725.pdf) *ACCV,* 2018. 
### Known Operator
- [**PYRO-NN: Python Reconstruction Operators in Neural Networks.**](https://pubmed.ncbi.nlm.nih.gov/31389023/)  *Medical Physics,* 2019.

### 2.10 Synthesis, Simulation and Blending
- **GP-GAN: Towards Realistic High-Resolution Image Blending**  *ACMMM* (2019) [[Paper]](https://arxiv.org/pdf/1703.07195.pdf) [[Code]](https://github.com/wuhuikai/GP-GAN)  
- **Learning to Simulate Complex Scenes**  *ACMMM* (2019) [[Paper]](https://arxiv.org/pdf/2006.14611.pdf)
- **Learning to Simulate**  *ICLR* (2019) [[Paper]](http://arxiv.org/pdf/1810.02513v2.pdf)

#### 2.10.1 MR Simulator
- **Fast Realistic MRI Simulations Based on Generalized Multi-Pool Exchange Tissue Model**  *IEEE Transctions on Medical Imaging* (2017) [[Paper]](https://ieeexplore.ieee.org/document/7676360) [[Code]](https://github.com/leoliuf/MRiLab)
- **MRISIMUL: A GPU-Based Parallel Approach to MRI Simulations**  *IEEE Transctions on Medical Imaging* (2017) [[Paper]](https://ieeexplore.ieee.org/document/6671435)

### 2.11 Shape Learning
- **Learning to Infer Semantic Parameters for 3D Shape Editing**  *ArXiv* (2020) [[Paper]](https://arxiv.org/pdf/2011.04755.pdf)

### 2.11 Efficient Learning
- **Shapeshifter Networks: Cross-layer Parameter Sharing for Scalable and Effective Deep Learning**  *ArXiv* (2020) [[Paper]](https://arxiv.org/pdf/2006.10598.pdf) 


## 3. CodeBox
> CodeBox for fast coding  

- **Metrics**
  - [segmentation](codebox/metrics/segmentation.py)
    - dice_ratio
  - [registration](codebox/metrics/registration.py)
    - negative_Jacobian

