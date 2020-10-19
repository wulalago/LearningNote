# LearningNote
> A record of my course for deep learning

## Programming Framework
> Contents of the programming framework for machine learning and medical image computing.  

### [PyTorch](https://pytorch.org/)  
 An open source machine learning framework that accelerates the path from research prototyping to production deployment.

### PyTorch Extension Library
- [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric)
  > A Geometric Deep Learning Extension Library for PyTorch
- [Kaolin](https://github.com/NVIDIA/GameWorkskaolin)
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
  > Pytorch Framework for Doing Deep Learning on Point Clouds.

  
### Python Extension Library
- [ANTsPy](https://github.com/ANTsX/ANTsPy)
  > Advanced Normalization Tools in Python
- [CuPy](https://github.com/cupy/cupy)
  > NumPy-like API Accelerated with CUDA
- [MedPy](https://github.com/loli/medpy)  
  > Medical Image Processing in Python

## Papers
> Paper list for deep learning in computer vision and medical image computing. 

### Summary, Survey and Review
- **Deep Learning**  *Nature* (2015) [[Paper]](https://www.nature.com/articles/nature14539)
- **Bag of Tricks for Image Classification with Convolutional Neural Networks**  *CVPR* (2019) [[Paper]](https://arxiv.org/pdf/1812.01187v2.pdf)
- **Deep Learning for Cardiac Image Segmentation: A Review**  *Frontiers in Cardiovascular Medicine* (2020) [[Paper]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7066212/)
  
### Contrast and Consistent Learning
- **Robust Learning Through Cross-Task Consistency**  *CVPR* (2020) [[Paper]](https://consistency.epfl.ch/Cross_Task_Consistency_CVPR2020.pdf) [[Code]](https://github.com/EPFL-VILAB/XTConsistency)  
- **A Simple Framework for Contrastive Learning of Visual Representations**  *ArXiv* (2020) [[Paper]](https://arxiv.org/pdf/2002.05709.pdf) [[Code]](https://github.com/google-research/simclr)

### Recognition
- **Focus Longer to See Better: Recursively Refined Attention for Fine-Grained Image Classification**  *CVPR* (2020) [[Paper]](https://arxiv.org/pdf/2005.10979.pdf) [[Code]](https://github.com/TAMU-VITA/Focus-Longer-to-See-Better)
- **MixConv: Mixed Depthwise Convolutional Kernels**  *BMVC* (2019). [[Paper]](https://arxiv.org/pdf/1907.09595.pdf) [[Code]](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet)    
  
### Detection
#### Anomaly Detection:
- **GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training**  *ACCV* (2018) [[Paper]](https://arxiv.org/pdf/1805.06725.pdf) [[Code]](https://github.com/samet-akcay/ganomaly)

### Registration
- **An Unsupervised Learning Model for Deformable Medical Image Registration**  *CVPR* (2018) [[Paper]](https://arxiv.org/abs/1802.02604) [[Code]](https://github.com/voxelmorph/voxelmorph)
- **JSSR: A Joint Synthesis, Segmentation, and Registration System for 3D Multi-Modal Image Alignment of Large-scale Pathological CT Scans**  *ArXiv* (2020) [[Paper]](https://arxiv.org/pdf/2005.12209.pdf)
- **Adversarial Uni- and Multi-modal Stream Networks for Multimodal Image Registration**  *MICCAI* (2020) [[Paper]](https://arxiv.org/pdf/2007.02790.pdf) 
- **Unsupervised Multi-Modal Image Registration via Geometry Preserving Image-to-Image Translation**  *CVPR* (2020) [[Paper]](https://arxiv.org/abs/2003.08073) 


### Segmentation
#### High Resolution and Efficient Model
- **ICNet for Real-Time Semantic Segmentation on High-Resolution Images**  *ECCV* (2018) [[Paper]](https://arxiv.org/abs/1704.08545) [[Code]](https://github.com/hszhao/ICNet)

- **Collaborative Global-Local Networks for Memory-Efficient Segmentation of Ultra-High Resolution Images**  *ICCV* (2019) [[Paper]](https://arxiv.org/pdf/1905.06368.pdf) [[Code]](https://github.com/TAMU-VITA/GLNet)

#### Semi, Weakly, and One-shot
- **LT-Net: Label Transfer by Learning Reversible Voxel-wise Correspondence for One-shot Medical Image Segmentation**  *CVPR* (2020) [[Paper]](https://arxiv.org/abs/2003.07072)

- **Self-supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentation**  *CVPR* (2020) [[Paper]](https://arxiv.org/abs/2004.04581) [[Code]](https://github.com/YudeWang/SEAM)
 
#### Interaction
- **Iteratively-Refined Interactive 3D Medical Image Segmentation with Multi-Agent Reinforcement Learning**  
  *CVPR* (2020) [[Paper]](https://arxiv.org/abs/1911.10334)  

#### Uncertainty and Attention
- **Cars Can’t Fly up in the Sky: Improving Urban-Scene Segmentation via Height-driven Attention Networks**  *CVPR* (2020) [[Paper]](https://arxiv.org/pdf/2003.05128.pdf) [[Code]](https://github.com/shachoi/HANet)

- **Attention U-Net: Learning Where to Look for the Pancreas**  *MIDL* (2018) [[Paper]](https://openreview.net/pdf?id=Skft7cijM) [[Code]](https://github.com/ozan-oktay/Attention-Gated-Networks)

#### Loss Function
- **Learning Active Contour Models for Medical Image Segmentation**  *CVPR* (2019) [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Learning_Active_Contour_Models_for_Medical_Image_Segmentation_CVPR_2019_paper.pdf) [[Code]](https://github.com/xuuuuuuchen/Active-Contour-Loss)

  
### Plane Localization
- **Automatic View Planning with Multi-scale Deep Reinforcement Learning Agents**  *MICCAI* (2018) [[Paper]](https://arxiv.org/pdf/1806.03228.pdf) [[Code]](https://git.io/vhuMZ)  
- **Ultrasound Video Summarization using Deep Reinforcement Learning**  *MICCAI* (2020) [[Paper]](https://arxiv.org/pdf/2005.09531.pdf)


### Synthesis, Simulation and Blending
- **GP-GAN: Towards Realistic High-Resolution Image Blending**  *ACMMM* (2019) [[Paper]](https://arxiv.org/pdf/1703.07195.pdf) [[Code]](https://github.com/wuhuikai/GP-GAN)  
- **Learning to Simulate Complex Scenes**  *ACMMM* (2019) [[Paper]](https://arxiv.org/pdf/2006.14611.pdf)
- **Learning to Simulate**  *ICLR* (2019) [[Paper]](http://arxiv.org/pdf/1810.02513v2.pdf)

### Other
- **Shapeshifter Networks: Cross-layer Parameter Sharing for Scalable and Effective Deep Learning**  *ArXiv* (2020) [[Paper]](https://arxiv.org/pdf/2006.10598.pdf) 


## CodeBox
> CodeBox for fast coding  

- **Metrics**
  - [segmentation](codebox/metrics/segmentation.py)
    - dice_ratio
  - [registration](codebox/metrics/registration.py)
    - negative_Jacobian

