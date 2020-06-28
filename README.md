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
- **Deep Learning**   
  *Nature* (2015). [[Paper]](https://www.nature.com/articles/nature14539)
  > From Yann LeCun (FAIR & NYU), Yoshua Bengio (Univ. de Montréal), Geoffrey Hinton (Google & Univ. of Toronto).
- **Bag of Tricks for Image Classification with Convolutional Neural Networks**  
  *CVPR* (2019). [[Paper]](https://arxiv.org/pdf/1812.01187v2.pdf)
  > From Tong He<sup>1</sup> (AWS) and Mu Li<sup>*</sup> (AWS)
- **Deep Learning for Cardiac Image Segmentation: A Review**  
  *Frontiers in Cardiovascular Medicine* (2020). [[Paper]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7066212/)
  > From Chen Chen<sup>1</sup> (ICL) and Daniel Rueckert<sup>*</sup> (ICL)
  

### Learning
- **Robust Learning Through Cross-Task Consistency**  
  *CVPR* (2020). [[Paper]](https://consistency.epfl.ch/Cross_Task_Consistency_CVPR2020.pdf) [[Code]](https://github.com/EPFL-VILAB/XTConsistency)  
  > From Amir R. Zamir<sup>1</sup> (EPFL), Alexander Sax<sup>1</sup> (UC Berkeley) and Leonidas Guibas<sup>*</sup> (Stanford Univ.)
- **Shapeshifter Networks: Cross-layer Parameter Sharing for Scalable and Effective Deep Learning**   
  *ArXiv* (2020). [[Paper]](https://arxiv.org/pdf/2006.10598.pdf) 
  > From Bryan A. Plummer<sup>1</sup> (Boston Univ.) and Kate Saenko<sup>*</sup> (Boston Univ. & MIT-IBM Watson AI Lab) 

### Recognition
- **Focus Longer to See Better: Recursively Refined Attention for Fine-Grained Image Classification**  
  *CVPR* (2020). [[Paper]](https://arxiv.org/pdf/2005.10979.pdf) [[Code]](https://github.com/TAMU-VITA/Focus-Longer-to-See-Better)
  > From Prateek Shroff<sup>1</sup> (TAMU) and Zhangyang Wang<sup>*</sup> (TAMU)

### Registration
- **An Unsupervised Learning Model for Deformable Medical Image Registration**  
  *CVPR* (2018). [[Paper]](https://arxiv.org/abs/1802.02604) [[Code]](https://github.com/voxelmorph/voxelmorph)
  > From Balakrishnan Guha<sup>1</sup> (MIT) and Adrian V. Dalca<sup>*</sup> (MIT & MGH).
- **JSSR: A Joint Synthesis, Segmentation, and Registration System for 3D Multi-Modal Image Alignment of Large-scale Pathological CT Scans**  
  *ArXiv* (2020). [[Paper]](https://arxiv.org/pdf/2005.12209.pdf)
  > From Fengze Liu<sup>1</sup> (PAII Inc. & Johns Hopkins Univ.) and Adam P Harrison<sup>*</sup> (PAII Inc.)
  
### Segmentation
- **Learning Active Contour Models for Medical Image Segmentation**  
  *CVPR* (2019). [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Learning_Active_Contour_Models_for_Medical_Image_Segmentation_CVPR_2019_paper.pdf) [[Code]](https://github.com/xuuuuuuchen/Active-Contour-Loss)
  > From Xu Chen<sup>1</sup> (Univ. of Liverpool) and Yalin Zheng<sup>*</sup> (Univ. of Liverpool).
- **Collaborative Global-Local Networks for Memory-Efficient Segmentation of Ultra-High Resolution Images**  
  *ICCV* (2019) [[Paper]](https://arxiv.org/pdf/1905.06368.pdf) [[Code]](https://github.com/TAMU-VITA/GLNet)
  > From Wuyang Chen<sup>1</sup> (TAMU), Ziyu Jiang<sup>1</sup>  (TAMU) and Xiaoning Qian<sup>*</sup> (TAMU)
- **Cars Can’t Fly up in the Sky: Improving Urban-Scene Segmentation via Height-driven Attention Networks**  
  *CVPR* (2020). [[Paper]](https://arxiv.org/pdf/2003.05128.pdf) [[Code]](https://github.com/shachoi/HANet)
  > From Sungha Choi<sup>1</sup> (Korea Univ.) and Jaegul Choo<sup>*</sup> (KAIST).
- **LT-Net: Label Transfer by Learning Reversible Voxel-wise Correspondence for One-shot Medical Image Segmentation**  
  *CVPR* (2020). [[Paper]](https://arxiv.org/abs/2003.07072)
  > From Shuxin Wang<sup>1</sup> (Xiamen Univ. & Jarvis Lab), Shilei Cao<sup>1</sup> (Jarvis Lab), Dong Wei<sup>1</sup> (Jarvis Lab) and Yefeng Zheng<sup>*</sup> (Jarvis Lab)
- **Iteratively-Refined Interactive 3D Medical Image Segmentation with Multi-Agent Reinforcement Learning**  
  *CVPR* (2020). [[Paper]](https://arxiv.org/abs/1911.10334)  
  > From Xuan Liao<sup>1</sup> (SJTU), Wenhao Li<sup>1</sup> (East China Normal Univ.) and Yanfeng Wang<sup>*</sup> (SJTU)
- **Self-supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentation**  
  *CVPR* (2020). [[Paper]](https://arxiv.org/abs/2004.04581) [[Code]](https://github.com/YudeWang/SEAM)
  > From Yude Wang<sup>1</sup> (UCAS & CAS) and Xilin Chen<sup>*</sup> (UCAS & CAS)
  
 
 ### Plane Localization
- **Automatic View Planning with Multi-scale Deep Reinforcement Learning Agents**   
  *MICCAI* (2018). [[Paper]](https://arxiv.org/pdf/1806.03228.pdf) [[Code]](https://git.io/vhuMZ)  
  > From Amir Alansary<sup>1</sup> (ICL) and Daniel Rueckert<sup>*</sup> (ICL)  
- **Ultrasound Video Summarization using Deep Reinforcement Learning**  
  *MICCAI* (2020). [[Paper]](https://arxiv.org/pdf/2005.09531.pdf)
  > From Tianrui Liu<sup>1</sup> (ICL) and Bernhard Kainz<sup>*</sup> (ICL)

### Blending, Simulation
- **GP-GAN: Towards Realistic High-Resolution Image Blending**  
  *ACMMM* (2019). [[Paper]](https://arxiv.org/pdf/1703.07195.pdf) [[Code]](https://github.com/wuhuikai/GP-GAN)  
  > From Huikai Wu<sup>1</sup> (CAS) and Kaiqi Huang<sup>*</sup> (CAS).
- **Learning to Simulate Complex Scenes**  
  *ACMMM* (2019). [[Paper]](https://arxiv.org/pdf/2006.14611.pdf)
  > From Zhenfeng Xue<sup>1</sup> (Zhejiang Univ.) and Weijie Mao<sup>*</sup> (Zhejiang Univ.)

## CodeBox
> CodeBox for fast coding  

- **Metrics**
  - [segmentation](codebox/metrics/segmentation.py)
    - dice_ratio
  - [registration](codebox/metrics/registration.py)
    - negative_Jacobian

