# LearningNote
> A record of my course for deep learning

## Contents
- [Programming Framework](#program)
- [Dataset](#dataset)
- [Journals](#journal)
- [Paper](#paper)
- [Leading Group & Researcher](#researcher)
- [CodeBox](#code)


## <span id = "program">Programming Framework</span>
> Contents of the programming framework for machine learning and medical image computing  

### [PyTorch](https://pytorch.org/)  
 An open source machine learning framework that accelerates the path from research prototyping to production deployment

### PyTorch Extension Library
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
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d)
  > FAIR's Library of Reusable Components for Deep Learning with 3D Data
- [Torch Optimizer](https://github.com/jettify/pytorch-optimizer)
  > Collection of Optimizers for Pytorch

### Python Extension Library
- [ANTsPy](https://github.com/ANTsX/ANTsPy)
  > Advanced Normalization Tools in Python
- [CuPy](https://github.com/cupy/cupy)
  > NumPy-like API Accelerated with CUDA
- [MedPy](https://github.com/loli/medpy)  
  > Medical Image Processing in Python
- [SimpleITK](https://github.com/SimpleITK/SimpleITK)
  > An Image Analysis Toolkit
- [Gym](https://github.com/openai/gym)
  > A Toolkit for Developing and Comparing Reinforcement Learning Algorithms
- [OpenCV](https://opencv.org/)
  > Open Source Computer Vision Library
- [DIPY](https://github.com/dipy/dipy)
  > Python Library for the Analysis of MR Diffusion Imaging.
- [Trimesh](https://github.com/mikedh/trimesh)
  > Python Library for Loading and Using Triangular Meshes.
- [Open3d](https://github.com/intel-isl/Open3D)
  > A Modern Library for 3D Data Processing

### Software, APP & SDK 
- [itk-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php)
  > A Software Application Used to Segment Structures in 3D Medical Images
- [ParaView](https://www.paraview.org/)
  > An Open-source, Multi-platform Data Analysis and Visualization Application
- [3DMeshMetric](https://www.nitrc.org/projects/meshmetric3d/)
  > A Visualization Tool Based on the VTK Library
- [3DSlicer](https://www.slicer.org/)
  > An Open Source Software Platform for Medical Image Informatics, Image Processing, and Three-dimensional Visualization
- [TensorRT](https://developer.nvidia.com/tensorrt)
  > An SDK for High-performance Deep Learning Inference

## <span id = "dataset">Dataset</span>
### Natural Images
#### Classification
- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html)
#### 3D Shape
- [ShapeNet](https://www.shapenet.org/)
### Medical Images
#### Heart
- [Automated Cardiac Diagnosis Challenge (ACDC)](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html)
- [Left Atrial Wall Thickness Challenge (SLAWT)](https://www.doc.ic.ac.uk/~rkarim/la_lv_framework/wall/index.html)
- [Left Atrium Fibrosis Benchmark (cDEMRIS)](https://www.doc.ic.ac.uk/~rkarim/la_lv_framework/fibrosis/index.html)
- [Left Ventricle Infarct](https://www.doc.ic.ac.uk/~rkarim/la_lv_framework/lv_infarct/index.html)
#### Eye
- [DRIVE: Digital Retinal Images for Vessel Extraction](https://drive.grand-challenge.org/DRIVE/)
- [IOSTAR – Retinal Vessel Segmentation and Artery/Vein Classification Dataset](http://www.retinacheck.org/datasets)
#### Lung
- [JSRT](http://db.jsrt.or.jp/eng.php)
- [Montgomery County X-ray Set](https://lhncbc.nlm.nih.gov/publication/pub9931)
- [Shenzhen Hospital X-ray Set](https://lhncbc.nlm.nih.gov/publication/pub9931)

## <span id = "journal">Journals</span>
### Springer
- [**Nature Medicine**](https://www.nature.com/nm/) IF=36.13
- [**Nature Biomedical Engineering**](https://www.nature.com/natbiomedeng/) IF=18.952
- [**Nature Machine Intelligence**](https://www.nature.com/natmachintell/) 
- [**International Journal of Computer Vision**](https://www.springer.com/journal/11263) IF=5.698
### IEEE
- [**IEEE Transactions on Pattern Analysis and Machine Intelligence**](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34) IF=17.861
- [**IEEE Transactions on Cybernetics**](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6221036) IF=11.079
- [**IEEE Transactions on Medical Imaging**](https://www.embs.org/tmi/) IF=6.685
- [**IEEE Transactions on Biomedical Engineering**](https://www.embs.org/tbme/) IF=4.424
- [**IEEE Transactions on Computational Imaging**](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6745852) IF=4.015
- [**IEEE Journal of Biomedical And Health Informatics**](https://www.embs.org/jbhi/) IF=5.223
- [**IEEE Access**](https://ieeeaccess.ieee.org/) IF=3.745
### Elsevier
- [**Medical Image Analysis**](https://www.journals.elsevier.com/medical-image-analysis) IF=11.148
- [**Pattern Recognition**](https://www.journals.elsevier.com/pattern-recognition) IF=7.196
- [**NeuroImage**](https://www.journals.elsevier.com/neuroimage) IF=5.902
- [**Neurocomputing**](https://www.journals.elsevier.com/neurocomputing) IF=4.438
- [**Ultrasound in Medicine & Biology**](https://www.journals.elsevier.com/ultrasound-in-medicine-and-biology) IF=2.514
- [**Computer Methods and Programs in Biomedicine**](https://www.journals.elsevier.com/computer-methods-and-programs-in-biomedicine) IF=3.632

## <span id = "paper">Papers</span>
> Paper list for deep learning in computer vision and medical image computing 

### Summary, Survey & Review
- [**Deep Learning**](https://www.nature.com/articles/nature14539)  *Nature,* 2015.
- [**A Survey on Deep Learning in Medical Image Analysis**](https://www.sciencedirect.com/science/article/pii/S1361841517301135)  *Medical Image Analysis,* 2017.
- [**Bag of Tricks for Image Classification with Convolutional Neural Networks**](https://arxiv.org/pdf/1812.01187v2.pdf)  *CVPR,* 2019.
- [**Deep Learning in Medical Ultrasound Analysis: A Review**](https://www.sciencedirect.com/science/article/pii/S2095809918301887) *Engineering,* 2019.
- [**Deep Learning for Cardiac Image Segmentation: A Review**](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7066212/)  *Frontiers in Cardiovascular Medicine,* 2020.
- [**Self-supervised Visual Feature Learning with Deep Neural Networks: A Survey**](https://ieeexplore.ieee.org/abstract/document/9086055)  *IEEE Transactions on Pattern Analysis and Machine Intelligence,* 2020.

### Basic Technology
- [**Dropout: A Simple Way to Prevent Neural Networks from Overfitting**](https://www.datopia.ir/wp-content/uploads/2018/12/srivastava14a.pdf) *Journal of Machine Learning Research,* 2014.
#### Normalization
- [**Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift**](http://proceedings.mlr.press/v37/ioffe15.html) *ICML,* 2015.
- [**Instance Normalization: The Missing Ingredient for Fast Stylization**](https://arxiv.org/abs/1607.08022) *arXiv,* 2016.
- [**Layer Normalization**](https://openreview.net/forum?id=BJLa_ZC9) *NeurIPS,* 2016.
- [**Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization**](https://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.pdf) *ICCV,* 2017.
- [**Group Normalization**](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.pdf) *ECCV,* 2018.

### Recognition
- [**ImageNet Classification with Deep Convolutional Neural Networks**](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)  *NeurIPS,* 2012.
- [**Very Deep Convolutional Networks for Large-Scale Image Recognition**](https://arxiv.org/abs/1409.1556)  *ICLR,* 2015.
- [**Going Deeper with Convolutions**](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)  *CVPR,* 2015.
- [**Deep Residual Learning for Image Recognition**](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)   *CVPR,* 2016.
- [**Identity Mappings in Deep Residual Networks**](https://rd.springer.com/chapter/10.1007/978-3-319-46493-0_38)  *ECCV,* 2016.
- [**Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning**](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14806/14311) *AAAI,* 2017.
- [**Densely Connected Convolutional Networks**](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf)  *CVPR,* 2017.
- [**Aggregated Residual Transformations for Deep Neural Networks**](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf)  *CVPR,* 2017.
- [**Dynamic Routing Between Capsules**](https://arxiv.org/pdf/1710.09829.pdf) *NeurIPS,* 2017.
- [**Squeeze-and-Excitation Networks**](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf)  *CVPR,* 2018.
- [**MixConv: Mixed Depthwise Convolutional Kernels**](https://arxiv.org/pdf/1907.09595.pdf)  *BMVC,* 2019.
- [**Res2Net: A New Multi-scale Backbone Architecture**](https://ieeexplore.ieee.org/abstract/document/8821313)  *IEEE Transactions on Pattern Analysis and Machine Intelligence,* 2019.
#### Fine-grained
- [**Focus Longer to See Better: Recursively Refined Attention for Fine-grained Image Classification**](https://arxiv.org/pdf/2005.10979.pdf)  *CVPR,* 2020.

### Segmentation
- [**Fully Convolutional Networks for Semantic Segmentation**](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)  *CVPR,* 2015. 
- [**U-Net: Convolutional Networks for Biomedical Image Segmentation**](https://rd.springer.com/chapter/10.1007/978-3-319-24574-4_28)  *MICCAI,* 2015. 
- [**3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation**](https://rd.springer.com/chapter/10.1007/978-3-319-46723-8_49)  *MICCAI,* 2016.
- [**V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation**](https://ieeexplore.ieee.org/abstract/document/7785132)  *3DV,* 2016.
- [**SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation**](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7803544) *IEEE Transactions on Pattern Analysis and Machine Intelligence,* 2017. 
- [**UNet++: A Nested U-Net Architecture for Medical Image Segmentation**](https://rd.springer.com/chapter/10.1007/978-3-030-00889-5_1) *DLMIA,* 2018.
- [**DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs**](https://ieeexplore.ieee.org/abstract/document/7913730)  *IEEE Transactions on Pattern Analysis and Machine Intelligence,* 2018.
- [**Capsules for Biomedical Image Segmentation**](https://www.sciencedirect.com/science/article/pii/S136184152030253X) *Medical Image Analysis,* 2020.
#### High-resolution & Efficient Model
- [**ICNet for Real-Time Semantic Segmentation on High-Resolution Images**](https://arxiv.org/abs/1704.08545)  *ECCV,* 2018.
- [**Collaborative Global-Local Networks for Memory-Efficient Segmentation of Ultra-High Resolution Images**](https://arxiv.org/pdf/1905.06368.pdf)   *ICCV,* 2019.
#### Semi-, Weakly-, One-shot & Few-shot Learning
- [**Uncertainty-aware Self-ensembling Model for Semi-supervised 3D Left Atrium Segmentation**](https://arxiv.org/pdf/1907.07034.pdf) *MICCAI,* 2019.
- [**LT-Net: Label Transfer by Learning Reversible Voxel-wise Correspondence for One-shot Medical Image Segmentation**](https://arxiv.org/abs/2003.07072)  *CVPR,* 2020.
- [**Self-supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentation**](https://arxiv.org/abs/2004.04581) *CVPR,* 2020.
#### Interaction
- [**Iteratively-Refined Interactive 3D Medical Image Segmentation with Multi-Agent Reinforcement Learning**](https://arxiv.org/abs/1911.10334) *CVPR,* 2020.
#### Uncertainty & Attention
- [**Attention U-Net: Learning Where to Look for the Pancreas**](https://openreview.net/pdf?id=Skft7cijM)  *MIDL,* 2018.
- [**Recalibrating Fully Convolutional Networks with Spatial and Channel “Squeeze and Excitation” Blocks**](https://arxiv.org/pdf/1808.08127.pdf) *IEEE Transactions on Medical Imaging,* 2018.
- [**Attention Gated Networks: Learning to Leverage Salient Regions in Medical Images**](https://www.sciencedirect.com/science/article/pii/S1361841518306133) *Medical Image Analysis,* 2019.
- [**Cars Can’t Fly up in the Sky: Improving Urban-Scene Segmentation via Height-driven Attention Networks**](https://arxiv.org/pdf/2003.05128.pdf) *CVPR,* 2020.
#### Loss Function
- [**Learning Active Contour Models for Medical Image Segmentation**](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Learning_Active_Contour_Models_for_Medical_Image_Segmentation_CVPR_2019_paper.pdf)  *CVPR,* 2019.
  
### Detection
- [**Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation**](https://openaccess.thecvf.com/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf)  *CVPR,* 2014. 
- [**Fast R-CNN**](https://openaccess.thecvf.com/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf) *ICCV,* 2015.
- [**Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks**](http://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf)  *NeurIPS,* 2015.
- [**Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition**](https://ieeexplore.ieee.org/abstract/document/7005506)  *IEEE Transactions on Pattern Analysis and Machine Intelligence,* 2015.
- [**SSD: Single Shot MultiBox Detector**](https://arxiv.org/pdf/1512.02325.pdf) *ECCV,* 2016.
- [**Focal Loss for Dense Object Detection**](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)  *ICCV,* 2017.
- [**Feature Pyramid Networks for Object Detection**](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf)  *CVPR,* 2017.
- [**Mask R-CNN**](https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf)  *ICCV,* 2017.

### Registration
- [**An Unsupervised Learning Model for Deformable Medical Image Registration**](https://arxiv.org/abs/1802.02604)  *CVPR,* 2018.
- [**Learning Conditional Deformable Templates with Convolutional Networks**](https://arxiv.org/pdf/1908.02738.pdf)  *NeurIPS,* 2019. 
#### Multi-modal registration
- [**Weakly-supervised Convolutional Neural Networks for Multi-modal Image Registration**](https://www.sciencedirect.com/science/article/pii/S1361841518301051)  *Medical Image Analysis,* 2018.
- [**Adversarial Learning for Mono- or Multi-modal Registration**](http://website60s.com/upload/files/adversarial-learning-for-mono-or-multi-modal-regis_2019_medical-image-analy.pdf) *Medical Image Analysis,* 2019.
- [**JSSR: A Joint Synthesis, Segmentation, and Registration System for 3D Multi-Modal Image Alignment of Large-scale Pathological CT Scans**](https://arxiv.org/pdf/2005.12209.pdf) *arXiv,* 2020.
- [**Adversarial Uni- and Multi-modal Stream Networks for Multimodal Image Registration**](https://arxiv.org/pdf/2007.02790.pdf) *MICCAI,* 2020.
- [**Unsupervised Multi-Modal Image Registration via Geometry Preserving Image-to-Image Translation**](https://arxiv.org/abs/2003.08073) *CVPR,* 2020.

### Reinforcement Learning
- [**Human-level Control through Deep Reinforcement Learning**](https://www.nature.com/articles/nature14236) *Nature,* 2015.
- [**Deep Reinforcement Learning with Double Q-learning**](https://arxiv.org/pdf/1509.06461.pdf) *AAAI,* 2016.
- [**Dueling Network Architectures for Deep Reinforcement Learning**](https://arxiv.org/pdf/1511.06581.pdf) *ICML,* 2016.
- [**Prioritized Experience Replay**](https://arxiv.org/pdf/1511.05952.pdf) *ICLR,* 2016.

### Self-supervised Learning
#### Contrast & Consistent Learning
- [**Robust Learning Through Cross-Task Consistency**](https://consistency.epfl.ch/Cross_Task_Consistency_CVPR2020.pdf)  *CVPR,* 2020.
- - [**Momentum Contrast for Unsupervised Visual Representation Learning**](https://arxiv.org/pdf/1911.05722.pdf) *CVPR,* 2020. 
- [**A Simple Framework for Contrastive Learning of Visual Representations**](https://arxiv.org/pdf/2002.05709.pdf)  *arXiv,* 2020.

### Graph Neural Network
- [**Learning Convolutional Neural Networks for Graphs**](http://proceedings.mlr.press/v48/niepert16.pdf) *ICML,* 2016 

### Pointset, Mesh & Shape Learning
#### Shape Generation & Editing
- [**Variational Graph Auto-Encoders**](https://arxiv.org/pdf/1611.07308.pdf) *arXiv,* 2016.
- [**A Point Set Generation Network for 3D Object Reconstruction from a Single Image**](https://openaccess.thecvf.com/content_cvpr_2017/papers/Fan_A_Point_Set_CVPR_2017_paper.pdf) *CVPR,* 2017.
- [**Variational Shape Completion for Virtual Planning of Jaw Reconstructive Surgery**](https://arxiv.org/pdf/1906.11957.pdf) *MICCAI,* 2019.
- [**Pixel2Mesh: 3D Mesh Model Generation via Image Guided Deformation**](https://ieeexplore.ieee.org/abstract/document/9055070) *IEEE Transactions on Pattern Analysis and Machine Intelligence,* 2020.
- [**Learning to Infer Semantic Parameters for 3D Shape Editing**](https://arxiv.org/pdf/2011.04755.pdf)  *arXiv,* 2020.
- [**Learning Part Generation and Assembly for Structure-Aware Shape Synthesis**](https://ojs.aaai.org//index.php/AAAI/article/view/6798) *AAAI,* 2020.
- [**Voxel2Mesh: 3D Mesh Model Generation from Volumetric Data**](https://arxiv.org/abs/1912.03681) *MICCAI,* 2020.

### Generative & Simulation model.
- [**Auto-Encoding Variational Bayes**](https://arxiv.org/pdf/1312.6114.pdf) *arXiv,* 2013. 
- [**Learning to Simulate Complex Scenes**](https://arxiv.org/pdf/2006.14611.pdf)  *ACMMM,* 2019.
- [**Learning to Simulate**](http://arxiv.org/pdf/1810.02513v2.pdf)  *ICLR,* 2019.
#### Medical Image Simulator
- [**Fast Realistic MRI Simulations Based on Generalized Multi-Pool Exchange Tissue Model**](https://ieeexplore.ieee.org/document/7676360)  *IEEE Transactions on Medical Imaging,* 2017.
- [**MRISIMUL: A GPU-Based Parallel Approach to MRI Simulations**](https://ieeexplore.ieee.org/document/6671435) *IEEE Transactions on Medical Imaging,* 2017.
#### Style Transfer
- [**A Style-Based Generator Architecture for Generative Adversarial Networks**](https://arxiv.org/abs/1812.04948) *CVPR,* 2019.

### Landmark Detection
- [**Evaluating Reinforcement Learning Agents for Anatomical Landmark Detection**](https://www.sciencedirect.com/science/article/pii/S1361841518306121) *Medical Image Analysis,* 2019.
- [**Multi-Scale Deep Reinforcement Learning for Real-Time 3D-Landmark Detection in CT Scans**](https://ieeexplore.ieee.org/abstract/document/8187667) *IEEE Transactions on Pattern Analysis and Machine Intelligence,* 2019.

### View Planning
#### 2D View Planning
- [**SonoNet: Real-Time Detection and Localisation of Fetal Standard Scan Planes in Freehand Ultrasound**](https://ieeexplore.ieee.org/document/7974824) *IEEE Transactions on Medical Imaging,* 2017.
- [**Ultrasound Video Summarization using Deep Reinforcement Learning**](https://arxiv.org/pdf/2005.09531.pdf)  *MICCAI,* 2020. 
#### 3D View Planning
- [**Standard Plane Detection in 3D Fetal Ultrasound Using an Iterative Transformation Network**](https://rd.springer.com/chapter/10.1007/978-3-030-00928-1_45)  *MICCAI,* 2018. 
- [**Automatic View Planning with Multi-scale Deep Reinforcement Learning Agents**](https://arxiv.org/pdf/1806.03228.pdf)  *MICCAI,* 2018. 

### Others
#### Anomaly Detection:
- [**GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training**](https://arxiv.org/pdf/1805.06725.pdf) *ACCV,* 2018. 
#### Known Operator
- [**PYRO-NN: Python Reconstruction Operators in Neural Networks**](https://pubmed.ncbi.nlm.nih.gov/31389023/)  *Medical Physics,* 2019.
#### Blending
- [**GP-GAN: Towards Realistic High-Resolution Image Blending**](https://arxiv.org/pdf/1703.07195.pdf) *ACMMM,* 2019.
#### Parameter Sharing
- [**Shapeshifter Networks: Cross-layer Parameter Sharing for Scalable and Effective Deep Learning**](https://arxiv.org/pdf/2006.10598.pdf)   *arXiv,* 2020.

## <span id = "researcher">Leading Group & Researcher</span>
| Affiliation | Name|
| :- | :- | 
| Apple | [Ian Goodfellow](https://scholar.google.com/citations?hl=en&user=iYN86KEAAAAJ&view_op=list_works&sortby=pubdate) |
| China Academy of Science | [S. Kevin Zhou](https://scholar.google.com/citations?user=8eNm2GMAAAAJ&hl=en&oi=ao) |
| Chinese University of Hong Kong | [Jiaya Jia](https://scholar.google.com/citations?user=XPAkzTEAAAAJ&hl=zh-CN) |
| Chinese University of Hong Kong | [Pheng-Ann Heng](https://scholar.google.com/citations?user=OFdytjoAAAAJ&hl=en&oi=ao) |
| Chinese University of Hong Kong | [Qi Dou](https://scholar.google.com/citations?user=iHh7IJQAAAAJ&hl=en) |
| Chinese University of Hong Kong| [Xiaoou Tang](https://scholar.google.com/citations?user=qpBtpGsAAAAJ&hl=en) |
| Cornell University | [Mert R. Sabuncu](https://scholar.google.com/citations?hl=en&user=Pig-I4QAAAAJ&view_op=list_works&sortby=pubdate) |
| Eindhoven University of Technology| [Josien Pluim](https://scholar.google.com/citations?hl=en&user=wjB-g1wAAAAJ)
| Facebook AI Research | [Kaiming He](https://scholar.google.com/citations?user=DhtAFkwAAAAJ&hl=en&oi=ao) |
| Facebook AI Research | [Ross Girshick](https://scholar.google.com/citations?user=W8VIEZgAAAAJ&hl=en) |
| Google Brain | [Quoc V. Le](https://scholar.google.com/citations?user=vfT6-XIAAAAJ&hl=en&oi=ao) |
| Google DeepMind | [David Silver](https://scholar.google.com/citations?user=-8DNE4UAAAAJ&hl=en&oi=ao) |
| Google Research | [Jeff Dean](https://scholar.google.com/citations?user=NMS69lQAAAAJ&hl=en&oi=ao) |
| Imperial College London | [Bernhard Kainz](https://scholar.google.com/citations?user=Igxq-YEAAAAJ&hl=en&oi=ao) |
| Imperial College London | [Daniel Rueckert](https://scholar.google.com/citations?user=H0O0WnQAAAAJ&hl=en&oi=ao) |
| Imperial College London | [Wenjia Bai](https://scholar.google.com/citations?user=IA1QFM4AAAAJ&hl=en&oi=ao) |
| King's College London | [Tom Vercauteren](https://scholar.google.com/citations?user=zduEJkcAAAAJ&hl=en&oi=ao) |
| Harvard Medical School | [Adrian V. Dalca](https://scholar.google.com/citations?user=zRy-zdAAAAAJ&hl=en&oi=ao) |
| Harvard Medical School | [Ali Gholipour](https://scholar.google.com/citations?user=mPB7nkYAAAAJ&hl=en&oi=ao) |
| Huazhong University of Science and Technology | [Xin Yang](https://scholar.google.com/citations?hl=en&user=lsz8OOYAAAAJ&view_op=list_works&sortby=pubdate)
| Massachusetts Institute of Technology | [Polina Golland](https://scholar.google.com/citations?user=4GpKQUIAAAAJ&hl=en) |
| Megvii | [Jian Sun](https://scholar.google.com/citations?user=ALVSZAYAAAAJ&hl=en) |
| New York University| [Yann LeCun](https://scholar.google.com/citations?user=WLN3QrAAAAAJ&hl=en&oi=ao) |
| Radboud University | [Bram van Ginneken](https://scholar.google.com/citations?user=O1j6_MsAAAAJ&hl=en&oi=ao) |
| Shanghai Jiao Tong University | [Qian Wang](https://scholar.google.com/citations?hl=en&user=m6ZNDewAAAAJ) |
| Shanghai Tech University | [Dinggang Shen](https://scholar.google.com/citations?user=v6VYQC8AAAAJ&hl=en) |
| Shenzhen University | [Dong Ni](https://scholar.google.com/citations?user=J27J2VUAAAAJ&hl=en&oi=ao) |
| Swiss Federal Institute of Technology Zurich | [Luc Van Gool](https://scholar.google.com/citations?hl=zh-CN&user=TwMib_QAAAAJ&view_op=list_works) |
| Technical University of Munich | [Nassir Navab](https://scholar.google.com/citations?user=kzoVUPYAAAAJ&hl=en&oi=ao) |
| Tencent | [Yefeng Zheng](https://scholar.google.com/citations?user=vAIECxgAAAAJ&hl=en&oi=ao) |
| United Imaging Intelligence | [Yaozong Gao](https://scholar.google.com/citations?hl=zh-CN&user=8BNBTaQAAAAJ&view_op=list_works&sortby=pubdate)
| University of Amsterdam | [Max Welling](https://scholar.google.com/citations?user=8200InoAAAAJ&hl=en&oi=ao) |
| University of British Columbia | [Purang Abolmaesumi](https://scholar.google.com/citations?user=gKZS5-IAAAAJ&hl=en&oi=ao) |
| University of Electronic Science and Technology of China | [Guotai Wang](https://scholar.google.com/citations?user=Z2sFN4EAAAAJ&hl=en&oi=ao) |
| University College London | [Yipeng Hu](https://scholar.google.com/citations?user=_jYXK0IAAAAJ&hl=en&oi=ao) |
| University of Leeds | [Alejandro Frangi](https://scholar.google.com/citations?user=9fGrB0sAAAAJ&hl=en&oi=ao) |
| University of Montreal | [Yoshua Bengio](https://scholar.google.com/citations?user=kukA0LcAAAAJ&hl=en) |
| University of North Carolina at Chapel Hill | [Pew-Thian Yap](https://scholar.google.com/citations?user=QGdnthwAAAAJ&hl=en&oi=ao) |
| University of Toronto | [Geoffrey Hinton](https://scholar.google.com/citations?user=JicYPdAAAAAJ&hl=en&oi=ao) |
| University of Western Ontario | [Shuo Li](https://scholar.google.com/citations?hl=en&user=6WNtJa0AAAAJ) |

## <span id = "code">CodeBox</span>
> CodeBox for fast coding  

- **Metrics**
  - [segmentation](codebox/metrics/segmentation.py)
    - dice_ratio
  - [registration](codebox/metrics/registration.py)
    - negative_Jacobian

