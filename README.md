# Adversarial-Attack-Defense-Papers

*A collection of Adversarial Attack & Defense papers.*

## Outline
* [Dataset and Network](#0)
* [Adversarial Attack Method](#1)
* [Adversarial Defense Method](#2)

<h2 id="0">Dataset and Network</h2>

| Dataset | Introduction | Link
| :----: | :----: | :----: |
| NIPS2017 | A subset of ImageNet validation set containing 1000 images, which are used in the NIPS 2017 competition. | [Download](https://drive.google.com/file/d/1Z5VTMQrtJRymfN8AZJ2jjU8zdjMrMqIR/view?usp=sharing)
| SACP2019 | A subset of ImageNet validation set containing 1216 images, which are used in Tianchi Security AI Challenger Program Competition. | [Download](https://drive.google.com/file/d/1oC1ITY8SnQeeC4JxAnGh5HNItasdTQnx/view?usp=sharing)
| ImageNet-A | [Natural Adversarial Examples: Real-world, unmodified, and naturally occurring examples that cause classifier accuracy to significantly degrade](https://openaccess.thecvf.com/content/CVPR2021/html/Hendrycks_Natural_Adversarial_Examples_CVPR_2021_paper.html). | [Download](https://github.com/hendrycks/natural-adv-examples)
| ImageNet-C | [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.org/abs/1903.12261). | [Download](https://github.com/hendrycks/robustness)

| Network | Paper | Source | Code 
| :----: | :----: | :----: | :----: | 
| VggNet | [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) | ICLR2015  | [Code](https://github.com/tensorflow/models/tree/master/research/slim)
| InceptionNet | [Rethinking the inception architecture for computer vision](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.html) | CVPR2016 | [Code](https://github.com/tensorflow/models/tree/master/research/slim)
| ResNet | [Deep residual learning for image recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) | CVPR2016 | [Code](https://github.com/tensorflow/models/tree/master/research/slim)
| DenseNet | [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) | CVPR2017 | [Code](https://github.com/tensorflow/models/tree/master/research/slim)
| EfficientNet | [EfficientNet: Rethinking model scaling for convolutional neural networks](http://proceedings.mlr.press/v97/tan19a.html) | ICML2019 | [Code](https://github.com/tensorflow/models/tree/master/research/slim)
| ViT | [An image is worth 16x16 words: Transformers for image recognition at scale](https://arxiv.org/abs/2010.11929) | ICLR2020 | [Code](https://github.com/google-research/vision_transformer)
| AdvModels | [Adversarial Machine Learning at Scale](https://arxiv.org/abs/1611.01236) | ICLR2016 | [Code1](https://github.com/tensorflow/models/tree/benchmark/research/adv_imagenet_models) [Code2](https://drive.google.com/drive/folders/1sfD2IIDbBdbWc13IQlRLEBWHJ1NDo-lm?usp=drive_link)
| EnsAdvModels | [Ensemble Adversarial Training: Attacks and Defenses](https://arxiv.org/abs/1705.07204) | ICLR2018 | [Code1](https://github.com/tensorflow/models/tree/benchmark/research/adv_imagenet_models) [Code2](https://drive.google.com/drive/folders/1sfD2IIDbBdbWc13IQlRLEBWHJ1NDo-lm?usp=drive_link)


<h2 id="1">Adversarial Attack</h2>

| Name | Paper | Source | Type | Code  
| :----: | :----: | :----: | :----: | :----:  
| FGSM | [Explaining and harnessing adversarial examples](https://arxiv.org/abs/1412.6572) | ICLR2015 | white-box & transfer | 
| BIM | [Adversarial examples in the physical world](https://arxiv.org/pdf/1607.02533.pdf) | ICLR2017 Workshop | white-box | 
| UAP | [Universal Adversarial Perturbations](https://openaccess.thecvf.com/content_cvpr_2017/html/Moosavi-Dezfooli_Universal_Adversarial_Perturbations_CVPR_2017_paper.html) | CVPR2017 | white-box / universal |  [Code](https://github.com/LTS4/universal)
| C&W | [Towards Evaluating the Robustness of Neural Networks](https://ieeexplore.ieee.org/abstract/document/9428372/) | SP2017 | white-box / transfer |  [Code](https://github.com/carlini/nn_robust_attacks)
| MIM | [Boosting Adversarial Attacks With Momentum](https://openaccess.thecvf.com/content_cvpr_2018/papers/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.pdf) | CVPR2018 | white-box / transfer |  [Code](https://github.com/dongyp13/Non-Targeted-Adversarial-Attacks)
| DIM | [Improving Transferability of Adversarial Examples With Input Diversity](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xie_Improving_Transferability_of_Adversarial_Examples_With_Input_Diversity_CVPR_2019_paper.pdf) | CVPR2019 | white-box / transfer |  [Code](https://github.com/cihangxie/DI-2-FGSM)
| TIM | [Evading defenses to transferable adversarial examples by translation-invariant attacks](https://openaccess.thecvf.com/content_CVPR_2019/papers/Dong_Evading_Defenses_to_Transferable_Adversarial_Examples_by_Translation-Invariant_Attacks_CVPR_2019_paper.pdf) | CVPR2019 | white-box / transfer |  [Code](https://github.com/dongyp13/Translation-Invariant-Attacks)
| SIM | [Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks](https://arxiv.org/abs/1908.06281) | ICLR2020 | white-box / transfer |  [Code](https://github.com/JHL-HUST/SI-NI-FGSM)
| DEM | [Improving the Transferability of Adversarial Examples with Resized-Diverse-Inputs, Diversity-Ensemble and Region Fitting](https://link.springer.com/chapter/10.1007/978-3-030-58542-6_34) | ECCV2020 | white-box / transfer |  [Code](https://github.com/278287847/DEM)
| SAM | [Enhancing Adversarial Examples Via Self-Augmentation](https://ieeexplore.ieee.org/abstract/document/9428372/) | ICME2021 | white-box / transfer |  [Code](https://github.com/zhuangwz/ICME2021_self_augmentation)
| VIM | [Enhancing the Transferability of Adversarial Attacks through Variance Tuning](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Enhancing_the_Transferability_of_Adversarial_Attacks_Through_Variance_Tuning_CVPR_2021_paper.html) | CVPR2020 | white-box / transfer |  [Code](https://github.com/JHL-HUST/VT)
| BPDA | [Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples](https://arxiv.org/abs/1802.00420) | ICML2018 | white-box |  [Code](https://github.com/anishathalye/obfuscated-gradients)
| RP2 | [Robust Physical-World Attacks on Deep Learning Visual Classification](https://openaccess.thecvf.com/content_cvpr_2018/html/Eykholt_Robust_Physical-World_Attacks_CVPR_2018_paper) | CVPR2018 | white-box / physical |  [Code](https://github.com/evtimovi/robust_physical_perturbations)
| NAG | [NAG: Network for Adversary Generation](https://openaccess.thecvf.com/content_cvpr_2018/html/Mopuri_NAG_Network_for_CVPR_2018_paper.html) | CVPR2017 | white-box / GAN |  [Code](https://github.com/val-iisc/nag)
| SF | [SparseFool: a few pixels make a big difference](https://openaccess.thecvf.com/content_CVPR_2019/html/Modas_SparseFool_A_Few_Pixels_Make_a_Big_Difference_CVPR_2019_paper.html) | CVPR2019 | white-box |  [Code](http://github.com/LTS4/SparseFool)
| GAP | [Generative Adversarial Perturbations](https://openaccess.thecvf.com/content_cvpr_2018/html/Poursaeed_Generative_Adversarial_Perturbations_CVPR_2018_paper.html) | CVPR2018 | white-box / GAN |  [Code](https://github.com/OmidPoursaeed/Generative_Adversarial_Perturbations)
| RH | [Regional Homogeneity: Towards Learning Transferable Universal Adversarial Perturbations Against Defenses](https://link.springer.com/chapter/10.1007%2F978-3-030-58621-8_46) | ECCV2020 | white-box / transfer / universal |  [Code](https://github.com/LiYingwei/Regional-Homogeneity#regional-homogeneity-towards-learning-transferable-universal-adversarial-perturbations-against-defenses)
| CAMOU | [CAMOU: Learning Physical Vehicle Camouflages to Adversarially Attack Detectors in the Wild](https://openreview.net/pdf?id=SJgEl3A5tm) | ICLR2019 | black-box / physical |  [Code](https://github.com/naufalso/camou-iclr2019-tf)
| UPC | [Universal Physical Camouflage Attacks on Object Detectors](https://openaccess.thecvf.com/content_CVPR_2020/html/Huang_Universal_Physical_Camouflage_Attacks_on_Object_Detectors_CVPR_2020_paper.html) | CVPR2020 | white-box / physical |  [Code](https://mesunhlf.github.io/index_physical.html)
| AdvPatch | [Fooling automated surveillance cameras: adversarial patches to attack person detection](https://openaccess.thecvf.com/content_CVPRW_2019/html/CV-COPS/Thys_Fooling_Automated_Surveillance_Cameras_Adversarial_Patches_to_Attack_Person_Detection_CVPRW_2019_paper.html) | CVPRW2019 | white-box / physical |  [Code](https://gitlab.com/EAVISE/adversarial-yolo)
| TRA | [Trust Region Based Adversarial Attack on neural networks](https://openaccess.thecvf.com/content_CVPR_2019/html/Yao_Trust_Region_Based_Adversarial_Attack_on_Neural_Networks_CVPR_2019_paper.html) | CVPR2019 | white-box |  [Code](https://mesunhlf.github.io/index_physical.html)
| AA | [Feature Space Perturbations Yield More Transferable Adversarial Examples](https://openaccess.thecvf.com/content_CVPR_2019/html/Inkawhich_Feature_Space_Perturbations_Yield_More_Transferable_Adversarial_Examples_CVPR_2019_paper.html) | CVPR2019 | white-box / transfer |  [Code](https://github.com/QwQ2000/Activation-Attack-Pytorch)
| Ghost | [Learning Transferable Adversarial Examples via Ghost Networks](https://ojs.aaai.org/index.php/AAAI/article/view/6810) | AAA2019 | white-box / transfer |  [Code](https://github.com/LiYingwei/ghost-network)
| ILA | [Enhancing Adversarial Example Transferability with an Intermediate Level Attack](https://arxiv.org/abs/1907.10823) | ICCV2019 | white-box / transfer |  [Code](https://github.com/CUAI/Intermediate-Level-Attack)
| Boundary | [Decision-based adversarial attacks: Reliable attacks against black-box machine learning models](https://arxiv.org/abs/1712.04248) | ICLR2018 | black-box / decision |  [Code](https://github.com/greentfrapp/boundary-attack)
| HLA | [Query-Efficient Hard-label Black-box Attack: An Optimization-based Approach](https://arxiv.org/abs/1807.04457) | ICLR2019 | black-box / decision |  [Code](https://github.com/LeMinhThong/blackbox-attack)
| SimBA | [Simple Black-box Adversarial Attacks](https://proceedings.mlr.press/v97/guo19a.html) | ICML2019 | black-box / decision |  [Code](https://github.com/cg563/simple-blackbox-attack)
| GDA | [A Geometry-Inspired Decision-Based Attack](https://openaccess.thecvf.com/content_ICCV_2019/html/Liu_A_Geometry-Inspired_Decision-Based_Attack_ICCV_2019_paper.html) | ICCV2019 | black-box / decision |  [Code](https://github.com/ShahryarBQ/qFool)
| C&W | [Curls &Whey: Boosting Black-Box Adversarial Attacks](https://openaccess.thecvf.com/content_CVPR_2019/html/Shi_Curls__Whey_Boosting_Black-Box_Adversarial_Attacks_CVPR_2019_paper.html) | CVPR2019 | black-box |  [Code](https://github.com/walegahaha/Curls-Whey)
| AutoAttack | [Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks](http://proceedings.mlr.press/v119/croce20b.html) | ICML2020 | white-box |  [Code](https://github.com/hcguoO0/FIA)
| AdvLB | [Adversarial Laser Beam: Effective Physical-World Attack to DNNs in a Blink](https://arxiv.org/abs/2103.06504) | CVPR2021 | white-box / physical |  [Code](https://github.com/RjDuan/Advlight)
| LAFEAT | [LAFEAT: Piercing Through Adversarial Defenses with Latent Features](https://openaccess.thecvf.com/content/CVPR2021/html/Yu_LAFEAT_Piercing_Through_Adversarial_Defenses_With_Latent_Features_CVPR_2021_paper.html) | CVPR2021 | white-box |  [Code](https://github.com/lafeat/lafeat)
| FIA | [Feature Importance-aware Transferable Adversarial Attacks](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Feature_Importance-Aware_Transferable_Adversarial_Attacks_ICCV_2021_paper.html) | ICCV2021 | white-box / transfer |  [Code](https://github.com/hcguoO0/FIA)
| TTP | [On Generating Transferable Targeted Perturbations](https://openaccess.thecvf.com/content/ICCV2021/html/Naseer_On_Generating_Transferable_Targeted_Perturbations_ICCV_2021_paper.html) | ICCV2021 | white-box / transfer |  [Code](https://github.com/Muzammal-Naseer/TTP)
| DUAP | [Data-free Universal Adversarial Perturbation and Black-box Attack](https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Data-Free_Universal_Adversarial_Perturbation_and_Black-Box_Attack_ICCV_2021_paper.html) | ICCV2021 | white-box / universal |  [Code](https://bit.ly/3y0ZTIC)
| Admix | [Admix: Enhancing the Transferability of Adversarial Attacks](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Admix_Enhancing_the_Transferability_of_Adversarial_Attacks_ICCV_2021_paper.html) | ICCV2021 | white-box / transfer |  [Code](https://github.com/JHL-HUST/Admix)
| CA2 | [Cyclical Adversarial Attack Pierces Black-box Deep Neural Networks](https://www.sciencedirect.com/science/article/pii/S0031320322003120) | PR2022 | white-box / transfer |  [Code](https://github.com/mesunhlf/CA2)
| DEFEAT | [DEFEAT: Decoupled feature attack across deep neural networks. Neural Networks](https://www.sciencedirect.com/science/article/abs/pii/S0893608022003434) | NN2022 | white-box / transfer |  [Code](https://github.com/mesunhlf/DEFEAT)
| C-GSP | [Boosting Transferability of Targeted Adversarial Examples via Hierarchical Generative Networks](https://link.springer.com/chapter/10.1007/978-3-031-19772-7_42) | ECCV2022 | white-box / transfer |  [Code](https://github.com/ShawnXYang/C-GSP)


<h2 id="2">Adversarial Defense</h2>

| Name | Paper | Source | Type | Dataset | Code  
| :----: | :----: | :----: | :----: | :----: | :----: |
| AT | [Towards deep learning models resistant to adversarial attacks](https://arxiv.org/abs/1706.06083) | ICLR2018 | adversarial training | ImageNet / CIFAR10 / MNIST | [Code](https://github.com/tensorflow/models/tree/r1.12.0/research/adv_imagenet_models)
| EAT | [Ensemble Adversarial Training: Attacks and Defenses](https://arxiv.org/abs/1705.07204) | ICLR2018 | adversarial training | ImageNet | [Code](https://github.com/tensorflow/models/tree/r1.12.0/research/adv_imagenet_models)
| CAT | [Curriculum Adversarial Training](https://arxiv.org/abs/1805.04807) | IJCAI2018 | adversarial training | CIFAR100 / CIFAR10 / MNIST | [Code](https://github.com/sunblaze-ucb/curriculum-adversarial-training-CAT)
| FAT | [Attacks which do not kill training make adversarial learning stronger](http://proceedings.mlr.press/v119/zhang20z.html) | ICML2020 | adversarial training | CIFAR10 / SVHN | [Code](https://github.com/zjfheart/Friendly-Adversarial-Training)
| RS | [Certified Adversarial Robustness via Randomized Smoothing](http://proceedings.mlr.press/v97/cohen19c.html) | ICML2019 | adversarial training | ImageNet | [Code](http://github.com/locuslab/smoothing)
| STN | [Certified Adversarial Robustness with Additive Noise](https://arxiv.org/abs/1809.03113) | NIPS2019 | adversarial training | CIFAR10 | [Code](https://github.com/Bai-Li/STN-Code)
| R&P | [Mitigating Adversarial Effects Through Randomization](https://arxiv.org/abs/1711.01991) | ICLR2018 | plug-in | ImageNet | [Code](https://github.com/cihangxie/NIPS2017_adv_challenge_defense)
| HGD | [Defense against Adversarial Attacks Using High-Level Representation Guided Denoiser](https://openaccess.thecvf.com/content_cvpr_2018/html/Liao_Defense_Against_Adversarial_CVPR_2018_paper.html) | CVPR2018 | plug-in | ImageNet |  [Code](https://github.com/lfz/Guided-Denoise)
| BDR | [Countering Adversarial Images using Input Transformations](https://arxiv.org/abs/1711.00117) | ICLR2018 | plug-in | ImageNet |  [Code](https://github.com/facebookresearch/adversarial_image_defenses)
| JPEG | [Feature Distillation: DNN-Oriented JPEG Compression Against Adversarial Examples](https://arxiv.org/abs/1803.05787) | CVPR2019 | plug-in | ImageNet | [Code](https://github.com/sibosutd/feature-distillation)
| FD | [Feature Denoising for Improving Adversarial Robustness](https://openaccess.thecvf.com/content_CVPR_2019/html/Xie_Feature_Denoising_for_Improving_Adversarial_Robustness_CVPR_2019_paper.html) | CVPR2019 | adversarial training | ImageNet |  [Code](https://github.com/facebookresearch/ImageNet-Adversarial-Training)
| PD | [Deflecting Adversarial Attacks with Pixel Deflection](https://openaccess.thecvf.com/content_cvpr_2018/html/Prakash_Deflecting_Adversarial_Attacks_CVPR_2018_paper.html) | CVPR2019 | plug-in | ImageNet |  [Code](https://github.com/iamaaditya/pixel-deflection)
| PRN | [Defense against Universal Adversarial Perturbations](https://openaccess.thecvf.com/content_cvpr_2018/html/Akhtar_Defense_Against_Universal_CVPR_2018_paper.html) | CVPR2018 | adversarial training | ImageNet |  [Code](https://github.com/liujianee/Pertrubation_Rectifying_Network)
| COM | [ComDefend: An Efficient Image Compression Model to Defend Adversarial Examples](https://openaccess.thecvf.com/content_CVPR_2019/html/Jia_ComDefend_An_Efficient_Image_Compression_Model_to_Defend_Adversarial_Examples_CVPR_2019_paper.html) | CVPR2019 | plug-in | ImageNet |  [Code](https://github.com/jiaxiaojunQAQ/Comdefend)
| RD | [Defending against adversarial attacks by randomized diversification](https://openaccess.thecvf.com/content_CVPR_2019/html/Taran_Defending_Against_Adversarial_Attacks_by_Randomized_Diversification_CVPR_2019_paper.html) | CVPR2019 | adversarial training | CIFAR10 / MNIST | [Code](https://github.com/taranO/defending-adversarial-attacks-by-RD)
| ADP | [Improving adversarial robustness via promoting ensemble diversity](http://proceedings.mlr.press/v97/pang19a) | ICML2019 | adversarial training / ensemble |CIFAR100 / CIFAR10 / MNIST | [Code](https://github.com/P2333/Adaptive-Diversity-Promoting)
| TRADES | [Theoretically Principled Trade-off between Robustness and Accuracy](http://proceedings.mlr.press/v97/zhang19p.html) | ICML2019 | adversarial training | CIFAR10 / MNIST | [Code](https://github.com/yaodongyu/TRADES)
| FS | [Defense Against Adversarial Attacks Using Feature Scattering-based Adversarial Training](https://proceedings.neurips.cc/paper/2019/hash/d8700cbd38cc9f30cecb34f0c195b137-Abstract.html) | NIPS2019 | adversarial training | CIFAR10 | [Code](https://github.com/Haichao-Zhang/FeatureScatter)
| DVERGE | [DVERGE: Diversifying Vulnerabilities for Enhanced Robust Generation of Ensembles](https://arxiv.org/abs/2009.14720) | NIPS2020 | adversarial training / ensemble | CIFAR10 | [Code](https://github.com/zjysteven/DVERGE)
| RobNet | [When NAS Meets Robustness: In Search of Robust Architectures against Adversarial Attacks](https://arxiv.org/abs/1911.10695) | CVPR2020 | adversarial training | CIFAR10 / SVHN / ImageNet | [Code](https://github.com/gmh14/RobNets)
| NRP | [A Self-supervised Approach for Adversarial Robustness](https://arxiv.org/abs/2006.04924) | CVPR2020 | adversarial training | ImageNet | [Code](https://github.com/gmh14/RobNets)
| AVMixup | [Adversarial Vertex Mixup: Toward Better Adversarially Robust Generalization](https://arxiv.org/abs/2003.02484) | CVPR2020 | adversarial training | MNIST / CIFAR10 / SVHN | [Code](https://github.com/Saehyung-Lee/cifar10_challenge)
| ADT | [Adversarial distributional training for robust deep learning](https://arxiv.org/abs/2002.05999) | NIPS2020 | adversarial training | CIFAR100 / SVHN | [Code](https://github.com/dongyp13/Adversarial-Distributional-Training)
| FLSS | [Boosting Adversarial Robustness using Feature Level Stochastic Smoothing](https://openaccess.thecvf.com/content/CVPR2021W/SAIAD/html/Addepalli_Boosting_Adversarial_Robustness_Using_Feature_Level_Stochastic_Smoothing_CVPRW_2021_paper.html) | CVPR2021 | adversarial training | CIFAR100 / CIFAR10 | [Code](https://github.com/val-iisc/FLSS)
| Det-AdvProp | [Robust and Accurate Object Detection via Adversarial Learning](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Robust_and_Accurate_Object_Detection_via_Adversarial_Learning_CVPR_2021_paper.html) | CVPR2021 | adversarial training | COCO2017 | [Code](https://github.com/google/automl/blob/master/efficientdet/Det-AdvProp.md)
| AdvCL | [When Does Contrastive Learning Preserve Adversarial Robustness from Pretraining to Finetuning?](https://proceedings.neurips.cc/paper/2021/hash/b36ed8a07e3cd80ee37138524690eca1-Abstract.html) | NIPS2021 | adversarial training / contrastive learning | CIFAR10 / CIFAR100 | [Code](https://github.com/LijieFan/AdvCL)
| PAT | [Perceptual Adversarial Robustness: Defense Against Unseen Threat Models](https://arxiv.org/abs/2006.12655) | ICLR2021 | adversarial training | CIFAR10 / ImageNet | [Code](https://github.com/cassidylaidlaw/perceptual-advex)
| ERS | [Boosting randomized smoothing with variance reduced classifiers](https://arxiv.org/abs/2106.06946) | ICLR2022 | adversarial training / ensemble | CIFAR10 / ImageNet | [Code](https://github.com/eth-sri/smoothing-ensembles)
| LAS-AT | [LAS-AT: Adversarial Training with Learnable Attack Strategy](https://openaccess.thecvf.com/content/CVPR2022/html/Jia_LAS-AT_Adversarial_Training_With_Learnable_Attack_Strategy_CVPR_2022_paper.html) | CVPR2022 | adversarial training | CIFAR10 / CIFAR100 / Tiny Imagenet | [Code](https://github.com/jiaxiaojunQAQ/LAS-AT)
| DeACL | [Decoupled Adversarial Contrastive Learning for Self-supervised Adversarial Robustness](https://link.springer.com/chapter/10.1007/978-3-031-20056-4_42) | ECCV2022 | adversarial training / contrastive learning | CIFAR10 / CIFAR100 | [Code](https://github.com/pantheon5100/DeACL)
| RAW | [All You Need Is RAW: Defending Against Adversarial Attacks with Camera Image Pipelines](https://link.springer.com/chapter/10.1007/978-3-031-19800-7_19) | ECCV2022 | plug-in | ImageNet / COCO | [Code](https://github.com/princeton-computational-imaging/allyouneedisraw)