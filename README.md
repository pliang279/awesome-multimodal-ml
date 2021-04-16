# Reading List for Topics in Multimodal Machine Learning
By [Paul Liang](http://www.cs.cmu.edu/~pliang/) (pliang@cs.cmu.edu), [Machine Learning Department](http://www.ml.cmu.edu/) and [Language Technologies Institute](https://www.lti.cs.cmu.edu/), [CMU](https://www.cmu.edu/), with help from members of the [MultiComp Lab](http://multicomp.cs.cmu.edu/) at LTI, CMU. If there are any areas, papers, and datasets I missed, please let me know!

## Recent Content

[Social Intelligence in Humans and Robots](https://social-intelligence-human-ai.github.io/) @ ICRA 2021

[LANTERN 2021](https://www.lantern.uni-saarland.de/2021/): The Third Workshop Beyond Vision and LANguage: inTEgrating Real-world kNowledge @ EACL 2021

Multimodal workshops @ CVPR 2021: [MUltimodal Learning and Applications](https://mula-workshop.github.io/), [Sight and Sound](http://sightsound.org/), [Visual Question Answering](https://visualqa.org/workshop), [Embodied AI](https://embodied-ai.org/), [Language for 3D Scenes](http://language3dscenes.github.io/).

Multimodal workshops @ NAACL 2021: [MAI-Workshop](http://multicomp.cs.cmu.edu/naacl2021multimodalworkshop/), [ALVR](https://alvr-workshop.github.io/).

ICLR 2021 workshop on [Embodied Multimodal Learning](https://eml-workshop.github.io/).

Microsoft Research's work on [VinVL](https://www.microsoft.com/en-us/research/blog/vinvl-advancing-the-state-of-the-art-for-vision-language-models/?OCID=msr_blog_VinVL_fb): [blog](https://www.microsoft.com/en-us/research/blog/vinvl-advancing-the-state-of-the-art-for-vision-language-models/?OCID=msr_blog_VinVL_fb), [paper](https://arxiv.org/pdf/2101.00529.pdf).

OpenAI's work on [DALL·E](https://openai.com/blog/dall-e/) and CLIP: [blog](https://openai.com/blog/clip/), [paper](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language.pdf), and [code](https://github.com/openai/CLIP).

Follow our course [11-777 Multimodal Machine Learning](https://cmu-multicomp-lab.github.io/mmml-course/fall2020/), Fall 2020 @ CMU.

NeurIPS 2020 workshop on [Wordplay: When Language Meets Games](https://wordplay-workshop.github.io/).

ACL 2020 workshops on [Multimodal Language](http://multicomp.cs.cmu.edu/acl2020multimodalworkshop/) [(proceedings)](https://www.aclweb.org/anthology/volumes/2020.challengehml-1/) and [Advances in Language and Vision Research](https://alvr-workshop.github.io/).

Multimodal workshops @ ECCV 2020: [EVAL](https://askforalfred.com/EVAL/), [CAMP](https://camp-workshop.stanford.edu/), and [MVA](https://sites.google.com/view/multimodalvideo-v2).

## Table of Contents

* [Survey Papers](#survey-papers)
* [Core Areas](#core-areas)
  * [Representation Learning](#representation-learning)
  * [Multimodal Fusion](#multimodal-fusion)
  * [Multimodal Alignment](#multimodal-alignment)
  * [Multimodal Translation](#multimodal-translation)
  * [Missing or Imperfect Modalities](#missing-or-imperfect-modalities)
  * [Knowledge Graphs and Knowledge Bases](#knowledge-graphs-and-knowledge-bases)
  * [Intepretable Learning](#intepretable-learning)
  * [Generative Learning](#generative-learning)
  * [Semi-supervised Learning](#semi-supervised-learning)
  * [Self-supervised Learning](#self-supervised-learning)
  * [Language Models](#language-models)
  * [Adversarial Attacks](#adversarial-attacks)
  * [Few-Shot Learning](#few-shot-learning)
  * [Bias and Fairness](#bias-and-fairness)
  * [Human in the Loop Learning](#human-in-the-loop-learning)
* [Applications and Datasets](#applications-and-datasets)
  * [Language and Visual QA](#language-and-visual-qa)
  * [Language Grounding in Vision](#language-grounding-in-vision)
  * [Language Grouding in Navigation](#language-grouding-in-navigation)
  * [Multimodal Machine Translation](#multimodal-machine-translation)
  * [Multi-agent Communication](#multi-agent-communication)
  * [Commonsense Reasoning](#commonsense-reasoning)
  * [Multimodal Reinforcement Learning](#multimodal-reinforcement-learning)
  * [Multimodal Dialog](#multimodal-dialog)
  * [Language and Audio](#language-and-audio)
  * [Audio and Visual](#audio-and-visual)
  * [Media Description](#media-description)
  * [Video Generation from Text](#video-generation-from-text)
  * [Affect Recognition and Multimodal Language](#affect-recognition-and-multimodal-language)
  * [Healthcare](#healthcare)
  * [Robotics](#robotics)
  * [Autonomous Driving](#Autonomous-Driving)
  * [Finance](#Finance)
  * [Human AI Interaction](#Human-AI-Interaction)
* [Workshops](#workshops)
* [Tutorials](#tutorials)
* [Courses](#courses)


# Research Papers

## Survey Papers

[Experience Grounds Language](https://arxiv.org/abs/2004.10151), EMNLP 2020

[A Survey of Reinforcement Learning Informed by Natural Language](https://arxiv.org/abs/1906.03926), IJCAI 2019

[Multimodal Machine Learning: A Survey and Taxonomy](https://arxiv.org/abs/1705.09406), TPAMI 2019

[Multimodal Intelligence: Representation Learning, Information Fusion, and Applications](https://arxiv.org/abs/1911.03977), arXiv 2019

[Trends in Integration of Vision and Language Research: A Survey of Tasks, Datasets, and Methods](https://arxiv.org/abs/1907.09358), arXiv 2019

[Deep Multimodal Representation Learning: A Survey](https://ieeexplore.ieee.org/abstract/document/8715409), arXiv 2019

[Guest Editorial: Image and Language Understanding](https://link.springer.com/article/10.1007/s11263-017-0993-y), IJCV 2017

[Representation Learning: A Review and New Perspectives](https://arxiv.org/abs/1206.5538), TPAMI 2013

[A Survey of Socially Interactive Robots](https://www.cs.cmu.edu/~illah/PAPERS/socialroboticssurvey.pdf), 2003

## Core Areas

### Representation Learning

[Learning Transferable Visual Models From Natural Language Supervision](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language.pdf), arXiv 2020 [[blog]](https://openai.com/blog/clip/) [[code]](https://github.com/openai/CLIP)

[Large-Scale Adversarial Training for Vision-and-Language Representation Learning](https://arxiv.org/abs/2006.06195), NeurIPS 2020 [[code]](https://github.com/zhegan27/VILLA)

[Vokenization: Improving Language Understanding with Contextualized, Visual-Grounded Supervision](https://arxiv.org/abs/2010.06775), EMNLP 2020 [[code]](https://github.com/airsplay/vokenization)

[Integrating Multimodal Information in Large Pretrained Transformers](https://arxiv.org/abs/1908.05787), ACL 2020

[12-in-1: Multi-Task Vision and Language Representation Learning](https://arxiv.org/abs/1912.02315), CVPR 2020 [[code]](https://github.com/facebookresearch/vilbert-multi-task)

[Watching the World Go By: Representation Learning from Unlabeled Videos](https://arxiv.org/abs/2003.07990), arXiv 2020

[Learning Video Representations using Contrastive Bidirectional Transformer](https://arxiv.org/abs/1906.05743), arXiv 2019

[Visual Concept-Metaconcept Learning](https://papers.nips.cc/paper/8745-visual-concept-metaconcept-learning.pdf), NeurIPS 2019 [[code]](http://vcml.csail.mit.edu/)

[VL-BERT: Pre-training of Generic Visual-Linguistic Representations](https://arxiv.org/abs/1908.08530), arXiv 2019 [[code]](https://github.com/jackroos/VL-BERT)

[VisualBERT: A Simple and Performant Baseline for Vision and Language](https://arxiv.org/abs/1908.03557), arXiv 2019 [[code]](https://github.com/uclanlp/visualbert)

[ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks](https://arxiv.org/abs/1908.02265), NeurIPS 2019 [[code]](https://github.com/jiasenlu/vilbert_beta)

[OmniNet: A Unified Architecture for Multi-modal Multi-task Learning](https://arxiv.org/abs/1907.07804), arXiv 2019 [[code]](https://github.com/subho406/OmniNet)

[Learning Representations by Maximizing Mutual Information Across Views](https://arxiv.org/abs/1906.00910), arXiv 2019 [[code]](https://github.com/Philip-Bachman/amdim-public)

[Unicoder-VL: A Universal Encoder for Vision and Language by Cross-modal Pre-training](https://arxiv.org/abs/1908.06066), arXiv 2019

[LXMERT: Learning Cross-Modality Encoder Representations from Transformers](https://arxiv.org/abs/1908.07490), EMNLP 2019 [[code]](https://github.com/airsplay/lxmert)

[ViCo: Word Embeddings from Visual Co-occurrences](https://arxiv.org/abs/1908.08527), ICCV 2019 [[code]](https://github.com/BigRedT/vico)

[M-BERT: Injecting Multimodal Information in the BERT Structure](https://arxiv.org/abs/1908.05787), arXiv 2019

[VideoBERT: A Joint Model for Video and Language Representation Learning](https://arxiv.org/abs/1904.01766), ICCV 2019

[Unified Visual-Semantic Embeddings: Bridging Vision and Language With Structured Meaning Representations](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Unified_Visual-Semantic_Embeddings_Bridging_Vision_and_Language_With_Structured_Meaning_CVPR_2019_paper.pdf), CVPR 2019

[Multi-Task Learning of Hierarchical Vision-Language Representation](https://arxiv.org/abs/1812.00500), CVPR 2019

[Learning Factorized Multimodal Representations](https://arxiv.org/abs/1806.06176), ICLR 2019 [[code]](https://github.com/pliang279/factorized/)

[A Probabilistic Framework for Multi-view Feature Learning with Many-to-many Associations via Neural Networks](https://arxiv.org/abs/1802.04630), ICML 2018

[Do Neural Network Cross-Modal Mappings Really Bridge Modalities?](https://aclweb.org/anthology/P18-2074), ACL 2018

[Learning Robust Visual-Semantic Embeddings](https://arxiv.org/abs/1703.05908), ICCV 2017

[Deep Multimodal Representation Learning from Temporal Data](https://arxiv.org/abs/1704.03152), CVPR 2017

[Is an Image Worth More than a Thousand Words? On the Fine-Grain Semantic Differences between Visual and Linguistic Representations](https://www.aclweb.org/anthology/C16-1264), COLING 2016

[Combining Language and Vision with a Multimodal Skip-gram Model](https://www.aclweb.org/anthology/N15-1016), NAACL 2015

[Deep Fragment Embeddings for Bidirectional Image Sentence Mapping](https://arxiv.org/abs/1406.5679), NIPS 2014

[Multimodal Learning with Deep Boltzmann Machines](https://dl.acm.org/citation.cfm?id=2697059), JMLR 2014

[Learning Grounded Meaning Representations with Autoencoders](https://www.aclweb.org/anthology/P14-1068), ACL 2014

[DeViSE: A Deep Visual-Semantic Embedding Model](https://papers.nips.cc/paper/5204-devise-a-deep-visual-semantic-embedding-model), NeurIPS 2013

[Multimodal Deep Learning](https://dl.acm.org/citation.cfm?id=3104569), ICML 2011

### Multimodal Fusion

[Deep-HOSeq: Deep Higher-Order Sequence Fusion for Multimodal Sentiment Analysis](https://arxiv.org/pdf/2010.08218.pdf), ICDM 2020 

[Removing Bias in Multi-modal Classifiers: Regularization by Maximizing Functional Entropies](https://arxiv.org/abs/2010.10802), NeurIPS 2020 [[code]](https://github.com/itaigat/removing-bias-in-multi-modal-classifiers)

[Deep Multimodal Fusion by Channel Exchanging](https://arxiv.org/abs/2011.05005?context=cs.LG), NeurIPS 2020 [[code]](https://github.com/yikaiw/CEN)

[What Makes Training Multi-Modal Classification Networks Hard?](https://arxiv.org/abs/1905.12681), CVPR 2020

[Dynamic Fusion for Multimodal Data](https://arxiv.org/abs/1911.03821), arXiv 2019

[DeepCU: Integrating Both Common and Unique Latent Information for Multimodal Sentiment Analysis](https://www.ijcai.org/proceedings/2019/503), IJCAI 2019 [[code]](https://github.com/sverma88/DeepCU-IJCAI19)

[Deep Multimodal Multilinear Fusion with High-order Polynomial Pooling](https://papers.nips.cc/paper/9381-deep-multimodal-multilinear-fusion-with-high-order-polynomial-pooling), NeurIPS 2019

[XFlow: Cross-modal Deep Neural Networks for Audiovisual Classification](https://ieeexplore.ieee.org/abstract/document/8894404), IEEE TNNLS 2019 [[code]](https://github.com/catalina17/XFlow)

[MFAS: Multimodal Fusion Architecture Search](https://arxiv.org/abs/1903.06496), CVPR 2019

[The Neuro-Symbolic Concept Learner: Interpreting Scenes, Words, and Sentences From Natural Supervision](https://arxiv.org/abs/1904.12584), ICLR 2019 [[code]](http://nscl.csail.mit.edu/)

[Unifying and merging well-trained deep neural networks for inference stage](https://www.ijcai.org/Proceedings/2018/0283.pdf), IJCAI 2018 [[code]](https://github.com/ivclab/NeuralMerger)

[Efficient Low-rank Multimodal Fusion with Modality-Specific Factors](https://arxiv.org/abs/1806.00064), ACL 2018 [[code]](https://github.com/Justin1904/Low-rank-Multimodal-Fusion)

[Memory Fusion Network for Multi-view Sequential Learning](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/17341/16122), AAAI 2018 [[code]](https://github.com/pliang279/MFN)

[Tensor Fusion Network for Multimodal Sentiment Analysis](https://arxiv.org/abs/1707.07250), EMNLP 2017 [[code]](https://github.com/A2Zadeh/TensorFusionNetwork)

[Jointly Modeling Deep Video and Compositional Text to Bridge Vision and Language in a Unified Framework](http://web.eecs.umich.edu/~jjcorso/pubs/xu_corso_AAAI2015_v2t.pdf), AAAI 2015

### Multimodal Alignment

[CoMIR: Contrastive Multimodal Image Representation for Registration](https://arxiv.org/pdf/2006.06325.pdf), NeurIPS 2020 [[code]](https://github.com/MIDA-group/CoMIR)

[Multimodal Transformer for Unaligned Multimodal Language Sequences](https://arxiv.org/abs/1906.00295), ACL 2019 [[code]](https://github.com/yaohungt/Multimodal-Transformer)

[Temporal Cycle-Consistency Learning](https://arxiv.org/abs/1904.07846), CVPR 2019 [[code]](https://github.com/google-research/google-research/tree/master/tcc)

[See, Hear, and Read: Deep Aligned Representations](https://people.csail.mit.edu/yusuf/see-hear-read/paper.pdf), arXiv 2017

[On Deep Multi-View Representation Learning](http://proceedings.mlr.press/v37/wangb15.pdf), ICML 2015

[Unsupervised Alignment of Natural Language Instructions with Video Segments](https://dl.acm.org/citation.cfm?id=2892753.2892769), AAAI 2014

[Multimodal Alignment of Videos](https://dl.acm.org/citation.cfm?id=2654862), MM 2014

[Deep Canonical Correlation Analysis](http://proceedings.mlr.press/v28/andrew13.html), ICML 2013 [[code]](https://github.com/VahidooX/DeepCCA)

### Multimodal Translation

[Language2Pose: Natural Language Grounded Pose Forecasting](https://arxiv.org/abs/1907.01108), 3DV 2019 [[code]](http://chahuja.com/language2pose/)

[Reconstructing Faces from Voices](https://arxiv.org/abs/1905.10604), NeurIPS 2019 [[code]](https://github.com/cmu-mlsp/reconstructing_faces_from_voices)

[Speech2Face: Learning the Face Behind a Voice](https://arxiv.org/abs/1905.09773), CVPR 2019 [[code]](https://speech2face.github.io/)

[Found in Translation: Learning Robust Joint Representations by Cyclic Translations Between Modalities](https://arxiv.org/abs/1812.07809), AAAI 2019 [[code]](https://github.com/hainow/MCTN)

[Natural TTS Synthesis by Conditioning Wavenet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884), ICASSP 2018 [[code]](https://github.com/NVIDIA/tacotron2)

### Missing or Imperfect Modalities

[Factorized Inference in Deep Markov Models for Incomplete Multimodal Time Series](https://arxiv.org/abs/1905.13570), arXiv 2019

[Learning Representations from Imperfect Time Series Data via Tensor Rank Regularization](https://arxiv.org/abs/1907.01011), ACL 2019

[Multimodal Deep Learning for Robust RGB-D Object Recognition](https://arxiv.org/abs/1507.06821), IROS 2015

### Knowledge Graphs and Knowledge Bases

[MMKG: Multi-Modal Knowledge Graphs](https://arxiv.org/abs/1903.05485), ESWC 2019

[Answering Visual-Relational Queries in Web-Extracted Knowledge Graphs](https://arxiv.org/abs/1709.02314), AKBC 2019

[Embedding Multimodal Relational Data for Knowledge Base Completion](https://arxiv.org/abs/1809.01341), EMNLP 2018

[A Multimodal Translation-Based Approach for Knowledge Graph Representation Learning](https://www.aclweb.org/anthology/S18-2027), SEM 2018 [[code]](https://github.com/UKPLab/starsem18-multimodalKB)

[Order-Embeddings of Images and Language](https://arxiv.org/abs/1511.06361), ICLR 2016 [[code]](https://github.com/ivendrov/order-embedding)

[Building a Large-scale Multimodal Knowledge Base System for Answering Visual Queries](https://arxiv.org/abs/1507.05670), arXiv 2015

### Intepretable Learning

[Multimodal Explanations by Predicting Counterfactuality in Videos](https://arxiv.org/abs/1812.01263), CVPR 2019

[Multimodal Explanations: Justifying Decisions and Pointing to the Evidence](https://arxiv.org/abs/1802.08129), CVPR 2018 [[code]](https://github.com/Seth-Park/MultimodalExplanations)

[Do Explanations make VQA Models more Predictable to a Human?](https://arxiv.org/abs/1810.12366), EMNLP 2018

[Towards Transparent AI Systems: Interpreting Visual Question Answering Models](https://arxiv.org/abs/1608.08974), ICML Workshop on Visualization for Deep Learning 2016

### Generative Learning

[Few-shot Video-to-Video Synthesis](https://arxiv.org/abs/1910.12713), NeurIPS 2019 [[code]](https://nvlabs.github.io/few-shot-vid2vid/)

[Multimodal Generative Models for Scalable Weakly-Supervised Learning](https://arxiv.org/abs/1802.05335), NeurIPS 2018 [[code1]](https://github.com/mhw32/multimodal-vae-public) [[code2]](https://github.com/panpan2/Multimodal-Variational-Autoencoder)

[Look, Imagine and Match: Improving Textual-Visual Cross-Modal Retrieval with Generative Models](https://arxiv.org/abs/1711.06420), CVPR 2018

[The Multi-Entity Variational Autoencoder](http://charlienash.github.io/assets/docs/mevae2017.pdf), NeurIPS 2017

### Semi-supervised Learning

[Semi-supervised Vision-language Mapping via Variational Learning](https://ieeexplore.ieee.org/document/7989160), ICRA 2017

[Semi-supervised Multimodal Hashing](https://arxiv.org/abs/1712.03404), arXiv 2017

[Semi-Supervised Multimodal Deep Learning for RGB-D Object Recognition](https://www.ijcai.org/Proceedings/16/Papers/473.pdf), IJCAI 2016

[Multimodal Semi-supervised Learning for Image Classification](https://ieeexplore.ieee.org/abstract/document/5540120), CVPR 2010

### Self-supervised Learning

[Self-Supervised Learning by Cross-Modal Audio-Video Clustering](https://arxiv.org/abs/1911.12667), NeurIPS 2020 [[code]](https://github.com/HumamAlwassel/XDC)

[Self-Supervised MultiModal Versatile Networks](https://arxiv.org/abs/2006.16228), NeurIPS 2020 [[code]](https://tfhub.dev/deepmind/mmv/s3d/1)

[Labelling Unlabelled Videos from Scratch with Multi-modal Self-supervision](https://arxiv.org/abs/2006.13662), NeurIPS 2020 [[code]](https://www.robots.ox.ac.uk/~vgg/research/selavi/)

[Self-Supervised Learning from Web Data for Multimodal Retrieval](https://arxiv.org/abs/1901.02004), arXiv 2019

[Self-Supervised Learning of Visual Features through Embedding Images into Text Topic Spaces](https://ieeexplore.ieee.org/document/8099701), CVPR 2017

[Multimodal Dynamics : Self-supervised Learning in Perceptual and Motor Systems](https://dl.acm.org/citation.cfm?id=1269207), 2016

### Language Models

[Neural Language Modeling with Visual Features](https://arxiv.org/abs/1903.02930), arXiv 2019

[Learning Multi-Modal Word Representation Grounded in Visual Context](https://arxiv.org/abs/1711.03483), AAAI 2018

[Visual Word2Vec (vis-w2v): Learning Visually Grounded Word Embeddings Using Abstract Scenes](https://arxiv.org/abs/1511.07067), CVPR 2016

[Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models](http://proceedings.mlr.press/v32/kiros14.html), ICML 2014 [[code]](https://github.com/ryankiros/visual-semantic-embedding)

### Adversarial Attacks

[Attend and Attack: Attention Guided Adversarial Attacks on Visual Question Answering Models](https://nips2018vigil.github.io/static/papers/accepted/33.pdf), NeurIPS Workshop on Visually Grounded Interaction and Language 2018

[Attacking Visual Language Grounding with Adversarial Examples: A Case Study on Neural Image Captioning](https://arxiv.org/abs/1712.02051), ACL 2018 [[code]](https://github.com/huanzhang12/ImageCaptioningAttack)

[Fooling Vision and Language Models Despite Localization and Attention Mechanism](https://arxiv.org/abs/1709.08693), CVPR 2018

### Few-Shot Learning

[Language to Network: Conditional Parameter Adaptation with Natural Language Descriptions](https://www.aclweb.org/anthology/2020.acl-main.625/), ACL 2020

[Shaping Visual Representations with Language for Few-shot Classification](https://arxiv.org/abs/1911.02683), ACL 2020

[Zero-Shot Learning - The Good, the Bad and the Ugly](https://arxiv.org/abs/1703.04394), CVPR 2017

[Zero-Shot Learning Through Cross-Modal Transfer](https://nlp.stanford.edu/~socherr/SocherGanjooManningNg_NIPS2013.pdf), NIPS 2013

### Bias and Fairness

[Towards Debiasing Sentence Representations](https://arxiv.org/abs/2007.08100), ACL 2020 [[code]](https://github.com/pliang279/sent_debias)

[FairCVtest Demo: Understanding Bias in Multimodal Learning with a Testbed in Fair Automatic Recruitment](https://arxiv.org/abs/2009.07025), ICMI 2020 [[code]](https://github.com/BiDAlab/FairCVtest)

[Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993), FAccT 2019 

[Black is to Criminal as Caucasian is to Police: Detecting and Removing Multiclass Bias in Word Embeddings](https://arxiv.org/abs/1904.04047), NAACL 2019 [[code]](https://github.com/TManzini/DebiasMulticlassWordEmbedding)

[Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification](http://proceedings.mlr.press/v81/buolamwini18a.html?mod=article_inline), FAccT 2018

[Datasheets for Datasets](https://arxiv.org/abs/1803.09010), arXiv 2018

[Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://arxiv.org/abs/1607.06520), NeurIPS 2016

### Human in the Loop Learning

[Human in the Loop Dialogue Systems](https://sites.google.com/view/hlds-2020/home), NeurIPS 2020 workshop

[Human And Machine in-the-Loop Evaluation and Learning Strategies](https://hamlets-workshop.github.io/), NeurIPS 2020 workshop

[Human-centric dialog training via offline reinforcement learning](https://arxiv.org/abs/2010.05848), EMNLP 2020 [[code]](https://github.com/natashamjaques/neural_chat/tree/master/BatchRL)

[Human-In-The-Loop Machine Learning with Intelligent Multimodal Interfaces](https://csjzhou.github.io/homepage/papers/ICML2017_Syed.pdf), ICML 2017 workshop

## Applications and Datasets

### Language and Visual QA

[MultiModalQA: complex question answering over text, tables and images](https://openreview.net/forum?id=ee6W5UgQLa), ICLR 2021

[ManyModalQA: Modality Disambiguation and QA over Diverse Inputs](https://arxiv.org/abs/2001.08034), AAAI 2020 [[code]](https://github.com/hannandarryl/ManyModalQA)

[Iterative Answer Prediction with Pointer-Augmented Multimodal Transformers for TextVQA](https://arxiv.org/abs/1911.06258), CVPR 2020

[Interactive Language Learning by Question Answering](https://arxiv.org/abs/1908.10909), EMNLP 2019 [[code]](https://github.com/xingdi-eric-yuan/qait_public)

[Fusion of Detected Objects in Text for Visual Question Answering](https://arxiv.org/abs/1908.05054), arXiv 2019 

[RUBi: Reducing Unimodal Biases in Visual Question Answering](https://arxiv.org/abs/1906.10169), NeurIPS 2019 [[code]](https://github.com/cdancette/rubi.bootstrap.pytorch)

[GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering](https://arxiv.org/abs/1902.09506), CVPR 2019 [[code]](https://cs.stanford.edu/people/dorarad/gqa/)

[OK-VQA: A Visual Question Answering Benchmark Requiring External Knowledge](https://arxiv.org/abs/1906.00067), CVPR 2019 [[code]](http://okvqa.allenai.org/)

[MUREL: Multimodal Relational Reasoning for Visual Question Answering](https://arxiv.org/abs/1902.09487), CVPR 2019 [[code]](https://github.com/Cadene/murel.bootstrap.pytorch)

[Social-IQ: A Question Answering Benchmark for Artificial Social Intelligence](http://openaccess.thecvf.com/content_CVPR_2019/html/Zadeh_Social-IQ_A_Question_Answering_Benchmark_for_Artificial_Social_Intelligence_CVPR_2019_paper.html), CVPR 2019 [[code]](https://github.com/A2Zadeh/Social-IQ)

[Probabilistic Neural-symbolic Models for Interpretable Visual Question Answering](https://arxiv.org/abs/1902.07864), ICML 2019 [[code]](https://github.com/kdexd/probnmn-clevr)

[Learning to Count Objects in Natural Images for Visual Question Answering](https://arxiv.org/abs/1802.05766), ICLR 2018, [[code]](https://github.com/Cyanogenoid/vqa-counting)

[Overcoming Language Priors in Visual Question Answering with Adversarial Regularization](https://arxiv.org/abs/1810.03649), NeurIPS 2018

[Neural-Symbolic VQA: Disentangling Reasoning from Vision and Language Understanding](https://arxiv.org/abs/1810.02338), NeurIPS 2018 [[code]](https://github.com/kexinyi/ns-vqa)

[RecipeQA: A Challenge Dataset for Multimodal Comprehension of Cooking Recipes](https://arxiv.org/abs/1809.00812), EMNLP 2018 [[code]](https://hucvl.github.io/recipeqa/)

[TVQA: Localized, Compositional Video Question Answering](https://www.aclweb.org/anthology/D18-1167), EMNLP 2018 [[code]](https://github.com/jayleicn/TVQA)

[Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998), CVPR 2018 [[code]](https://github.com/facebookresearch/pythia)

[Don't Just Assume; Look and Answer: Overcoming Priors for Visual Question Answering](https://arxiv.org/abs/1712.00377), CVPR 2018 [[code]](https://github.com/AishwaryaAgrawal/GVQA)

[Stacked Latent Attention for Multimodal Reasoning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Fan_Stacked_Latent_Attention_CVPR_2018_paper.pdf), CVPR 2018

[Learning to Reason: End-to-End Module Networks for Visual Question Answering](https://arxiv.org/abs/1704.05526), ICCV 2017 [[code]](https://github.com/ronghanghu/n2nmn)

[CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning](https://arxiv.org/abs/1612.06890), CVPR 2017 [[code]](https://github.com/facebookresearch/clevr-iep) [[dataset generation]](https://github.com/facebookresearch/clevr-dataset-gen)

[Are You Smarter Than A Sixth Grader? Textbook Question Answering for Multimodal Machine Comprehension](https://ieeexplore.ieee.org/document/8100054/), CVPR 2017 [[code]](http://vuchallenge.org/tqa.html)

[Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding](https://arxiv.org/abs/1606.01847), EMNLP 2016 [[code]](https://github.com/akirafukui/vqa-mcb)

[MovieQA: Understanding Stories in Movies through Question-Answering](https://arxiv.org/abs/1512.02902), CVPR 2016 [[code]](http://movieqa.cs.toronto.edu/home/)

[VQA: Visual Question Answering](https://arxiv.org/abs/1505.00468), ICCV 2015 [[code]](https://visualqa.org/)

### Language Grounding in Vision

[The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes](https://arxiv.org/abs/2005.04790), NeurIPS 2020 [[code]](https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set/)

[What Does BERT with Vision Look At?](https://www.aclweb.org/anthology/2020.acl-main.469/), ACL 2020

[Visual Grounding in Video for Unsupervised Word Translation](https://arxiv.org/abs/2003.05078), CVPR 2020 [[code]](https://github.com/gsig/visual-grounding)

[VIOLIN: A Large-Scale Dataset for Video-and-Language Inference](https://arxiv.org/abs/2003.11618), CVPR 2020 [[code]](https://github.com/jimmy646/violin)

[Grounded Video Description](https://arxiv.org/abs/1812.06587), CVPR 2019

[Show, Control and Tell: A Framework for Generating Controllable and Grounded Captions](https://arxiv.org/abs/1811.10652), CVPR 2019

[Multilevel Language and Vision Integration for Text-to-Clip Retrieval](https://arxiv.org/abs/1804.05113), AAAI 2019 [[code]](https://github.com/VisionLearningGroup/Text-to-Clip_Retrieval)

[Binary Image Selection (BISON): Interpretable Evaluation of Visual Grounding](https://arxiv.org/abs/1901.06595), arXiv 2019 [[code]](https://github.com/facebookresearch/binary-image-selection)

[Finding “It”: Weakly-Supervised Reference-Aware Visual Grounding in Instructional Videos](http://openaccess.thecvf.com/content_cvpr_2018/papers/Huang_Finding_It_Weakly-Supervised_CVPR_2018_paper.pdf), CVPR 2018

[SCAN: Learning Hierarchical Compositional Visual Concepts](https://arxiv.org/abs/1707.03389), ICLR 2018

[Visual Coreference Resolution in Visual Dialog using Neural Module Networks](https://arxiv.org/abs/1809.01816), ECCV 2018 [[code]](https://github.com/facebookresearch/corefnmn)

[Gated-Attention Architectures for Task-Oriented Language Grounding](https://arxiv.org/abs/1706.07230), AAAI 2018 [[code]](https://github.com/devendrachaplot/DeepRL-Grounding)

[Using Syntax to Ground Referring Expressions in Natural Images](https://arxiv.org/abs/1805.10547), AAAI 2018 [[code]](https://github.com/volkancirik/groundnet)

[Grounding language acquisition by training semantic parsers using captioned videos](https://cbmm.mit.edu/sites/default/files/publications/Ross-et-al_ACL2018_Grounding%20language%20acquisition%20by%20training%20semantic%20parsing%20using%20caption%20videos.pdf), ACL 2018

[Interpretable and Globally Optimal Prediction for Textual Grounding using Image Concepts](https://arxiv.org/abs/1803.11209), NeurIPS 2017

[Localizing Moments in Video with Natural Language](https://arxiv.org/abs/1708.01641), ICCV 2017

[What are you talking about? Text-to-Image Coreference](https://ieeexplore.ieee.org/abstract/document/6909850/), CVPR 2014

[Grounded Language Learning from Video Described with Sentences](https://www.aclweb.org/anthology/P13-1006), ACL 2013

[Grounded Compositional Semantics for Finding and Describing Images with Sentences](https://nlp.stanford.edu/~socherr/SocherKarpathyLeManningNg_TACL2013.pdf), TACL 2013

### Language Grouding in Navigation

[Improving Vision-and-Language Navigation with Image-Text Pairs from the Web](https://arxiv.org/abs/2004.14973), ECCV 2020

[Towards Learning a Generic Agent for Vision-and-Language Navigation via Pre-training](https://arxiv.org/abs/2002.10638), CVPR 2020 [[code]](https://github.com/weituo12321/PREVALENT)

[VideoNavQA: Bridging the Gap between Visual and Embodied Question Answering](https://arxiv.org/abs/1908.04950), BMVC 2019 [[code]](https://github.com/catalina17/VideoNavQA)

[Vision-and-Dialog Navigation](https://arxiv.org/abs/1907.04957), arXiv 2019 [[code]](https://github.com/mmurray/cvdn)

[Hierarchical Decision Making by Generating and Following Natural Language Instructions](https://arxiv.org/abs/1906.00744), arXiv 2019 [[code]](https://www.minirts.net/)

[Stay on the Path: Instruction Fidelity in Vision-and-Language Navigation](https://arxiv.org/abs/1905.12255), ACL 2019

[Are You Looking? Grounding to Multiple Modalities in Vision-and-Language Navigation](https://arxiv.org/abs/1906.00347), ACL 2019

[Touchdown: Natural Language Navigation and Spatial Reasoning in Visual Street Environments](https://arxiv.org/abs/1811.12354), CVPR 2019 [[code]](https://github.com/lil-lab/touchdown)

[Reinforced Cross-Modal Matching and Self-Supervised Imitation Learning for Vision-Language Navigation](https://arxiv.org/abs/1811.10092), CVPR 2019

[The Regretful Navigation Agent for Vision-and-Language Navigation](https://arxiv.org/abs/1903.01602), CVPR 2019 [[code]](https://github.com/chihyaoma/regretful-agent)

[Tactical Rewind: Self-Correction via Backtracking in Vision-and-Language Navigation](https://arxiv.org/abs/1903.02547), CVPR 2019 [[code]](https://github.com/Kelym/FAST)

[Multi-modal Discriminative Model for Vision-and-Language Navigation](https://www.aclweb.org/anthology/W19-1605), NAACL SpLU-RoboNLP Workshop 2019

[Self-Monitoring Navigation Agent via Auxiliary Progress Estimation](https://arxiv.org/abs/1901.03035), ICLR 2019 [[code]](https://github.com/chihyaoma/selfmonitoring-agent)

[From Language to Goals: Inverse Reinforcement Learning for Vision-Based Instruction Following](https://arxiv.org/abs/1902.07742), ICLR 2019

[Read, Watch, and Move: Reinforcement Learning for Temporally Grounding Natural Language Descriptions in Videos](https://arxiv.org/abs/1901.06829), AAAI 2019

[Learning to Navigate Unseen Environments: Back Translation with Environmental Dropout](https://www.aclweb.org/anthology/N19-1268), NAACL 2019 [[code]](https://github.com/airsplay/R2R-EnvDrop)

[Attention Based Natural Language Grounding by Navigating Virtual Environment](https://arxiv.org/abs/1804.08454), IEEE WACV 2019

[Mapping Instructions to Actions in 3D Environments with Visual Goal Prediction](https://arxiv.org/abs/1809.00786), EMNLP 2018 [[code]](https://github.com/lil-lab/ciff)

[Vision-and-Language Navigation: Interpreting Visually-Grounded Navigation Instructions in Real Environments](https://arxiv.org/abs/1711.07280), CVPR 2018 [[code]](https://bringmeaspoon.org/)

[Embodied Question Answering](https://arxiv.org/abs/1711.11543), CVPR 2018 [[code]](https://embodiedqa.org/)

[Look Before You Leap: Bridging Model-Free and Model-Based Reinforcement Learning for Planned-Ahead Vision-and-Language Navigation](https://arxiv.org/abs/1803.07729), ECCV 2018

### Multimodal Machine Translation

[Unsupervised Multimodal Neural Machine Translation with Pseudo Visual Pivoting](https://arxiv.org/abs/2005.03119), ACL 2020

[Multimodal Transformer for Multimodal Machine Translation](https://www.aclweb.org/anthology/2020.acl-main.400/), ACL 2020

[Neural Machine Translation with Universal Visual Representation](https://openreview.net/forum?id=Byl8hhNYPS), ICLR 2020 [[code]](https://github.com/cooelf/UVR-NMT)

[Visual Agreement Regularized Training for Multi-Modal Machine Translation](https://arxiv.org/abs/1912.12014), AAAI 2020

[VATEX: A Large-Scale, High-Quality Multilingual Dataset for Video-and-Language Research](https://arxiv.org/abs/1904.03493), ICCV 2019 [[code]](http://vatex.org/main/index.html)

[Latent Variable Model for Multi-modal Translation](https://arxiv.org/pdf/1811.00357), ACL 2019

[Distilling Translations with Visual Awareness](https://arxiv.org/pdf/1906.07701), ACL 2019

[Probing the Need for Visual Context in Multimodal Machine Translation](https://www.aclweb.org/anthology/N19-1422), NAACL 2019

[Emergent Translation in Multi-Agent Communication](https://openreview.net/pdf?id=H1vEXaxA-), ICLR 2018

[Zero-Resource Neural Machine Translation with Multi-Agent Communication Game](https://arxiv.org/pdf/1802.03116), AAAI 2018

[Learning Translations via Images with a Massively Multilingual Image Dataset](http://aclweb.org/anthology/P18-1239), ACL 2018

[A Visual Attention Grounding Neural Model for Multimodal Machine Translation](http://aclweb.org/anthology/D18-1400), EMNLP 2018

[Adversarial Evaluation of Multimodal Machine Translation](http://aclweb.org/anthology/D18-1329), EMNLP 2018

[Doubly-Attentive Decoder for Multi-modal Neural Machine Translation](http://aclweb.org/anthology/P17-1175), ACL 2017 [[code]](https://github.com/iacercalixto/MultimodalNMT)

[An empirical study on the effectiveness of images in Multimodal Neural Machine Translation](http://aclweb.org/anthology/D17-1095), EMNLP 2017

[Incorporating Global Visual Features into Attention-based Neural Machine Translation](http://aclweb.org/anthology/D17-1105), EMNLP 2017 [[code]](https://github.com/iacercalixto/MultimodalNMT)

[Multimodal Pivots for Image Caption Translation](http://aclweb.org/anthology/P16-1227), ACL 2016

[Multi30K: Multilingual English-German Image Descriptions](https://aclweb.org/anthology/W16-3210.pdf), ACL Workshop on Language and Vision 2016 [[code]](https://github.com/multi30k/dataset)

[Does Multimodality Help Human and Machine for Translation and Image Captioning?](http://www.statmt.org/wmt16/pdf/W16-2358.pdf), ACL WMT 2016

### Multi-agent Communication

[Multi-agent Communication meets Natural Language: Synergies between Functional and Structural Language Learning](https://arxiv.org/abs/2005.07064), ACL 2020

[Emergence of Compositional Language with Deep Generational Transmission](https://arxiv.org/abs/1904.09067), ICML 2019

[On the Pitfalls of Measuring Emergent Communication](https://arxiv.org/abs/1903.05168), AAMAS 2019 [[code]](https://github.com/facebookresearch/measuring-emergent-comm)

[Emergent Translation in Multi-Agent Communication](https://arxiv.org/abs/1710.06922), ICLR 2018 [[code]](https://github.com/facebookresearch/translagent)

[Emergent Communication in a Multi-Modal, Multi-Step Referential Game](https://openreview.net/pdf?id=rJGZq6g0-), ICLR 2018 [[code]](https://github.com/nyu-dl/MultimodalGame)

[Emergence of Linguistic Communication From Referential Games with Symbolic and Pixel Input](https://openreview.net/pdf?id=HJGv1Z-AW), ICLR 2018

[Emergent Communication through Negotiation](https://openreview.net/pdf?id=Hk6WhagRW), ICLR 2018 [[code]](https://github.com/ASAPPinc/emergent_comms_negotiation)

[Emergence of Grounded Compositional Language in Multi-Agent Populations](https://arxiv.org/abs/1703.04908), AAAI 2018

[Emergence of Language with Multi-agent Games: Learning to Communicate with Sequences of Symbols](https://arxiv.org/abs/1705.11192), NeurIPS 2017

[Natural Language Does Not Emerge 'Naturally' in Multi-Agent Dialog](https://arxiv.org/abs/1706.08502), EMNLP 2017 [[code1]](https://github.com/batra-mlp-lab/lang-emerge) [[code2]](https://github.com/kdexd/lang-emerge-parlai)

[Learning Cooperative Visual Dialog Agents with Deep Reinforcement Learning](https://arxiv.org/abs/1703.06585), ICCV 2017 [code](https://github.com/batra-mlp-lab/visdial-rl)

[Multi-agent Cooperation and the Emergence of (natural) Language](https://arxiv.org/abs/1612.07182), ICLR 2017

[Learning to Communicate with Deep Multi-agent Reinforcement Learning](https://arxiv.org/abs/1605.06676), NIPS 2016.

[Learning multiagent communication with backpropagation](http://papers.nips.cc/paper/6398-learning-multiagent-communication-with-backpropagation.pdf), NIPS 2016.

[The Emergence of Compositional Structures in Perceptually Grounded Language Games](https://www.cs.utexas.edu/~kuipers/readings/Vogt-aij-05.pdf), AI 2005

### Commonsense Reasoning

[Adventures in Flatland: Perceiving Social Interactions Under Physical Dynamics](https://www.tshu.io/HeiderSimmel/CogSci20/Flatland_CogSci20.pdf), CogSci 2020

[A Logical Model for Supporting Social Commonsense Knowledge Acquisition](https://arxiv.org/abs/1912.11599), arXiv 2019

[Heterogeneous Graph Learning for Visual Commonsense Reasoning](https://arxiv.org/abs/1910.11475), NeurIPS 2019

[SocialIQA: Commonsense Reasoning about Social Interactions](https://arxiv.org/abs/1904.09728), arXiv 2019

[From Recognition to Cognition: Visual Commonsense Reasoning](https://arxiv.org/abs/1811.10830), CVPR 2019 [[code]](https://visualcommonsense.com/)

[CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge](https://arxiv.org/abs/1811.00937), NAACL 2019

### Multimodal Reinforcement Learning

[Imitating Interactive Intelligence](https://arxiv.org/abs/2012.05672), arXiv 2020

[Grounded Language Learning Fast and Slow](https://arxiv.org/abs/2009.01719), ICLR 2021

[RTFM: Generalising to Novel Environment Dynamics via Reading](https://arxiv.org/abs/1910.08210), ICLR 2020 [[code]](https://github.com/facebookresearch/RTFM)

[Embodied Multimodal Multitask Learning](https://arxiv.org/abs/1902.01385), IJCAI 2020

[Learning to Speak and Act in a Fantasy Text Adventure Game](https://arxiv.org/abs/1903.03094), arXiv 2019 [[code]](https://parl.ai/projects/light/)

[Language as an Abstraction for Hierarchical Deep Reinforcement Learning](https://arxiv.org/abs/1906.07343), NeurIPS 2019

[Hierarchical Decision Making by Generating and Following Natural Language Instructions](https://arxiv.org/abs/1906.00744), NeurIPS 2019 [[code]](https://github.com/facebookresearch/minirts)

[Habitat: A Platform for Embodied AI Research](https://arxiv.org/abs/1904.01201), ICCV 2019 [[code]](https://aihabitat.org/)

[Multimodal Hierarchical Reinforcement Learning Policy for Task-Oriented Visual Dialog](https://arxiv.org/abs/1805.03257), SIGDIAL 2018

[Mapping Instructions and Visual Observations to Actions with Reinforcement Learning](https://www.cs.cornell.edu/~dkm/papers/mla-emnlp.2017.pdf), EMNLP 2017

[Reinforcement Learning for Mapping Instructions to Actions](https://people.csail.mit.edu/regina/my_papers/RL.pdf), ACL 2009

### Multimodal Dialog

[Two Causal Principles for Improving Visual Dialog](https://arxiv.org/abs/1911.10496), CVPR 2020

[MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations](https://arxiv.org/abs/1810.02508), ACL 2019 [[code]](http://affective-meld.github.io/)

[CLEVR-Dialog: A Diagnostic Dataset for Multi-Round Reasoning in Visual Dialog](https://www.aclweb.org/anthology/N19-1058), NAACL 2019 [[code]](https://github.com/satwikkottur/clevr-dialog)

[Talk the Walk: Navigating New York City through Grounded Dialogue](https://arxiv.org/abs/1807.03367), arXiv 2018

[Dialog-based Interactive Image Retrieval](https://arxiv.org/abs/1805.00145), NeurIPS 2018 [[code]](https://github.com/XiaoxiaoGuo/fashion-retrieval)

[Towards Building Large Scale Multimodal Domain-Aware Conversation Systems](https://arxiv.org/abs/1704.00200), arXiv 2017 [[code]](https://amritasaha1812.github.io/MMD/)

[Visual Dialog](https://arxiv.org/abs/1611.08669), CVPR 2017 [[code]](https://github.com/batra-mlp-lab/visdial)

### Language and Audio

[Lattice Transformer for Speech Translation](https://arxiv.org/abs/1906.05551), ACL 2019

[Exploring Phoneme-Level Speech Representations for End-to-End Speech Translation](https://arxiv.org/abs/1906.01199), ACL 2019

[Audio Caption: Listen and Tell](https://arxiv.org/abs/1902.09254), ICASSP 2019

[Audio-Linguistic Embeddings for Spoken Sentences](https://arxiv.org/abs/1902.07817), ICASSP 2019

[From Semi-supervised to Almost-unsupervised Speech Recognition with Very-low Resource by Jointly Learning Phonetic Structures from Audio and Text Embeddings](https://arxiv.org/abs/1904.05078), arXiv 2019

[From Audio to Semantics: Approaches To End-to-end Spoken Language Understanding](https://arxiv.org/abs/1809.09190), arXiv 2018

[Natural TTS Synthesis by Conditioning Wavenet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884), ICASSP 2018 [[code]](https://github.com/NVIDIA/tacotron2)

[Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning](https://arxiv.org/abs/1710.07654), ICLR 2018

[Deep Voice 2: Multi-Speaker Neural Text-to-Speech](https://arxiv.org/abs/1705.08947), NeurIPS 2017

[Deep Voice: Real-time Neural Text-to-Speech](https://arxiv.org/abs/1702.07825), ICML 2017

[Text-to-Speech Synthesis](https://dl.acm.org/citation.cfm?id=1592988), 2009

### Audio and Visual

[Music Gesture for Visual Sound Separation](https://arxiv.org/abs/2004.09476), CVPR 2020

[Co-Compressing and Unifying Deep CNN Models for Efficient Human Face and Speaker Recognition](http://openaccess.thecvf.com/content_CVPRW_2019/papers/MULA/Wan_Co-Compressing_and_Unifying_Deep_CNN_Models_for_Efficient_Human_Face_CVPRW_2019_paper.pdf), CVPRW 2019

[Learning Individual Styles of Conversational Gesture](https://arxiv.org/abs/1906.04160), CVPR 2019 [[code]](http://people.eecs.berkeley.edu/~shiry/speech2gesture)

[Capture, Learning, and Synthesis of 3D Speaking Styles](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/510/paper_final.pdf), CVPR 2019 [[code]](https://github.com/TimoBolkart/voca)

[Disjoint Mapping Network for Cross-modal Matching of Voices and Faces](https://arxiv.org/abs/1807.04836), ICLR 2019

[Wav2Pix: Speech-conditioned Face Generation using Generative Adversarial Networks](https://arxiv.org/abs/1903.10195), ICASSP 2019 [[code]](https://imatge-upc.github.io/wav2pix/)

[Jointly Discovering Visual Objects and Spoken Words from Raw Sensory Input](https://arxiv.org/abs/1804.01452), ECCV 2018 [[code]](https://github.com/LiqunChen0606/Jointly-Discovering-Visual-Objects-and-Spoken-Words)

[Seeing Voices and Hearing Faces: Cross-modal Biometric Matching](https://arxiv.org/abs/1804.00326), CVPR 2018 [[code]](https://github.com/a-nagrani/SVHF-Net)

[Learning to Separate Object Sounds by Watching Unlabeled Video](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w49/Gao_Learning_to_Separate_CVPR_2018_paper.pdf), CVPR 2018

[Deep Audio-Visual Speech Recognition](https://arxiv.org/abs/1809.02108), IEEE TPAMI 2018

[Look, Listen and Learn](http://openaccess.thecvf.com/content_ICCV_2017/papers/Arandjelovic_Look_Listen_and_ICCV_2017_paper.pdf), ICCV 2017

[Unsupervised Learning of Spoken Language with Visual Context](https://papers.nips.cc/paper/6186-unsupervised-learning-of-spoken-language-with-visual-context.pdf), NeurIPS 2016

[SoundNet: Learning Sound Representations from Unlabeled Video](https://arxiv.org/abs/1610.09001), NeurIPS 2016 [[code]](http://projects.csail.mit.edu/soundnet/)

### Media Description

[Towards Unsupervised Image Captioning with Shared Multimodal Embeddings](https://arxiv.org/abs/1908.09317), ICCV 2019

[Video Relationship Reasoning using Gated Spatio-Temporal Energy Graph](https://arxiv.org/abs/1903.10547), CVPR 2019 [[code]](https://github.com/yaohungt/GSTEG_CVPR_2019)

[Joint Event Detection and Description in Continuous Video Streams](https://arxiv.org/abs/1802.10250), WACVW 2019

[Learning to Compose and Reason with Language Tree Structures for Visual Grounding](https://arxiv.org/abs/1906.01784), TPAMI 2019

[Neural Baby Talk](https://arxiv.org/abs/1803.09845), CVPR 2018 [[code]](https://github.com/jiasenlu/NeuralBabyTalk)

[Grounding Referring Expressions in Images by Variational Context](https://arxiv.org/abs/1712.01892), CVPR 2018

[Video Captioning via Hierarchical Reinforcement Learning](https://arxiv.org/abs/1711.11135), CVPR 2018

[Charades-Ego: A Large-Scale Dataset of Paired Third and First Person Videos](https://arxiv.org/abs/1804.09626), CVPR 2018 [[code]](https://allenai.org/plato/charades/) 

[Neural Motifs: Scene Graph Parsing with Global Context](https://arxiv.org/abs/1711.06640), CVPR 2018 [[code]](http://github.com/rowanz/neural-motifs)

[No Metrics Are Perfect: Adversarial Reward Learning for Visual Storytelling](https://arxiv.org/abs/1804.09160), ACL 2018

[Generating Descriptions with Grounded and Co-Referenced People](https://arxiv.org/abs/1704.01518), CVPR 2017

[DenseCap: Fully Convolutional Localization Networks for Dense Captioning](https://cs.stanford.edu/people/karpathy/densecap/), CVPR 2016

[Review Networks for Caption Generation](https://arxiv.org/abs/1605.07912), NeurIPS 2016 [[code]](https://github.com/kimiyoung/review_net)

[Hollywood in Homes: Crowdsourcing Data Collection for Activity Understanding](https://arxiv.org/abs/1604.01753), ECCV 2016 [[code]](https://allenai.org/plato/charades/)

[Show and Tell: Lessons learned from the 2015 MSCOCO Image Captioning Challenge](https://arxiv.org/abs/1609.06647), TPAMI 2016 [[code]](https://github.com/tensorflow/models/tree/master/research/im2txt)

[Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044), ICML 2015 [[code]](https://github.com/kelvinxu/arctic-captions)

[Deep Visual-Semantic Alignments for Generating Image Descriptions](https://arxiv.org/abs/1412.2306v2), CVPR 2015 [[code]](https://github.com/karpathy/neuraltalk2)

[Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555), CVPR 2015 [[code]](https://github.com/karpathy/neuraltalk2)

[A Dataset for Movie Description](https://arxiv.org/abs/1501.02530), CVPR 2015 [[code]](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/vision-and-language/mpii-movie-description-dataset/)

[What’s Cookin’? Interpreting Cooking Videos using Text, Speech and Vision](https://arxiv.org/abs/1503.01558), NAACL 2015 [[code]](https://github.com/malmaud/whats_cookin)

[Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312), ECCV 2014 [[code]](http://cocodataset.org/#home)

### Video Generation from Text

[Image Generation from Scene Graphs](https://arxiv.org/abs/1804.01622), CVPR 2018

[Learning to Color from Language](https://arxiv.org/abs/1804.06026), NAACL 2018

[Generative Adversarial Text to Image Synthesis](https://arxiv.org/abs/1605.05396), ICML 2016

### Affect Recognition and Multimodal Language

[End-to-end Facial and Physiological Model for Affective Computing and Applications](https://arxiv.org/abs/1912.04711), arXiv 2019

[Affective Computing for Large-Scale Heterogeneous Multimedia Data: A Survey](https://arxiv.org/abs/1911.05609), ACM TOMM 2019

[Towards Multimodal Sarcasm Detection (An Obviously_Perfect Paper)](https://arxiv.org/abs/1906.01815), ACL 2019 [[code]](https://github.com/soujanyaporia/MUStARD)

[Multi-modal Approach for Affective Computing](https://arxiv.org/abs/1804.09452), EMBC 2018

[Multimodal Language Analysis with Recurrent Multistage Fusion](https://arxiv.org/abs/1808.03920), EMNLP 2018

[Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph](http://aclweb.org/anthology/P18-1208), ACL 2018 [[code]](https://github.com/A2Zadeh/CMU-MultimodalSDK)

[Multi-attention Recurrent Network for Human Communication Comprehension](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/17390/16123), AAAI 2018 [[code]](https://github.com/A2Zadeh/CMU-MultimodalSDK)

[End-to-End Multimodal Emotion Recognition using Deep Neural Networks](https://arxiv.org/abs/1704.08619), arXiv 2017

[AMHUSE - A Multimodal dataset for HUmor SEnsing](https://dl.acm.org/citation.cfm?id=3136806), ICMI 2017 [[code]](http://amhuse.phuselab.di.unimi.it/)

[Decoding Children’s Social Behavior](http://www.cbi.gatech.edu/mmdb/docs/mmdb_paper.pdf), CVPR 2013 [[code]](http://www.cbi.gatech.edu/mmdb/)

[Collecting Large, Richly Annotated Facial-Expression Databases from Movies](http://users.cecs.anu.edu.au/%7Eadhall/Dhall_Goecke_Lucey_Gedeon_M_2012.pdf), IEEE Multimedia 2012 [[code]](https://cs.anu.edu.au/few/AFEW.html)

[The Interactive Emotional Dyadic Motion Capture (IEMOCAP) Database](https://sail.usc.edu/iemocap/Busso_2008_iemocap.pdf), 2008 [[code]](https://sail.usc.edu/iemocap/)

### Healthcare

[Leveraging Medical Visual Question Answering with Supporting Facts](https://arxiv.org/abs/1905.12008), arXiv 2019

[Unsupervised Multimodal Representation Learning across Medical Images and Reports](https://arxiv.org/abs/1811.08615), ML4H 2018

[Multimodal Medical Image Retrieval based on Latent Topic Modeling](https://aiforsocialgood.github.io/2018/pdfs/track1/75_aisg_neurips2018.pdf), ML4H 2018

[Improving Hospital Mortality Prediction with Medical Named Entities and Multimodal Learning](https://arxiv.org/abs/1811.12276), ML4H 2018

[Knowledge-driven Generative Subspaces for Modeling Multi-view Dependencies in Medical Data](https://arxiv.org/abs/1812.00509), ML4H 2018

[Multimodal Depression Detection: Fusion Analysis of Paralinguistic, Head Pose and Eye Gaze Behaviors](https://ieeexplore.ieee.org/document/7763752), TAC 2018

[Learning the Joint Representation of Heterogeneous Temporal Events for Clinical Endpoint Prediction](https://arxiv.org/abs/1803.04837), AAAI 2018

[Understanding Coagulopathy using Multi-view Data in the Presence of Sub-Cohorts: A Hierarchical Subspace Approach](http://mucmd.org/CameraReadySubmissions/67%5CCameraReadySubmission%5Cunderstanding-coagulopathy-multi%20(6).pdf), MLHC 2017

[Machine Learning in Multimodal Medical Imaging](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5357511/), 2017

[Cross-modal Recurrent Models for Weight Objective Prediction from Multimodal Time-series Data](https://arxiv.org/abs/1709.08073), ML4H 2017

[SimSensei Kiosk: A Virtual Human Interviewer for Healthcare Decision Support](https://dl.acm.org/citation.cfm?id=2617388.2617415), AAMAS 2014

[Dyadic Behavior Analysis in Depression Severity Assessment Interviews](https://dl.acm.org/citation.cfm?doid=2663204.2663238), ICMI 2014

[Audiovisual Behavior Descriptors for Depression Assessment](https://dl.acm.org/citation.cfm?doid=2522848.2522886), ICMI 2013

### Robotics

[Detect, Reject, Correct: Crossmodal Compensation of Corrupted Sensors](https://arxiv.org/abs/2012.00201), 2020

[Concept2Robot: Learning Manipulation Concepts from Instructions and Human Demonstrations](http://www.roboticsproceedings.org/rss16/p082.pdf), RSS 2020

[See, Feel, Act: Hierarchical Learning for Complex Manipulation Skills with Multi-sensory Fusion](https://robotics.sciencemag.org/content/4/26/eaav3123), Science Robotics 2019 

[Early Fusion for Goal Directed Robotic Vision](https://arxiv.org/abs/1811.08824), IROS 2019

[Simultaneously Learning Vision and Feature-based Control Policies for Real-world Ball-in-a-Cup](https://arxiv.org/abs/1902.04706), RSS 2019

[Probabilistic Multimodal Modeling for Human-Robot Interaction Tasks](http://www.roboticsproceedings.org/rss15/p47.pdf), RSS 2019

[Making Sense of Vision and Touch: Self-Supervised Learning of Multimodal Representations for Contact-Rich Tasks](https://arxiv.org/abs/1810.10191), ICRA 2019

[Evolving Multimodal Robot Behavior via Many Stepping Stones with the Combinatorial Multi-Objective Evolutionary Algorithm
](https://arxiv.org/abs/1807.03392), arXiv 2018

[Multi-modal Predicate Identification using Dynamically Learned Robot Controllers](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/IJCAI18-saeid.pdf), IJCAI 2018

[Multimodal Probabilistic Model-Based Planning for Human-Robot Interaction](https://arxiv.org/abs/1710.09483), arXiv 2017 

[Perching and Vertical Climbing: Design of a Multimodal Robot](https://ieeexplore.ieee.org/document/6907472), ICRA 2014

[Multi-Modal Scene Understanding for Robotic Grasping](http://kth.diva-portal.org/smash/get/diva2:459199/FULLTEXT01), 2011

[Strategies for Multi-Modal Scene Exploration](https://am.is.tuebingen.mpg.de/uploads_file/attachment/attachment/307/2010_IROS_bjbk_camred.pdf), IROS 2010

### Autonomous Driving

[Deep Multi-modal Object Detection and Semantic Segmentation for Autonomous Driving: Datasets, Methods, and Challenges](https://arxiv.org/pdf/1902.07830.pdf), IEEE TITS 2020 [[website]](https://boschresearch.github.io/multimodalperception/) 

[nuScenes: A multimodal dataset for autonomous driving](https://openaccess.thecvf.com/content_CVPR_2020/papers/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.pdf), CVPR 2020 [[dataset]](https://www.nuscenes.org/)

[Multimodal End-to-End Autonomous Driving](https://arxiv.org/abs/1906.03199), arXiv 2020

### Finance

[A Multimodal Event-driven LSTM Model for Stock Prediction Using Online News](https://ailab-ua.github.io/courses/resources/Qing_TKDE_2020.pdf), TKDE 2020

[Multimodal Deep Learning for Finance: Integrating and Forecasting International Stock Markets](https://arxiv.org/abs/1903.06478), 2019

[Multimodal deep learning for short-term stock volatility prediction](https://arxiv.org/abs/1812.10479), 2018

### Human AI Interaction

[Multimodal Human Computer Interaction: A Survey](https://link.springer.com/chapter/10.1007/11573425_1), HCI 2005

[Affective multimodal human-computer interaction](https://dl.acm.org/doi/10.1145/1101149.1101299), Multimedia 2005

[Building a multimodal human-robot interface](https://ieeexplore.ieee.org/abstract/document/1183338?casa_token=tdKeY0Q0e-4AAAAA:XfKwp5Di1O5bCEOnebeaS58waSbWm80lxNuY8IhWW7DqDLvRQj-8ettJW1NrFrmoR_ShudTgzw), IEEE Intelligent Systems 2001

# Workshops

[Grand Challenge and Workshop on Human Multimodal Language](http://multicomp.cs.cmu.edu/acl2020multimodalworkshop/), ACL 2020, ACL 2018

[Advances in Language and Vision Research](https://alvr-workshop.github.io/), ACL 2020

[Visually Grounded Interaction and Language](https://vigilworkshop.github.io/), NeurIPS 2019, NeurIPS 2018

[Emergent Communication: Towards Natural Language](https://sites.google.com/view/emecom2019), NeurIPS 2019

[Workshop on Multimodal Understanding and Learning for Embodied Applications](https://sites.google.com/view/mulea2019/home), ACM Multimedia 2019

[Beyond Vision and Language: Integrating Real-World Knowledge](https://www.lantern.uni-saarland.de/), EMNLP 2019

[The How2 Challenge: New Tasks for Vision & Language](https://srvk.github.io/how2-challenge/), ICML 2019

[Visual Question Answering and Dialog](https://visualqa.org/workshop.html), CVPR 2019, CVPR 2017

[Multi-modal Learning from Videos](https://sites.google.com/view/mmlv/home), CVPR 2019

[Multimodal Learning and Applications Workshop](https://mula-workshop.github.io/), CVPR 2019, ECCV 2018

[Habitat: Embodied Agents Challenge and Workshop](https://aihabitat.org/workshop/), CVPR 2019

[Closing the Loop Between Vision and Language & LSMD Challenge](https://sites.google.com/site/iccv19clvllsmdc/), ICCV 2019

[Multi-modal Video Analysis and Moments in Time Challenge](https://sites.google.com/view/multimodalvideo/), ICCV 2019

[Cross-Modal Learning in Real World](https://cromol.github.io/), ICCV 2019

[Spatial Language Understanding and Grounded Communication for Robotics](https://splu-robonlp.github.io/), NAACL 2019

[YouTube-8M Large-Scale Video Understanding](https://research.google.com/youtube8m/workshop2018/), ICCV 2019, ECCV 2018, CVPR 2017

[Language and Vision Workshop](http://languageandvision.com/), CVPR 2019, CVPR 2018, CVPR 2017, CVPR 2015

[Sight and Sound](http://sightsound.org/), CVPR 2019, CVPR 2018

[The Large Scale Movie Description Challenge (LSMDC)](https://sites.google.com/site/describingmovies/), ICCV 2019, ICCV 2017

[Wordplay: Reinforcement and Language Learning in Text-based Games](https://www.wordplay2018.com/), NeurIPS 2018

[Interpretability and Robustness in Audio, Speech, and Language](https://irasl.gitlab.io/), NeurIPS 2018

[Multimodal Robot Perception](https://natanaso.github.io/rcw-icra18/), ICRA 2018

[WMT18: Shared Task on Multimodal Machine Translation](http://www.statmt.org/wmt18/multimodal-task.html), EMNLP 2018

[Shortcomings in Vision and Language](https://sites.google.com/view/sivl/), ECCV 2018

[Computational Approaches to Subjectivity, Sentiment and Social Media Analysis](https://wt-public.emm4u.eu/wassa2018/), EMNLP 2018, EMNLP 2017, NAACL-HLT 2016, EMNLP 2015, ACL 2014, NAACL-HLT 2013

[Visual Understanding Across Modalities](http://vuchallenge.org/), CVPR 2017

[International Workshop on Computer Vision for Audio-Visual Media](https://cvavm2017.wordpress.com/), ICCV 2017

[Language Grounding for Robotics](https://robo-nlp.github.io/2017_index.html), ACL 2017

[Computer Vision for Audio-visual Media](https://cvavm2016.wordpress.com/), ECCV 2016

[Language and Vision](https://vision.cs.hacettepe.edu.tr/vl2016/), ACL 2016, EMNLP 2015

# Tutorials
[Recent Advances in Vision-and-Language Research](https://rohit497.github.io/Recent-Advances-in-Vision-and-Language-Research/), CVPR 2020

[Connecting Language and Vision to Actions](https://lvatutorial.github.io/), ACL 2018

[Machine Learning for Clinicians: Advances for Multi-Modal Health Data](https://www.michaelchughes.com/mlhc2018_tutorial.html), MLHC 2018

[Multimodal Machine Learning](https://sites.google.com/site/multiml2016cvpr/), ACL 2017, CVPR 2016, ICMI 2016

[Vision and Language: Bridging Vision and Language with Deep Learning](https://www.microsoft.com/en-us/research/publication/vision-language-bridging-vision-language-deep-learning/), ICIP 2017

# Courses

[CMU 05-618, Human-AI Interaction](https://haiicmu.github.io/)

[CMU 11-777, Advanced Multimodal Machine Learning](https://piazza.com/cmu/fall2018/11777/resources)

[Stanford CS422: Interactive and Embodied Learning](http://cs422interactive.stanford.edu/)

[CMU 16-785, Integrated Intelligence in Robotics: Vision, Language, and Planning](http://www.cs.cmu.edu/~jeanoh/16-785/)

[CMU 10-808, Language Grounding to Vision and Control](https://katefvision.github.io/LanguageGrounding/)

[CMU 11-775, Large-Scale Multimedia Analysis](https://sites.google.com/a/is.cs.cmu.edu/lti-speech-classes/11-775-large-scale-multimedia-analysis)

[MIT 6.882, Embodied Intelligence](https://phillipi.github.io/6.882/)

[Georgia Tech CS 8803, Vision and Language](http://www.prism.gatech.edu/~arjun9/CS8803_CVL_Fall17/)

[Virginia Tech CS 6501-004, Vision & Language](http://www.cs.virginia.edu/~vicente/vislang/)

