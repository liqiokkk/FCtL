# From Contexts to Locality: Ultra-high Resolution Image Segmentation via Locality-aware Contextual Correlation
From Contexts to Locality: Ultra-high Resolution Image Segmentation via Locality-aware Contextual Correlation  
Qi Li, Weixiang Yang, Wenxi Liu, Yuanlong Yu, Shengfeng He  
Accepted to ICCV 2021
## Abstract
Ultra-high resolution image segmentation has raised increasing interests in recent years due to its realistic applications. In this paper, we innovate the widely used high-resolution image segmentation pipeline, in which an ultra-high resolution image is partitioned into regular patches for local segmentation and then the local results are merged into a high-resolution semantic mask. In particular, we introduce a novel locality-aware contextual correlation based segmentation model to process local patches, where the relevance between local patch and its various contexts are jointly and complementarily utilized to handle the semantic regions with large variations. Additionally, we present a contextual semantics refinement network that associates the local segmentation result with its contextual semantics, and thus is endowed with the ability of reducing boundary artifacts and refining mask contours during the generation of final high-resolution mask. Furthermore, in comprehensive experiments, we demonstrate that our model outperforms other state-of-the-art methods in public benchmarks.   
![tease](https://github.com/liqiokkk/FCtL/blob/main/img/tease.png)  
## Method
![framework](https://github.com/liqiokkk/FCtL/blob/main/img/framework.png)

## Test and train
Our codes are base on [GLNet](https://github.com/VITA-Group/GLNet)  
python>=3.6 and pytorch>=1.2.0  
Please install the dependencies: `pip install -r requirements.txt`
### Dataset
Please register and download the Deep Globe "Land Cover Classification" dataset [here](https://competitions.codalab.org/competitions/18468):
Create folder named 'data', its structure is  

### test
download following pretrianed-model here  
1.all.epoch.pth  
2.medium.epoch.pth  
3.global.epoch.pth  
`bash test_all.sh`  
### train
please sequentially finish the following steps:
1.`bash train_global.sh`
2.`bash train_pre.sh`(not necessary)  
3.`bash train_medium.sh`  
4.`bash train_all.sh`  
## Results
![result](https://github.com/liqiokkk/FCtL/blob/main/img/result.png)
