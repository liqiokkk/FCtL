# From Contexts to Locality: Ultra-high Resolution Image Segmentation via Locality-aware Contextual Correlation
[From Contexts to Locality: Ultra-high Resolution Image Segmentation via Locality-aware Contextual Correlation](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_From_Contexts_to_Locality_Ultra-High_Resolution_Image_Segmentation_via_Locality-Aware_ICCV_2021_paper.pdf)  
Qi Li, Weixiang Yang, Wenxi Liu, Yuanlong Yu, Shengfeng He  
Accepted to ICCV 2021
## Abstract
Ultra-high resolution image segmentation has raised increasing interests in recent years due to its realistic applications. In this paper, we innovate the widely used high-resolution image segmentation pipeline, in which an ultra-high resolution image is partitioned into regular patches for local segmentation and then the local results are merged into a high-resolution semantic mask. In particular, we introduce a novel locality-aware contextual correlation based segmentation model to process local patches, where the relevance between local patch and its various contexts are jointly and complementarily utilized to handle the semantic regions with large variations. Additionally, we present a contextual semantics refinement network that associates the local segmentation result with its contextual semantics, and thus is endowed with the ability of reducing boundary artifacts and refining mask contours during the generation of final high-resolution mask. Furthermore, in comprehensive experiments, we demonstrate that our model outperforms other state-of-the-art methods in public benchmarks.   
![tease](https://github.com/liqiokkk/FCtL/blob/main/img/tease.png)  
## Method
![framework](https://github.com/liqiokkk/FCtL/blob/main/img/framework.png)

## Test and train
Our code is based on [GLNet](https://github.com/VITA-Group/GLNet)  
python>=3.6 and pytorch>=1.2.0  
Please install the dependencies: `pip install -r requirements.txt`
### dataset
Please register and download [DeepGlobe](https://competitions.codalab.org/competitions/18468) dataset  
Create folder named 'data', its structure is  
```
data/
├── train
   ├── Sat
      ├── xxx_sat.jpg
      ├── ...
   ├── Label
      ├── xxx_mask.png(channel:0-6)
      ├── ...
├── crossvali
├── offical_crossvali
```
### test
Please download following pretrianed-model [here](https://drive.google.com/drive/folders/1OPsNQ-33LMBaiBytr_bX9sDFXjhZIlsm?usp=sharing)  
1.all.epoch.pth  2.medium.epoch.pth  3.global.epoch.pth  
`bash test_all.sh`  
### train
Please sequentially finish the following steps:   
1.`bash train_pre.sh`(not necessary)  
2.`bash train_medium.sh`(get medium context)  
3.`bash train_global.sh`(get large context)  
4.`bash train_all.sh`  
## Results
### DeepGlobe
![result](https://github.com/liqiokkk/FCtL/blob/main/img/result.png)
### Inria Aerial  
![result1](https://github.com/liqiokkk/FCtL/blob/main/img/result1.png)
## Citation
If you use this code and our results for your research, please cite our paper.
```
@inproceedings{li2021contexts,
  title={From Contexts to Locality: Ultra-high Resolution Image Segmentation via Locality-aware Contextual Correlation},
  author={Li, Qi and Yang, Weixiang and Liu, Wenxi and Yu, Yuanlong and He, Shengfeng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={7252--7261},
  year={2021}
}
```
