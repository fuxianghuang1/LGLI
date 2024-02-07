# Language Guided Local Infiltration for Interactive Image Retrieval - Accepted at CVPR Workshop 2023
- The paper can be accessed at [CVPR2023W/IMW](https://openaccess.thecvf.com/content/CVPR2023W/IMW/papers/Huang_Language_Guided_Local_Infiltration_for_Interactive_Image_Retrieval_CVPRW_2023_paper.pdf)


If you find this code useful in your research then please cite


'''
@inproceedings{LGLI2023,

    author    = {Huang, Fuxiang and Zhang, Lei},
	
    title     = {Language Guided Local Infiltration for Interactive Image Retrieval},
	
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
	
    year      = {2023},
}
'''


## Abstract

Interactive Image Retrieval (IIR) aims to retrieve imagesthat are generally similar to the reference image but under the requested text modification. The existing methodsusually concatenate or sum the features of image and textsimply and roughly, which, however, is difficult to preciselychange the local semantics of the image that the text intends to modify. To solve this problem, we propose a LanguageGuided Local Infiltration (LGLI) system, which fully utilizesthe text information and penetrates text features into imagefeatures as much as possible. Specifically, we first proposea Language Prompt Visual Localization (LPVL) module togenerate a localization mask which explicitly locates the region (semantics) intended to be modified. Then we introduce a Text Infiltration with Local Awareness (TILA) module, which is deployed in the network to precisely modifythe reference image and generate image-text infiltrated representation. Extensive experiments on various benchmarkdatabases validate that our method outperforms most stateof-the-art IIR approaches.

## Requirements and Installation
* Python 3.6
* [PyTorch](http://pytorch.org/) 1.2.0
* [NumPy](http://www.numpy.org/) (1.16.4)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)

## Description of the Code [(From TIRG)](https://github.com/google/tirg/edit/master/README.md)
The code is based on TIRG code. 


- `main.py`: driver script to run training/testing
- `datasets.py`: Dataset classes for loading images & generate training retrieval queries
- `text_model.py`: LSTM model to extract text features
- `img_text_composition_models.py`: various image text compostion models 
- `torch_function.py`: contains soft triplet loss function and feature normalization function
- `test_retrieval.py`: functions to perform retrieval test and compute recall performance

## Running the experiments 

### Download the datasets

### CSS3D dataset

Download the dataset from this [external website](https://drive.google.com/file/d/1wPqMw-HKmXUG2qTgYBiTNUnjz83hA2tY/view?usp=sharing).

Make sure the dataset include these files:
`<dataset_path>/css_toy_dataset_novel2_small.dup.npy`
`<dataset_path>/images/*.png`

#### MITStates dataset

Download the dataset via this [link](http://web.mit.edu/phillipi/Public/states_and_transformations/index.html) and save it in the ``data`` folder. Kindly take care that the dataset should have these files:

```data/mitstates/images/<adj noun>/*.jpg```


#### Fashion200k dataset

Download the dataset via this [link](https://github.com/xthan/fashion-200k) and save it in the ``data`` folder.
To ensure fair comparison, we employ the same test queries as TIRG. They can be downloaded from [here](https://storage.googleapis.com/image_retrieval_css/test_queries.txt). Kindly take care that the dataset should have these files:

```
data/fashion200k/labels/*.txt
data/fashion200k/women/<category>/<caption>/<id>/*.jpeg
data/fashion200k/test_queries.txt`
```



## Running the Code

For training and testing new models, pass the appropriate arguments. 

For instance, for training LGLI model on Fashion200k dataset run the following command:

```
python   main.py --dataset=fashion200k --dataset_path=../data/fashion200k/  --model=LGLI --loss=batch_based_classification --learning_rate_decay_frequency=50000 --num_iters=160000 --use_complete_text_query True  
```






