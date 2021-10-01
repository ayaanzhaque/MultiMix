# [MultiMix](https://ayaanzhaque.github.io/MultiMix/)
This repository contains the implementation of MultiMix. Our publications for this project are listed below:

["MultiMix: Sparingly Supervised, Extreme Multitask Learning From Medical Images,"](https://arxiv.org/abs/2010.14731) by Ayaan Haque, Abdullah-Al-Zubaer Imran, Adam Wang, and Demetri Terzopoulos. In [ISBI](https://biomedicalimaging.org/2021/), 2021.

["Generalized Multi-Task Learning from Substantially Unlabeled Multi-Source Medical Image Data,"]() by Ayaan Haque, Abdullah-Al-Zubaer Imran, Adam Wang, and Demetri Terzopoulos. In [MELBA](https://www.melba-journal.org/), 2021.

Our proposed model performs joint semi-supervised classification and segmentation by employing a confidence-based augmentation strategy for semi-supervised classification along with a novel saliency bridge module that guides segmentation and provides explainability for the joint tasks. 

## Abstract
Semi-supervised learning via learning from limited quantities of labeled data has been investigated as an alternative to supervised counterparts. Maximizing knowledge gains from copious unlabeled data benefit semi-supervised learning settings. Moreover, learning multiple tasks within the same model further improves model generalizability.  We propose a novel multitask learning model, namely MultiMix, which jointly learns disease classification and anatomical segmentation in a sparingly supervised manner, while preserving explainability through bridge saliency between the two tasks. Our extensive experimentation with varied quantities of labeled data in the training sets justify the effectiveness of our multitasking model for the classification of pneumonia and segmentation of lungs from chest X-ray images. Moreover, both in-domain and cross-domain evaluations across the tasks further showcase the potential of our model to adapt to challenging generalization scenarios.

## Model

![Figure](https://github.com/ayaanzhaque/MultiMix/blob/main/images/multimix_journal_diagram.jpg?raw=true)

For sparingly-supervised classification, we leverage data augmentation and pseudo-labeling. We take an unlabeled image and perform two separate augmentations. A single unlabeled image is first weakly augmented, and from that weakly augmented version of the image, a pseudo-label is assumed based on the prediction from the current state of the model. Secondly, the same unlabeled image is then augmented strongly, and a loss is calculated with the pseudo-label from the weakly augmented image and the strongly augmented image itself. Note that this image-label pair is retained only if the confidence with which the model generates the pseudo-label is above a tuned threshold, which prevents the model from learning from incorrect and poor labels.

For sparingly-supervised segmentation, we generate saliency maps based on the predicted classes using the gradients of the encoder. While the segmentation images do not necessarily represent pneumonia, the classification task, the generated maps highlight the lungs, creating images at the final segmentation resolution. These saliency maps can be used to guide the segmentation during the decoder phase, yielding improved segmentation while learning from limited labeled data. In our algorithm, the generated saliency maps are concatenated with the input images, downsampled, and added to the feature maps input to the first decoder stage. Moreover, to ensure consistency, we compute the KL divergence between segmentation predictions for labeled and unlabeled examples. This penalizes the model from making predictions that are increasingly different than those of the labeled data, which helps the model fit more appropriately for the unlabeled data.

## Results
A brief summary of our results are shown below. Our algorithm MultiMix is compared to various baselines. In the table, the best fully-supervised scores are underlined and the best semi-supervised scores are bolded.

![Results](https://github.com/ayaanzhaque/MultiMix/blob/main/images/results_table.png?raw=true)

![Boundaries](https://github.com/ayaanzhaque/MultiMix/blob/main/images/boundary_preds.png?raw=true)

## Code
The code has been written in Python using the Pytorch framework. Training requries a GPU. We provide a Jupyter Notebook, which can be run in Google Colab, containing the algorithm in a usable version. Open [`MultiMix.ipynb`](https://github.com/ayaanzhaque/MultiMix/blob/main/MultiMix.ipynb) and run it through. The notebook includes annotations to follow along. Open the sample_data folder and use the classification and segmentation sample images for making predictions. Load [`multimix_trained_model.pth`](https://github.com/ayaanzhaque/MultiMix/blob/main/sample_data/multimix_trained_model.pth) and make predictions on the provided images. Uncomment the training cell to train the model.

## Citation
If you find this repo or the paper useful, please cite: 

**ISBI Paper**
```
@inproceedings{haque2020multimix,
      author={Haque, Ayaan and Imran, Abdullah-Al-Zubaer and Wang, Adam and Terzopoulos, Demetri},
      booktitle={2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI)}, 
      title={Multimix: Sparingly-Supervised, Extreme Multitask Learning from Medical Images}, 
      year={2021},
      volume={},
      number={},
      pages={693-696},
      doi={10.1109/ISBI48211.2021.9434167}
}
```

**MELBA Paper**
```
To be released
```
