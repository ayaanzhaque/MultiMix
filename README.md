# MultiMix
This repository contains the implementation of the paper MultiMix: Sparingly Supervised, Extreme Multitask Learning From Medical Images, by Ayaan Haque, Abdullah-Al-Zubaer Imran, Adam Wang, and Demetri Terzopoulos. 

**The supplementary material can be found here:** [supplemental_document.pdf](https://github.com/ayaanzhaque/MultiMix/blob/main/supplemental_document.pdf)

Our proposed models performs joint semi-supervised classification and segmentation by employing a confidence-based augmentation strategy for semi-supervised classification along with a novel saliency bridge module that guides segmentation and provides explainability for the joint tasks. 

## Abstract
Semi-supervised learning via learning from limited quantities of labeled data has been investigated as an alternative to supervised counterparts. Maximizing knowledge gains from copious unlabeled data benefit semi-supervised learning settings. Moreover, learning multiple tasks within the same model further improves model generalizability.  We propose a novel multitask learning model, namely MultiMix, which jointly learns disease classification and anatomical segmentation in a sparingly supervised manner, while preserving explainability through bridge saliency between the two tasks. Our extensive experimentation with varied quantities of labeled data in the training sets justify the effectiveness of our multitasking model for the classification of pneumonia and segmentation of lungs from chest X-ray images. Moreover, both in-domain and cross-domain evaluations across the tasks further showcase the potential of our model to adapt to challenging generalization scenarios.

## Model

![Figure](https://github.com/ayaanzhaque/MultiMix/blob/main/images/multimix_diagram.png?raw=true)

For sparingly-supervised classification, we leverage data augmentation and pseudo-labeling. We take an unlabeled image and perform two separate augmentations. A single unlabeled image is first weakly augmented, and from that weakly augmented version of the image, a pseudo-label is assumed based on the prediction from the current state of the model. Secondly, the same unlabeled image is then augmented strongly, and a loss is calculated with the pseudo-label from the weakly augmented image and the strongly augmented image itself. Note that this image-label pair is retained only if the confidence with which the model generates the pseudo-label is above a tuned threshold, which prevents the model from learning from incorrect and poor labels.

For sparingly-supervised segmentation, we generate saliency maps based on the predicted classes using the gradients of the encoder. While the segmentation images do not necessarily represent pneumonia, the classification task, the generated maps highlight the lungs, creating images at the final segmentation resolution. These saliency maps can be used to guide the segmentation during the decoder phase, yielding improved segmentation while learning from limited labeled data. In our algorithm, the generated saliency maps are concatenated with the input images, downsampled, and added to the feature maps input to the first decoder stage. Moreover, to ensure consistency, we compute the KL divergence between segmentation predictions for labeled and unlabeled examples. This penalizes the model from making predictions that are increasingly different than those of the labeled data, which helps the model fit more appropriately for the unlabeled data.

## Datasets
The models were trained and tested on the combined classification and segmentation tasks using data from two dif-ferent sources:  pneumonia detection (CheX) [1] and JSRT[2]. Furthermore, we used the Montgomery County chest X-rays (MCU) [3] and a subset of the NIH chest X-ray dataset(NIHX) [4] for cross-domain evaluation.

![Datasets](https://github.com/ayaanzhaque/MultiMix/blob/main/images/datasets_table.png?raw=true)

## Results
A brief summary of our results are shown below. Our algorithm MultiMix is compared to various baselines. In the table, the best fully-supervised scores are underlined and the best semi-supervised scores are bolded.

![Results](https://github.com/ayaanzhaque/MultiMix/blob/main/images/results_table.png?raw=true)

![Boundaries](https://github.com/ayaanzhaque/MultiMix/blob/main/images/boundary_preds.png?raw=true)

## Code
The code has been written in Python using the Pytorch framework. Training requries a GPU. We provide a Jupyter Notebook, which can be run in Google Colab, containing the algorithm in a usable version. Open [`MultiMix.ipynb`](https://github.com/ayaanzhaque/MultiMix/blob/main/MultiMix.ipynb) and run it through. The notebook includes annotations to follow along.

## References
[1] Daniel S Kermany, Michael Goldbaum, Wenjia Cai, Car-olina CS Valentim, Huiying Liang, Sally L Baxter, AlexMcKeown, Ge Yang, Xiaokang Wu, Fangbing Yan, et al.,“Identifying medical diagnoses and treatable diseases byimage-based deep learning,”Cell, vol. 172, no. 5, pp.1122–1131, 2018.

[2] Junji Shiraishi, Shigehiko Katsuragawa, et al.,  “Devel-opment of a digital image database for chest radiographswith and without a lung nodule,”J of Roent, 2000.

[3] Stefan  Jaeger,  Sema  Candemir,  et  al.,    “Two  publicchest X-ray datasets for computer-aided screening ofpulmonary  diseases,”Quant Imag in Med and Surg,2014.

[4] Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mo-hammadhadi Bagheri, and Ronald M Summers, “Chestx-ray8:  Hospital-scale  chest  x-ray  database  and  bench-marks on weakly-supervised classification and localiza-tion of common thorax diseases,” inProceedings of theIEEE conference on computer vision and pattern recog-nition, 2017, pp. 2097–2106.
