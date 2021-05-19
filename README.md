# ELECTRA-for-Kinase-specific-Phosphorylation-Site-Prediction

##  Introduction

**ELECTRA for Kinase-specific Phosphorylation Site Prediction** provides a pre-trained protein fragments representation model for general and kinase-specific phosphorylation site prediction.

Existing methods only learn representations of a protein sequence segment from a small labeled dataset itself, which could result in biased or incomplete features, especially for kinase-specific phosphorylation site prediction in which training data are typically sparse. To learn a comprehensive contextual representation of a protein sequence segment for kinase-specific phosphorylation site prediction, we pre-trained our model from over 24 million unlabeled sequence fragments using ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately). The pre-trained model was applied to kinase-specific site prediction of kinases **CDK, PKA, CK2, MAPK, and PKC**. 

## Requirements
* Python 3
* [TensorFlow](https://www.tensorflow.org/) 1.15
* [NumPy](https://numpy.org/)
* [scikit-learn](https://scikit-learn.org/stable/) and [SciPy](https://www.scipy.org/) (for computing some evaluation metrics).
* [Pandas](https://pandas.pydata.org/)
* [Git LFS](https://git-lfs.github.com/) (for large model files download).

Because the model files are over 100M, please use **[Git LFS](https://git-lfs.github.com/)** to clone the repository. Please install **[Git LFS](https://git-lfs.github.com/)** before using git clone.

### Windows / Cygwin
Please go to the website [https://git-lfs.github.com/](https://git-lfs.github.com/) to download the package and install.

### macOS
```sh
brew install git-lfs
```

### Arch Linux
```sh
sudo pacman -S --noconfirm git-lfs
```

### Debian >= 9 / Ubuntu >= 18.04
```sh
sudo apt-get update
sudo apt-get install git-lfs
```

### CentOS
```sh
sudo yum install git-lfs
```
After installing [Git LFS](https://git-lfs.github.com/), please use the following command to clone the repository.

```sh
git lfs clone https://github.com/cirisjl/ELECTRA-for-Kinase-specific-Phosphorylation-Site-Prediction.git
```

The ELECTRA model was trained on one GPU and hence it needs one GPU to run the prediction.

## Running on GPU

The input of the tool is a file containing protein sequences in the **FASTA** format, and the output of the tool is a file that includes the probability scores for each candidate site for each kinase. 0.5 was used as a default cutoff and residues with probability scores higher than **0.5** are predicted as the substrates for the kinase under consideration. You may use your own data set to do prediction. We also provided a test dataset for predicting, which is located at:
>testdata

The default folder for output is:
>results

The default batch size is 128 for prediction, which can be adjusted according to the VRAM of the user’s local machine.

#### For general phosphorylation site prediction using our pre-trained model, run:
```sh
python3 predict.py -input [custom predicting data in fasta format] -predict-type general
```
##### Example:
```sh
python3 predict.py -input testdata/testing_proteins_ST_withannotation.fasta -predict-type general
```
For our general phosphorylation site prediction, only **S and T** are acceptable.

For details of other parameters, run:
```sh
python predict.py --help
```
or
```sh
python predict.py -h
```
#### For kinase-specific phosphorylation site prediction using our pre-trained model, run:
```sh
python3 predict.py -input [custom predicting data in fasta format] -predict-type kinase -kinase [custom specified kinase to predict]
```
##### Example:
Prediction for CDK:
```sh
python3 predict.py -input testdata/testing_proteins_CDK_withannotation.fasta -predict-type kinase -kinase CDK
```
For kinase-specific phosphorylation site prediction, we only consider kinase families **CDK, CK2, MAPK, PKA, and PKC**, each of which has more than 100 known substrate phosphorylation sites. 
