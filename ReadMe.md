# DPPA: Pruning Method for Large Language Model to Model Merging

The code for our paper ([paper](https://arxiv.org/abs/2403.02799))
 
## Directory

- [Usage](#Usage)
  - [Requirements](#Requirements)
  - [Installation](#Installation)
- [Document](#Document)
- [Acknowledgement](#Acknowledgement)

### Usage

Please change cache in "run.sh" to the path of the project.



###### Requirements

1. python 3.10
2. cuda 12.1
3. torch 2.1.0

###### Installation

```sh
conda create -n DPPA python=3.10
conda activate DPPA
conda install nvidia/label/cuda-12.1.0::cuda
pip install -r requirements.txt
pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### Document

```
DPPA 
├── README.md
├── /model/
│  ├── /llama2-7B/
│  ├── /Abel-7B-001/
│  ├── /other SFT models/
├── /gair_abel/
├── /scrips/
│  ├── /pruning/
├── run.sh
├── requirements.txt

```

### Acknowledgement
This repository is build upon the [OWL](https://github.com/luuyin/OWL) repositories.
