# PyTorch Variational Diffusion Model
PyTorch implementation of a variational diffusion model inspired by the paper [Kingma et al., “Variational Diffusion Models,” 2021](https://arxiv.org/pdf/2107.00630).

## Table of Contents
1. [Introduction](#introduction)
2. [File Structure](#file-structure)
3. [Setup](#setup)
4. [Results](#results)
5. [Authors](#authors)

## Introduction
Variational diffusion models provide new insight into likelihood-based generative modeling with diffusion models. They are considered as a way of connecting diffusion models and VAEs wherein the forward diffusion process is interpreted as part of the variational inference procedure.The model learns to approximate the true data distribution using a variational lower bound (VLB) and each diffusion step compresses data while the denoising network parameterises the posterior distributions.   

This repository compiles research and code analysis of existing VDM implementations. It implements a VDM that closely follows the implementation listed in the following paper [Kingma et al., “Variational Diffusion Models,” 2021](https://arxiv.org/pdf/2107.00630) but does not use variance minimisation.  

A general visualisation of the main function blocks of the code can be seen below.
<p align="center">
  <img src="/research/VDMvis.jpg" alt="VDM vis" width="50%">
</p>

## File Structure
vdm-02456/  
├── train_CIFAR.py # Initialisation, evaluation and logging for CIFAR-10  
├── VDM_our_implementation.py # Data transformations, forward process and loss training  
├── UNetModel.py # Time embedding and denoising model - passed into VDM_our_implementation.py 
├── CIFAR_Implementation/ # 
├── research/ # Existing code repository documentation and analysis  
├── testing/ # 2d VDM implementation and test inputs for CIFAR-10 forward pass and training  
├── requirements.txt  
└── README.md # Project documentation  


## Setup

### Prerequisites
- Python Version 3.9.21

### Installation
```bash
# Clone the repository
git clone https://github.com/MilaGolomozin/vdm-02456
cd vdm-02456

# Install dependencies
pip install -r requirements.txt

# Create an account on WandB first. Follow: https://docs.wandb.ai/models/quickstart
# WandB setup
wandb login

```
## Results
![2e186552-6118-443c-a93b-cd2cc66bf606](https://github.com/user-attachments/assets/8c71ba4a-0698-4b55-a545-b20a18d9d996)


## Authors
Ludmila Golomozin (s215114)  
Maja Klerk (S184488)   
Zoe Tonkin (s252284)  
