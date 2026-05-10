# In The Beginning Was Noise

> **A hands-on implementation journey through diffusion models, from DDPM to modern architectures.**

<p align = "center">
  <img src = "https://img.shields.io/badge/Status-In_Progress-yellow" alt = "Status">
  <img src = "https://img.shields.io/badge/Type-Individual_Project-blue" alt = "Type">
  <img src = "https://img.shields.io/badge/Focus-Diffusion_Models-orange" alt = "Focus">
  <img src = "https://img.shields.io/badge/Tech-PyTorch-informational" alt = "Tech">
</p>

<p align = "center">
  <a href = "https://fabioschiliro.it/In_The_Beginning_Was_Noise/">
    <img src = "assets/badges/docs.svg" alt = "Read the Documentation" height = "36">
  </a>
</p>

<p align = "center">
  <img src = "assets/images/Noise Image.png" width = "700">
</p>

## ⛓️ Table of Contents

1. [About The Project](#-about-the-project)
2. [Repository Structure](#-repository-structure)
3. [Getting Started](#-getting-starting)
10. [References and Citation](#-references-and-citation)


## 📖 About The Project
Diffusion models have rapidly evolved from the original Denoising Diffusion Probabilistic Models (DDPM) to highly optimized latent and transformer-based generative systems.  
This repository aims to reconstruct that evolution through implementation: instead of treating diffusion models as black boxes, each method is implemented incrementally, following the original papers as closely as possible while emphasizing the conceptual clarity, the mathematical understanding and the reproducibility trhough the PyTorch Implementation.  
At the moment, the implemented models are the followings:

| Year     | Paper                             | Concept                       | Status    | 
| :---:    | :---:                             | :---:                         | :---:     |
| 2020     | [DDPM](./assets/papers/DDPM.pdf)  | Probabilistic diffusion       |   🚧      |


## 🗼 Repository Structure
This repository is organized as follows:

<details>
<summary><b>Visualize Repository Structure</b></summary> 
<br>

TODO

</details>


## 📌 Roadmap
The main goals followed in this repo are listed below:

<details>
<summary><b>Visualize the list</b></summary> 

### Foundations
- [x] Repository setup
- [x] Training pipeline skeleton
- [x] Inference pipeline skeleton

- [x] Noise scheduler implementation
- [x] Sampling loop

### Diffusion Models
- [x] DDPM implementation

### Documentation
- [ ] DDPM Explaination and Documentation

### Fixes
- [ ] Logger problem in Colab


</details>



## 🚀 Getting Started
In order to start experimenting with this repo, you can choose to use your compuer or execute the models on Colab; however, it is importart to follow the specific section to correctly run the code.


<details>
<summary><b>Execute on your computer 💻</b></summary> 
<br>
If you want to execute the models on your computer, just follow the commands below:
<br></br>

0. **Clone the repository**

    ```bash
    git clone https://github.com/your-username/in_the_beginning_was_noise.git
    cd in_the_beginning_was_noise
    ```

1. **Create a Conda Environment**

    ```bash
    conda env create -f environment.yml
    ```

2. **Active the Conda Environment**

    ```bash
    conda activate ITBWN
    ```

<br>
In addition, if you want to remove the created environment, you can use the following commands:
<br></br>

1. **Deactivate the Conda Environment**

    ```bash
    conda deactivate
    ```

2. **Delete all the packages installed in the environment and remove it**

    ```bash
    conda remove -n ITBWN --all
    ```

</details>

<br>

<details>
<summary><b>Execute on Colab 📡</b></summary> 
<br>
If you want to execute the models on Google Colab, just run the commands below:
<br></br>

0. **Clone the repository**

    ```bash
    git clone https://github.com/your-username/in_the_beginning_was_noise.git
    cd in_the_beginning_was_noise
    ```

1. **Install the package to use Conda on Colab**

    ```bash
    !pip install -q condacolab
    ```
  
2. **Import and initialize the package**

    ```bash
    import condacolab
    condacolab.install()
    ```

   *⚠️ Note: This step usually restarts the runtime automatically after installation, just run the commands at step 3 after this event.*

3. **Install the dependencies listed in the environment file**

    ```bash
    !mamba env update -n base -f environment.yml
    ```

*⚠️ Note: Up to now, the rich logger seems to have problems when executed on Colab, not showing any output in the running cell (e.g. during training). For the training process you can consult the generated log file to monitor the process (i.e. close and open the file occasionally to see the update on the file itself). The problem will be fixed as soon as possible.*

</details>


## 💡 Models Usage
In this section you can understand how to use the implemented models both for training and inference purposes.
Be aware to look at the specific model subsection for every possible difference with respect to a traditional execution. 

For both the scenarios below you can use the `-h / --help` flag to understand the behaviour. 

### Training
The training can be performed using the following command:

  ```python
  python train.py --config configs.modelConfiguration
  ```

### Inference
The inference can be performed using the following command:

  ```python
  python inference.py --config configs.ddpm_celebahq \
    --checkpoint /path/to/Epoch_100.pth \
    --output-dir ./generated \
    --num-images 4 \
    --image-size 256
  ```


## 📖 References and Citation
Throughout this README, I have added references as links the first time a paper, framework, dataset or tool is mentioned. If you notice anything missing or think an additional citation would improve clarity or credit, feel free to open an issue or submit a PR.

If you use this repository (code, configs, datasets setup, pipeline) in your research or project, please cite it using the BibTeX entry below:

```bibtex
@misc{in_the_beginning_was_noise,
author  = {Schilirò, Biagio Fabio},
title   = {In The Beginning Was Noise},
url     = {https://github.com/FabioS08/In_The_Beginning_Was_Noise},
year    = {2026}
}
```