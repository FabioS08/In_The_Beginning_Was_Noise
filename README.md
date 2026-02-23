# In The Beginning Was Noise

> **A hands-on implementation journey through diffusion models, from DDPM to modern architectures.**

![Status](https://img.shields.io/badge/Status-In_Progress-yellow)
![Type](https://img.shields.io/badge/Type-Individual_Project-blue)
![Focus](https://img.shields.io/badge/Focus-Diffusion_Models-orange)
![Tech](https://img.shields.io/badge/Tech-PyTorch-informational)


<p align="center">
  <img src = "assets/Noise Image.png" width = "700">
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
- [ ] Training pipeline skeleton
- [ ] DDPM implementation
- [ ] Noise scheduler implementation
- [ ] Sampling loop

### Diffusion Improvements
- [ ] Improved DDPM
- [ ] DDIM sampling
- [ ] Classifier guidance
- [ ] Classifier-free guidance

### Modern Diffusion Models
- [ ] Latent Diffusion Models
- [ ] Stable Diffusion components
- [ ] Conditional generation
- [ ] Transformer-based diffusion

### Documentation
- [ ] Mathematical derivations
- [ ] Paper summaries
- [ ] Training tutorials

</details>



## 🚀 Getting Started
In order to start experimenting with this repo, run the following commands:

(LOOK FOR environment.yml ->> conda env create -f environment.yml)

1. **Clone the repository**

    ```bash
    git clone https://github.com/your-username/in_the_beginning_was_noise.git
    cd in_the_beginning_was_noise
    ```

1. **Create a Conda Environment**

    ```bash
    conda create -n $(basename $PWD) python=3.10
    ```

2. **Setup the Conda Environment**

    ```bash
    conda activate $(basename $PWD)
    pip install -r requirements.txt
    ```


Decide if to use it or not:
when finished:

- conda deactivate
- conda remove -n myproject --all


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