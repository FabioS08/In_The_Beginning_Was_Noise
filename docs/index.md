---
hide:
  - navigation
  - toc
---

# In The Beginning Was Noise
Diffusion models have rapidly evolved from the original Denoising Diffusion Probabilistic Models (DDPM) to highly optimized latent and transformer-based generative systems.  
This project aims to reconstruct that evolution through implementation: instead of treating diffusion models as black boxes, each method is implemented incrementally, following the original papers as closely as possible while emphasizing the conceptual clarity, the mathematical understanding and the reproducibility trhough the PyTorch Implementation.  

<figure style = "width: 1200px; margin: 2em auto; text-align: center;">

  <img src = "assets/images/Noise Image.png" style = "width: 100%" alt = "Diffusion Process">
  <figcaption style = "width: 100%; margin-top: 0.5em; font-size: 0.9em; color: gray;">
    Figure 1: Illustration of the Diffusion Process for generation, with noise gradually removed to create a final image.
  </figcaption>

</figure>


## 🚀 Getting Started
In order to start experimenting with this project, you can choose to use your compuer or execute the models on Colab; however, it is importart to follow the specific section to correctly run the code.

<details>

  <summary><b>Execute on your computer 💻</b></summary>

  <p>
    If you want to execute the models on your computer, just follow the commands below:
  </p>

  <ol start = "0">
    <li>
      <b>Clone the repository</b>
      <pre><code class = "language-bash">git clone https://github.com/your-username/in_the_beginning_was_noise.git
cd in_the_beginning_was_noise</code></pre>
    </li>

    <li>
      <b>Create a Conda Environment</b>
      <pre><code class="language-bash">conda env create -f environment.yml</code></pre>
    </li>

    <li>
      <b>Activate the Conda Environment</b>
      <pre><code class="language-bash">conda activate ITBWN</code></pre>
    </li>
  </ol>

  <p>
    In addition, if you want to remove the created environment, you can use the following commands:
  </p>

  <ol>
    <li>
      <b>Deactivate the Conda Environment</b>
      <pre><code class = "language-bash">conda deactivate</code></pre>
    </li>

    <li>
      <b>Delete all the packages installed in the environment and remove it</b>
      <pre><code class="language-bash">conda remove -n ITBWN --all</code></pre>
    </li>
  </ol>
</details>


<details>
  <summary><b>Execute on Colab 📡</b></summary>

  <p>
    If you want to execute the models on Google Colab, just run the commands below:
  </p>

  <ol start = "0">
    <li>
      <b>Clone the repository</b>
      <pre><code class = "language-bash">!git clone https://github.com/your-username/in_the_beginning_was_noise.git
cd in_the_beginning_was_noise</code></pre>
    </li>

    <li>
      <b>Install the package to use Conda on Colab</b>
      <pre><code class = "language-bash">!pip install -q condacolab</code></pre>
    </li>

    <li>
      <b>Import and initialize the package</b>
      <pre><code class = "language-python">import condacolab
condacolab.install()</code></pre>
      <p>
        <b>⚠️ Note:</b> This step usually restarts the runtime automatically after installation, just run the commands at step 3 after this event.
      </p>
    </li>

    <li>
      <b>Install the dependencies listed in the environment file</b>
      <pre><code class = "language-bash">!mamba env update -n base -f environment.yml</code></pre>
    </li>
  </ol>

  <p>
    <b>⚠️ Note:</b> Up to now, the rich logger seems to have problems when executed on Colab, not showing any output in the running cell (for example during training). For the training process you can consult the generated log file to monitor the process (that is, close and open the file occasionally to see updates in the file itself). The problem will be fixed as soon as possible.
  </p>
</details>

The training and inference processes are handled by the `train.py` and `inference.py` scripts, respectively.<br> 
For more details, refer to the [Architecture - Training Pipeline:octicons-arrow-up-right-24:](architecture/training_pipeline.md) and [Architecture - Inference Pipeline:octicons-arrow-up-right-24:](architecture/training_pipeline.md) sections.


## 📟 Supported Models
The models currently implemented or under development are listed below. For each one, you can access the original paper, explore a detailed explanation of how it works, gain a quick overview of its key innovations and check its implementation status.

<div align="center" markdown>

| Year | Paper + Explaination| Core Concept | Status |
|:----:|:-------------------:|:------------:|:------:|
| 2020 | [DDPM](papers/ddpm.md) | Probabilistic Diffusion | :material-check-circle: Fully Implemented |
| 202x | [???](papers/ddpm.md) | ??? | :material-progress-wrench: In Progress |

</div>

!!! tip "New here?"
    Start with the Paper Explaination related to the specific model to understand the theory and, then, check the Architecture Overview to see how it is mapped to code.


## 🧭 Explore the project
Dive into the sections below to gain a clear understanding of how the models operate during both training and inference. You will also find everything you need to customize, modify and create your own model. Let's start!

<div class = "grid cards" markdown>

-   :material-file-document:{ .lg .middle } __Paper Explanations__

    ---

    Gain deep insights into the theory behind each implemented model. These sections provide step-by-step mathematical breakdowns, intuitive explanations and direct connections to the code.

    [:octicons-arrow-right-24: Read the papers](papers/index.md)

-   :material-code-braces:{ .lg .middle } __Architecture__

    ---

    Explore the design principles of the framework, including how modules are structured and how different components interact within the system.

    [:octicons-arrow-right-24: Explore the architecture](architecture/index.md)

</div>

<div class = "grid cards" markdown>

-   :material-cog:{ .lg .middle } __Guides__

    ---

    Follow practical, hands-on tutorials for configuring models, training, running inference and extending the framework with your own ideas.

    [:octicons-arrow-right-24: Read the guides](guides/index.md)

-   :material-api:{ .lg .middle } __API Reference__

    ---

    Browse detailed documentation for all classes, functions and configuration options available in the codebase.

    [:octicons-arrow-right-24: Browse the API](api/index.md)

</div>

## ⚠ Disclaimer
This project started as a way for me to better understand a topic that felt pretty confusing at first. Because of that, the code isn’t meant to be super optimized or follow all the best coding practices: the goal was just to experiment and try building things from scratch to really get how they work.<br>
No AI tools have been used for writing the code itself (though I definitely loved them for the creation of this documentation 🙃); so if you spot any silly mistakes and wonder why they are there…it’s just me being human, working with limited time and trying to learn something new along the way.


## 📖 References and Citation
Throughout this documentation, I have added references as links the first time a paper, framework, dataset or tool is mentioned. If you notice anything missing or think an additional citation would improve clarity or credit, feel free to open an issue or submit a PR.

If you use this project (code, configs, datasets setup, pipeline) in your research or project, please cite it using the BibTeX entry below:

```bibtex
@misc{in_the_beginning_was_noise,
author  = {Schilirò, Biagio Fabio},
title   = {In The Beginning Was Noise},
url     = {https://github.com/FabioS08/In_The_Beginning_Was_Noise},
year    = {2026}
}
```