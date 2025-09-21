# Udacity Generative AI Nanodegree: Computer Vision and Generative AI

These are my personal notes taken while following the [Udacity Generative AI Nanodegree](https://www.udacity.com/course/generative-ai--nd608).

The Nanodegree has 4 modules:

1. Generative AI Fundamentals.
2. Large Language Models (LLMs) & Text Generation.
3. Computer Vision and Generative AI.
4. Building Generative AI Solutions.

This folder & guide refer to the **third module**: Computer Vision and Generative AI.

Mikel Sagardia, 2024.
No guarantees.

Overview of Contents:

## 1. Introduction to Image Generation

Introductory concepts:

- Discriminative vs. Generative Models
  - Discriminative models learn decision boundaries
  - Generative models learn distributions where they sample from
  - Mathematically they capture related but different models
  - Discriminative models cannot generate, but generative models can somehow discriminate
- Images are high-dimensional points
  - Realistic images are really points in a large vast of noise
  - Generative models learn *islands in an universe of noise*
- Type of Computer Vision Generetive Models
  - Unconditional models, `p(x)`: they learn to generate realistic images without any input; e.g., many GANs. See [https://thispersondoesnotexist.com/](https://thispersondoesnotexist.com/).
  - Conditional models `p(x|prompt)`: they generate according to an input prompt; e.g. :
    - [Stable Diffusion](https://huggingface.co/spaces/google/sdxl): text-to-image
    - [BLIP](https://huggingface.co/spaces/library-samples/image-captioning-with-blip): image-to-text (captioning)
    - [VideoLDM](https://research.nvidia.com/labs/toronto-ai/VideoLDM/): text-to-video
    - [DreamGaussian](https://arxiv.org/abs/2309.16653): image to 3d model
  - Multi-modal `p(x|prompt)`: they take images/texts/etc. and and work with them simultaneously, e.g., we ask it about something on an image; e.g.:
    - [GPT4-Vision](https://openai.com/research/gpt-4v-system-card)
    - [LLaVA](https://huggingface.co/spaces/badayvedat/LLaVA)

![Discriminative vs. Generative Models](./assets/discriminative_vs_generative.png)

### Exercise: Generate an Image with Stable Diffusion

In this section, we learn to play with Stable Diffusion XL 1.0.

To that end, the Gradio local UI from [Automatic1111 - Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) is installed and used.

![Automatic1111 - Stable Diffusion Web UI](./assets/stable_diffusion_ui_screenshot.png)

#### Setup

Instead of downloading and using the web UI locally, I used the Udacity Ubuntu VM.

    Udacity Menu Tab: Cloud Resources
    Start Cloud Resource + Open Cloud Console
    Provide any one of the following login password:

        Username: ubuntu
        Password: ubuntu
        Username: labuser
        Password: vocareum

    Ubuntu starts
    We can use the Terminal
    Clipboard: upper left corner; copy&paste between our machine and Ubuntu
    Install Python 3.10: done in Udacity
    Download Stable Diffusion Web UI: done un Udacity
    Udacity VM instructions:

        # Start Terminal + env
        cd automatic1111
        python3.10 -m venv venv
        source venv/bin/activate

        # Start app: takes 4 mins
        chmod +x webui.sh
        PYTHON=python3.10 ./webui.sh

        # Other Terminal: download weights
        wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors -O /home/ubuntu/automatic1111/stable-diffusion-webui/models/Stable-diffusion/sd_xl_base_1.0.safetensors

        # VM Browser
        http://127.0.0.1:7860
        http://localhost:7860

        # Load the previously downloaded model:
        Stable diffusion checkpoint (sd_xl_base_1.0.safetensors)

        # Provide prompt + generate!

    Remember to switch everything off -- we have 40 hours of usage.

For local installation and usage, check [AUTOMATIC1111/stable-diffusion-webui/installation-and-running](https://github.com/AUTOMATIC1111/stable-diffusion-webui?tab=readme-ov-file#installation-and-running).

Very first example tried:

> Prompt: *An astronaut riding a horse on a beach during the dawn*.  
> Paramaters: default
> ![An astronaut riding a horse on a beach during the dawn](./lab/generated_images/astronaut_horse_dawn.png)

### First Stage: Fixing the Random Seed

### Second Stage: Add Details

### Third Stage: Styles

### Fourth Stage: Negative Prompts

### Fifth Stage: "magic" Words

### Sixth Stage: Parameters

### Bonus Stage: Inpaint

### Other Examples, Links

- [https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki)
- [https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Extensions](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Extensions)
- [https://stablediffusion.fr/prompts](https://stablediffusion.fr/prompts)
- [https://stable-diffusion-art.com/automatic1111/](https://stable-diffusion-art.com/automatic1111/)


## 2. Computer Vision Fundamentals

TBD.

## 3. Transformer-Based Computer Vision Models

TBD.

## 4. Diffusion Models

TBD.

## 5. Project: AI Photo Editing with Inpainting

TBD.

