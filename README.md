# PaRa

This project is designed for generating and editing images using PaRa and LoRA models. It includes scripts for training models, generating images, and calculating various metrics.

## Setup

1. **Create a new Conda environment and activate it:**

    ```bash
    conda create --name newenv python=3.8
    conda activate newenv
    ```

2. **Install required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Download stable-diffusion-xl-base-1.0**

    Follow the instructions from the official repository https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 to download the Stable Diffusion XL Base 1.0 model.

4. **Download and setup Dreambooth dataset:**

    Clone the Dreambooth dataset repository:

    ```bash
    git clone https://github.com/google/dreambooth
    ```

## Training Models

1. **Train PaRa model for generation:**

    ```bash
    cd PaRa
    bash train_PaRa_model_for_generation.sh
    ```

2. **Train PaRa model for editing:**

    ```bash
    bash train_PaRa_model_for_editing.sh
    ```

## Image Generation

1. **Generate images using the ISRR model:**

    ```bash
    python demo_ISRR_generate_many_seed_ISRR.py bear_plushie
    python demo_ISRR_generate_many_seed_ISRR.py your_model_target_name
    ```

2. **Test PaRa model with different seeds:**

    ```bash
    bash test_PaRa_model_for_generation_seed_different.sh
    ```

3. **Images generated will be located in:**

    ```
    single_image_generate_ISRR_more_sample
    ```

## Jupyter Notebooks

1. **Generate images:**

    Use `demo_ISRR_generate.ipynb` to generate images. If the model is a one-shot model, it can also be used for image editing.

2. **Combine PaRa and PaRa effects:**

    Use `demo_ISRR_combine.ipynb` to show the combined effects of PaRa models, which involves multi-subject generation.

3. **Combine PaRa and LoRA effects:**

    Use `demo_ISRR_LoRA_combine.ipynb` to show the combined effects of PaRa and LoRA models.

## Calculating Metrics

1. **Calculate LPIPS score:**

    ```bash
    python lpip_score_calculate.py --original_path '/path/to/original_images' --edited_path '/path/to/edited_images'
    ```

    Example:

    ```bash
    python lpip_score_calculate.py --original_path '/home/ubuntu/data/datasets/dreambooth_paper_data/dreambooth/dataset/bear_plushie' --edited_path '/home/ubuntu/project/diffusion_model/concept_sliders/lora_as_controller/sliders/single_image_generate_ISRR_more_sample/prompt2_bear_plushie_rank_4'
    ```

2. **Calculate CLIP score:**

    ```bash
    python clip_score_calculate.py --prompt "your prompt here" --im_path /path/to/images
    ```

    Example:

    ```bash
    python clip_score_calculate.py --prompt "a bear" --im_path /home/ubuntu/project/diffusion_model/concept_sliders/lora_as_controller/sliders/single_image_generate_vanilla_SDXL_more_sample/bear_plushie_rank_4
    ```

3. **Calculate average SSIM:**

    ```bash
    python average_ssim.py /path/to/images
    ```

    Example:

    ```bash
    python average_ssim.py /home/ubuntu/project/diffusion_model/concept_sliders/lora_as_controller/sliders/single_image_generate_vanilla_SDXL_more_sample/bear_plushie_rank_4
    ```




