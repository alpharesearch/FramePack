#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import traceback
import einops
import numpy as np
import math
import torch
from PIL import Image, PngImagePlugin
import logging
import subprocess
import glob
import re
from pathlib import Path
import time
import unicodedata
import hashlib
from datetime import datetime


# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Environment Setup ---
# Set HF_HOME *before* importing transformers/diffusers if needed
# Using an absolute path relative to the script's location
# Handle case where __file__ might not be defined (e.g., interactive)
script_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
hf_download_path = os.path.abspath(os.path.realpath(os.path.join(script_dir, './hf_download')))
os.environ['HF_HOME'] = hf_download_path
os.makedirs(hf_download_path, exist_ok=True)
logging.info(f"HF_HOME set to: {hf_download_path}")

# --- Imports (Grouped after HF_HOME setup) ---
# Suppress excessive logging from libraries if desired
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("diffusers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


# Assuming these are custom helper modules/classes from the original project
# Make sure these files are accessible in your Python path
try:
    from diffusers import AutoencoderKLHunyuanVideo
    from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
    from transformers import SiglipImageProcessor, SiglipVisionModel
    # --- Custom Helpers (Ensure these exist in your project structure) ---
    from diffusers_helper.hf_login import login # Keep if needed for gated models
    from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
    from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, generate_timestamp
    from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
    from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
    from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
    from diffusers_helper.clip_vision import hf_clip_vision_encode
    from diffusers_helper.bucket_tools import find_nearest_bucket
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    logging.error("Please ensure 'diffusers', 'transformers', 'torch', 'numpy', 'Pillow', 'einops'")
    logging.error("and the custom 'diffusers_helper' package/modules are correctly installed and accessible.")
    exit(1)


# --- Filename Helper Functions ---

ADJECTIVES = [
    'brave', 'curious', 'gentle', 'swift', 'sneaky', 'bright', 'clever', 'bold',
    'sleepy', 'grumpy', 'witty', 'fancy', 'fuzzy', 'shiny', 'lucky', 'stormy',
    'mighty', 'tiny', 'wild', 'zany', 'quirky', 'crafty', 'spooky', 'jazzy',
    'calm', 'noisy', 'bouncy', 'cosmic', 'electric', 'dreamy', 'silent', 'fiery',
    'icy', 'glowing', 'frosty', 'sunny', 'dusky', 'shadowy', 'glorious', 'radiant',
    'haunted', 'twinkly', 'graceful', 'magnetic', 'restless', 'vivid', 'enigmatic',
    'loopy', 'silly', 'fierce', 'loyal', 'cheery', 'whimsical', 'mystic', 'chill',
    'howling', 'playful', 'feisty', 'epic', 'invisible', 'vibrant', 'rogue'
]

NOUNS = [
    'dolphin', 'falcon', 'otter', 'panther', 'tiger', 'koala', 'eagle', 'fox',
    'lion', 'dragon', 'phoenix', 'griffin', 'unicorn', 'badger', 'raven', 'panda',
    'wolf', 'elk', 'hawk', 'rhino', 'lizard', 'shark', 'whale', 'chameleon',
    'goblin', 'wizard', 'witch', 'robot', 'ghost', 'zombie', 'pirate', 'ninja',
    'samurai', 'giant', 'pixie', 'fairy', 'kraken', 'slug', 'gecko', 'slugcat',
    'moose', 'beetle', 'cricket', 'snail', 'spider', 'glitch', 'sprite', 'moth',
    'golem', 'chimera', 'serpent', 'wyvern', 'yetian', 'basilisk', 'mermaid',
    'gargoyle', 'centaur', 'hydra', 'cyclops', 'minotaur', 'wisp', 'cloud'
]


def get_deterministic_fun_name(seed_text):
    """Generate a fun, deterministic name based on a hash of the input."""
    h = hashlib.md5(seed_text.encode()).hexdigest()
    adj_index = int(h[:2], 16) % len(ADJECTIVES)
    noun_index = int(h[2:4], 16) % len(NOUNS)
    suffix = h[4:8]  # Add some uniqueness without being too long
    return f"{ADJECTIVES[adj_index]}_{NOUNS[noun_index]}_{suffix}"

def sanitize_filename(filename, max_len=50):
    """
    Sanitizes a filename by:
    - Removing extension
    - Replacing problematic characters
    - Removing disallowed characters
    - Truncating safely
    - Falling back to a fun name if empty
    """
    if not filename:
        filename = "file"
    # Normalize and strip
    filename = unicodedata.normalize('NFKD', filename).strip()
    # Remove extension
    base = os.path.splitext(filename)[0]
    # Replace problem characters with underscores
    base = re.sub(r'[\\/*?:"<>| \t\r\n]+', '_', base)
    # Keep only alphanumeric, underscore, hyphen
    base = re.sub(r'[^\w-]', '', base)
    # Truncate and clean
    base = base[:max_len].rstrip('_-')
    # Fallback: deterministic fun name
    if not base:
        base = get_deterministic_fun_name(filename)
    return base

def get_output_basename(input_image_path, job_id):
    """Creates the base part of the output filename like {input_basename}_{timestamp}."""
    input_filename_base = Path(input_image_path).name
    sanitized_input_name = sanitize_filename(input_filename_base)
    return f"{sanitized_input_name}_{job_id}"


# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="FramePack CLI Video Generation Tool", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- Core Input ---
    parser.add_argument("input_path", type=str, help="Path to the input image file OR a directory containing image files.")

    # --- Processing Options ---
    parser.add_argument("--prompt", type=str,
                        default='a flirty character doing some seductive body movements',
                        help="Text prompt describing the desired video content. Used for all images in batch mode unless overridden.")
    parser.add_argument("--create_loop", action=argparse.BooleanOptionalAction, default=True,
                        help="Automatically create a looping version (_final_loop.mp4) using ffmpeg after generation.")
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True,
                        help="In directory mode, skip processing an image if a corresponding output file (e.g., *_final_loop.mp4 or *_final.mp4) already exists.")
    parser.add_argument("--interactive_select", action='store_true',
                        help="In directory mode, interactively ask which detected image files to process.")

    # --- Generation Parameters ---
    parser.add_argument("--output_dir", type=str, default="./outputs/", help="Directory to save output videos and intermediate files.")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt (currently not heavily used by the model).")
    parser.add_argument("--seed", type=int, default=31337, help="Initial seed for generation, unless seed is -1, which uses a random seed.")
    parser.add_argument("--length", type=float, default=5.0, help="Total desired video length in seconds.")
    parser.add_argument("--steps", type=int, default=25, help="Number of diffusion steps (changing not recommended).")
    parser.add_argument("--gs", type=float, default=10.0, help="Distilled CFG Scale (changing not recommended).")

    # --- Performance & Quality ---
    parser.add_argument("--gpu_memory_preservation", type=float, default=6.0, help="GPU memory (GB) to preserve during inference (larger means slower). Increase if OOM.")
    parser.add_argument("--use_teacache", action=argparse.BooleanOptionalAction, default=False, help="Enable TeaCache optimization (faster, may slightly affect hands/fingers).")
    parser.add_argument("--mp4_crf", type=int, default=16, help="MP4 Constant Rate Factor (lower means better quality/larger file, 0=lossless, ~16-23 good balance).")
    parser.add_argument("--high_vram_threshold", type=float, default=60.0, help="VRAM amount (GB) to consider 'high VRAM mode'.")

    # --- External Tools & Setup ---
    parser.add_argument("--login_hf", action='store_true', help="Attempt Hugging Face login (needed for gated models).")
    parser.add_argument("--ffmpeg_path", type=str, default="ffmpeg", help="Path to the ffmpeg executable.")

    # --- Fixed internal defaults (originally hidden sliders) ---
    parser.set_defaults(latent_window_size=9)
    parser.set_defaults(cfg=1.0)
    parser.set_defaults(rs=0.0)

    args = parser.parse_args()

    # --- Input Validation ---
    if not Path(args.input_path).exists():
         logging.error(f"Input path not found: {args.input_path}")
         exit(1)

    if args.length <= 0:
        logging.error("Video length must be positive.")
        exit(1)
    if not 0 <= args.mp4_crf <= 63: # CRF range more typical for x264/x265 is ~0-51, but allow wider range.
        logging.warning(f"MP4 CRF value {args.mp4_crf} is outside the typical 0-51 range. Ensure it's valid for your ffmpeg setup.")

    # Ensure output directory exists *before* processing starts
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        args.output_dir = os.path.abspath(args.output_dir) # Use absolute path
        logging.info(f"Using output directory: {args.output_dir}")
    except OSError as e:
        logging.error(f"Failed to create or access output directory '{args.output_dir}': {e}")
        exit(1)

    return args

# --- Video Generation Worker ---
@torch.no_grad()
def generate_video(args, models, device_info, input_image_path, output_basename, current_seed):
    """Generates video based on parsed arguments and loaded models for a SINGLE image."""
    prompt = args.prompt
    n_prompt = args.negative_prompt
    total_second_length = args.length
    latent_window_size = args.latent_window_size # From defaults
    steps = args.steps
    cfg = args.cfg # From defaults
    gs = args.gs
    rs = args.rs # From defaults
    gpu_memory_preservation = args.gpu_memory_preservation
    use_teacache = args.use_teacache
    mp4_crf = args.mp4_crf
    outputs_folder = args.output_dir # Absolute path
    high_vram = device_info['high_vram']

    # Unpack models
    text_encoder = models['text_encoder']
    text_encoder_2 = models['text_encoder_2']
    tokenizer = models['tokenizer']
    tokenizer_2 = models['tokenizer_2']
    vae = models['vae']
    feature_extractor = models['feature_extractor']
    image_encoder = models['image_encoder']
    transformer = models['transformer']

    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4) # Assuming 30 fps base
    total_latent_sections = int(max(round(total_latent_sections), 1))

    logging.info(f"Calculated total latent sections: {total_latent_sections}")
    logging.info(f"Output base name: {output_basename}")
    logging.info(f"Using seed: {current_seed}")

    try:
        # --- Preprocessing ---
        # Clean GPU if needed (low VRAM mode)
        if not high_vram:
            logging.debug("Low VRAM mode: Unloading models before starting.")
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # --- Text Encoding ---
        logging.debug("Encoding text prompts...")
        if not high_vram:
            logging.debug("Low VRAM mode: Loading text encoders...")
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        logging.debug(f"Encoded main prompt. Shape: {llama_vec.shape}")

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        logging.debug(f"Encoded negative prompt. Shape: {llama_vec_n.shape}")

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # --- Image Processing ---
        logging.debug(f"Processing input image: {input_image_path}")
        try:
            # Ensure image is loaded correctly even from Path object
            with Image.open(str(input_image_path)) as img:
                input_pil_image = img.convert('RGB')
            input_image_np = np.array(input_pil_image)
        except Exception as e:
            logging.error(f"Failed to load or process input image '{input_image_path}': {e}")
            return None # Indicate failure

        H, W, C = input_image_np.shape
        height, width = find_nearest_bucket(H, W, resolution=640) # Assuming 640 default
        logging.debug(f"Original image size: {W}x{H}. Resizing to nearest bucket: {width}x{height}")
        input_image_resized_np = resize_and_center_crop(input_image_np, target_width=width, target_height=height)

        # Save the processed input image for reference - USE NEW BASENAME
        input_save_path = os.path.join(outputs_folder, f'{output_basename}_input.png')
        
        # Create metadata
        meta = PngImagePlugin.PngInfo()
        meta.add_text("Title", f"Input Image - {output_basename}")
        meta.add_text("Description", "This is the processed input image before inference.")
        meta.add_text("Author", "FramePack")
        meta.add_text("Source", "Auto-generated by FramePack")
        meta.add_text("Timestamp", datetime.now().isoformat())
        meta.add_text("Prompt", prompt)
        meta.add_text("Seed", str(current_seed))
        try:
            Image.fromarray(input_image_resized_np).save(input_save_path, pnginfo=meta)
            logging.debug(f"Saved processed input image to: {input_save_path}")
        except Exception as e:
             logging.warning(f"Could not save processed input image '{input_save_path}': {e}")


        input_image_pt = torch.from_numpy(input_image_resized_np).float() / 127.5 - 1.0
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None] # Add Batch and Time dimensions

        # --- VAE Encoding ---
        logging.debug("Encoding image with VAE...")
        if not high_vram:
            logging.debug("Low VRAM mode: Loading VAE...")
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)
        logging.debug(f"VAE encoded latent shape: {start_latent.shape}")

        # --- CLIP Vision Encoding ---
        logging.debug("Encoding image with CLIP Vision...")
        if not high_vram:
            logging.debug("Low VRAM mode: Loading Image Encoder...")
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_resized_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        logging.debug(f"CLIP Vision encoded state shape: {image_encoder_last_hidden_state.shape}")

        # --- Dtype Conversion ---
        logging.debug(f"Converting embeddings to transformer dtype: {transformer.dtype}")
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # --- Sampling Loop ---
        logging.info("Starting sampling process...") # Keep this info level

        rnd = torch.Generator(device="cpu").manual_seed(current_seed)
        num_frames = latent_window_size * 4 - 3

        # Initialize history on CPU
        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0
        final_output_filename = os.path.join(outputs_folder, f'{output_basename}_final.mp4') # Define final name upfront

        # Determine padding sequence
        latent_paddings = list(reversed(range(total_latent_sections)))
        if total_latent_sections > 4:
             latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
             logging.debug(f"Using modified padding sequence for long video: {latent_paddings}")
        else:
             logging.debug(f"Using standard padding sequence: {latent_paddings}")

        for i, latent_padding in enumerate(latent_paddings):
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            logging.info(f"--- Sampling Section {i+1}/{total_latent_sections} (Padding: {latent_padding}, Last: {is_last_section}) ---")

            # --- Index Calculation ---
            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents.device, dtype=history_latents.dtype)
            split_sizes = [1, 2, 16]
            current_hist_len = history_latents.shape[2]
            if current_hist_len < sum(split_sizes):
                 logging.warning(f"History latents time dimension ({current_hist_len}) is smaller than expected split size ({sum(split_sizes)}). Adjusting split.")
                 split_sizes = [min(1, current_hist_len),
                                min(2, max(0, current_hist_len-1)),
                                max(0, current_hist_len-1-2) ]
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :sum(split_sizes), :, :].split(split_sizes, dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            # --- Load Transformer (Low VRAM) ---
            if not high_vram:
                logging.debug("Low VRAM mode: Unloading other models and loading Transformer...")
                unload_complete_models() # Unload VAE, encoders if they were loaded
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            # --- TeaCache Setup ---
            if use_teacache:
                logging.debug("Initializing TeaCache...")
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                logging.debug("TeaCache disabled.")
                transformer.initialize_teacache(enable_teacache=False)

            # --- K-Diffusion Sampling Callback (for CLI progress) ---
            def callback(d):
                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                print(f"\r  Sampling Step: {current_step}/{steps} [{percentage}%]...", end="", flush=True)
                if current_step == steps:
                    print() # Newline after finishing steps for this section

            # --- Run Sampler ---
            logging.debug(f"Running sampler for {num_frames} frames with height={height}, width={width}...")
            generated_latents = sample_hunyuan(
                transformer=transformer, sampler='unipc', width=width, height=height, frames=num_frames,
                real_guidance_scale=cfg, distilled_guidance_scale=gs, guidance_rescale=rs,
                num_inference_steps=steps, generator=rnd,
                prompt_embeds=llama_vec, prompt_embeds_mask=llama_attention_mask, prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n, negative_prompt_embeds_mask=llama_attention_mask_n, negative_prompt_poolers=clip_l_pooler_n,
                device=gpu, dtype=torch.bfloat16, # Match transformer dtype
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents, clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x, clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x, clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )
            logging.debug(f"Generated latents shape for this section: {generated_latents.shape}")

            # --- Post-Sampling ---
            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents.device, dtype=generated_latents.dtype), generated_latents], dim=2)
                logging.debug("Prepended start latent to the final section.")

            total_generated_latent_frames += int(generated_latents.shape[2])
            # Update history (move generated to CPU to potentially save VRAM)
            history_latents = torch.cat([generated_latents.cpu().float(), history_latents], dim=2) # Prepend new latents
            logging.debug(f"Total generated latent frames so far: {total_generated_latent_frames}")

            # --- Offload Transformer / Load VAE (Low VRAM) ---
            if not high_vram:
                logging.debug("Low VRAM mode: Offloading Transformer and loading VAE...")
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=max(8.0, gpu_memory_preservation))
                load_model_as_complete(vae, target_device=gpu)

            # --- VAE Decoding & Stitching ---
            logging.debug("Decoding latents with VAE...")
            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None: # First section
                history_pixels = vae_decode(real_history_latents, vae).cpu()
                logging.debug(f"Decoded first section. Pixel shape: {history_pixels.shape}")
            else: # Subsequent sections
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3
                current_latents_to_decode = real_history_latents[:, :, :section_latent_frames, :, :]
                current_pixels = vae_decode(current_latents_to_decode, vae).cpu()
                logging.debug(f"Decoding section for stitching. Shape: {current_pixels.shape}, Overlap: {overlapped_frames}")
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
                logging.debug(f"Stitched video. New pixel shape: {history_pixels.shape}")

            # --- Offload VAE (Low VRAM) ---
            if not high_vram:
                logging.debug("Low VRAM mode: Unloading VAE.")
                unload_complete_models(vae)

            # --- Save Intermediate Video ---
            intermediate_filename = os.path.join(outputs_folder, f'{output_basename}_progress_{i+1}.mp4')
            save_bcthw_as_mp4(history_pixels, intermediate_filename, fps=30, crf=mp4_crf) # Assuming 30 fps default
            logging.info(f"Saved intermediate video: {Path(intermediate_filename).name}")

            if is_last_section:
                # Save final version (already defined path)
                save_bcthw_as_mp4(history_pixels, final_output_filename, fps=30, crf=mp4_crf)
                logging.info(f"Saved FINAL video: {Path(final_output_filename).name}")
                break # Exit loop

        logging.info("--- Sampling and Decoding Finished ---")
        return final_output_filename # Return the path to the final video

    except Exception as e:
        logging.error(f"An error occurred during video generation for {Path(input_image_path).name}: {e}")
        traceback.print_exc()
        return None # Indicate failure
    finally:
        # Final cleanup (especially important in low VRAM)
        if not high_vram:
            logging.debug("Low VRAM mode: Unloading all models after completion.")
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )
        # Clear CUDA cache potentially
        if torch.cuda.is_available():
             torch.cuda.empty_cache()
             logging.debug("Cleared CUDA cache.")

# --- FFmpeg Looping Function ---
def create_looping_video(final_video_path, output_basename, args):
    """
    Creates a looping video (forward -> reverse) using ffmpeg.
    """
    if not final_video_path or not os.path.exists(final_video_path):
        logging.error(f"Final video path '{final_video_path}' is invalid, cannot create loop.")
        return None

    loop_output_filename = os.path.join(args.output_dir, f'{output_basename}_final_loop.mp4')
    logging.info(f"Attempting to create forward-reverse video: {Path(loop_output_filename).name}")

    # Check if ffmpeg and ffprobe are available by trying to get version
    try:
        ffmpeg_ver = subprocess.run([args.ffmpeg_path, "-version"], capture_output=True, check=True, text=True, encoding='utf-8', errors='ignore').stdout.split('\n')[0]
        logging.debug(f"Found ffmpeg: {ffmpeg_ver}")
    except FileNotFoundError as e:
        logging.error(f"Executable not found: {e.filename}. Please ensure ffmpeg is in your PATH or specify --ffmpeg-path.")
        return None
    except subprocess.CalledProcessError as e:
         logging.error(f"Error checking ffmpeg version: {e}. Check paths and permissions.")
         return None
    except Exception as e:
         logging.error(f"Unexpected error checking ffmpeg: {e}")
         return None

    # --- No need to get frame count for this simpler filter ---
    # Frame count was only needed for the 'loop' filter's 'size' parameter

    # Construct and run ffmpeg command for forward -> reverse
    try:
        # Filter graph: Reverse the input, then concatenate original and reversed streams.
        filter_graph = f"[0:v]reverse[r];[0:v][r]concat=n=2:v=1:a=0[c];[c]setpts=N/(30*TB)[v_final]" # Assuming 30fps base

        ffmpeg_cmd = [
            args.ffmpeg_path,
            '-y',                     # Overwrite output files (use -n to prevent)
            '-i', final_video_path,
            '-filter_complex', filter_graph,
            '-map', '[v_final]',      # Map the final filtered video stream
            '-an',                    # No audio in the output
            '-c:v', 'libx264',        # Specify a common encoder
            '-preset', 'medium',      # Encoding speed/compression tradeoff
            '-crf', str(args.mp4_crf),# Use specified CRF for output quality
            '-movflags', '+faststart', # Good practice for web video
            loop_output_filename
        ]
        logging.info("Running ffmpeg forward-reverse command...")
        # print(" ".join(ffmpeg_cmd)) # For debugging
        result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        logging.debug(f"ffmpeg stdout:\n{result.stdout}")
        logging.debug(f"ffmpeg stderr:\n{result.stderr}")
        logging.info(f"Successfully created forward-reverse video: {Path(loop_output_filename).name}")
        return loop_output_filename

    except subprocess.CalledProcessError as e:
        logging.error(f"ffmpeg command failed for '{Path(final_video_path).name}':")
        logging.error(f"Return Code: {e.returncode}")
        logging.error(f"ffmpeg stdout:\n{e.stdout}")
        logging.error(f"ffmpeg stderr:\n{e.stderr}")
        if os.path.exists(loop_output_filename):
            try: os.remove(loop_output_filename)
            except OSError: pass
        return None
    except Exception as e:
         logging.error(f"Unexpected error running ffmpeg for '{Path(final_video_path).name}': {e}")
         if os.path.exists(loop_output_filename):
             try: os.remove(loop_output_filename)
             except OSError: pass
         return None

# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()

    # --- Hugging Face Login (Optional) ---
    if args.login_hf:
        logging.info("Attempting Hugging Face login...")
        try:
            login()
            logging.info("Hugging Face login successful or token found.")
        except Exception as e:
            logging.warning(f"Hugging Face login failed: {e}. This might be okay if models are public.")

    # --- Device Check ---
    if not torch.cuda.is_available():
        logging.error("CUDA not available. This model requires a GPU.")
        exit(1)

    try:
        free_mem_gb = get_cuda_free_memory_gb(gpu) # gpu is likely torch.device('cuda:0')
        gpu_name = torch.cuda.get_device_name(gpu)
    except Exception as e:
        logging.error(f"Failed to get CUDA device info: {e}")
        exit(1)

    high_vram = free_mem_gb > args.high_vram_threshold
    device_info = {'high_vram': high_vram, 'free_mem_gb': free_mem_gb}

    logging.info(f"Detected GPU: {gpu_name}")
    logging.info(f"Free VRAM: {free_mem_gb:.2f} GB")
    logging.info(f"High-VRAM Mode Active: {high_vram}")

    # --- Model Loading ---
    logging.info("Loading models... (This may take a while)")
    try:
        # Load models to CPU first, manage device placement later
        text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
        text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
        tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
        tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
        vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

        feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
        image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

        transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu() # Use bfloat16 as per original

        logging.info("Models loaded to CPU.")

        # Set eval mode and disable gradients
        vae.eval()
        text_encoder.eval()
        text_encoder_2.eval()
        image_encoder.eval()
        transformer.eval()

        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        text_encoder_2.requires_grad_(False)
        image_encoder.requires_grad_(False)
        transformer.requires_grad_(False)

        # Apply VAE optimizations
        vae.enable_slicing()
        vae.enable_tiling()
        logging.debug("Enabled VAE slicing and tiling.")

        # Apply specific transformer setting from original code
        transformer.high_quality_fp32_output_for_inference = True
        logging.debug('Set transformer.high_quality_fp32_output_for_inference = True')

        # --- Initial Device Placement ---
        transformer.to(dtype=torch.bfloat16) # Ensure correct dtype before moving
        vae.to(dtype=torch.float16)
        image_encoder.to(dtype=torch.float16)
        text_encoder.to(dtype=torch.float16)
        text_encoder_2.to(dtype=torch.float16)

        if high_vram:
            logging.info("High VRAM mode: Moving all models to GPU...")
            text_encoder.to(gpu)
            text_encoder_2.to(gpu)
            image_encoder.to(gpu)
            vae.to(gpu)
            transformer.to(gpu)
            logging.info("All models moved to GPU.")
        else:
            logging.info("Low VRAM mode: Setting up dynamic swapping...")
            DynamicSwapInstaller.install_model(transformer, device=gpu)
            DynamicSwapInstaller.install_model(text_encoder, device=gpu)
            logging.info("Dynamic swapping installed for Transformer and primary Text Encoder.")

        models = {
            'text_encoder': text_encoder, 'text_encoder_2': text_encoder_2,
            'tokenizer': tokenizer, 'tokenizer_2': tokenizer_2, 'vae': vae,
            'feature_extractor': feature_extractor, 'image_encoder': image_encoder,
            'transformer': transformer
        }
        logging.info("Model setup and device placement complete.")

    except Exception as e:
        logging.error(f"Failed to load or setup models: {e}")
        traceback.print_exc()
        exit(1)

    # --- Identify Input Files ---
    input_path = Path(args.input_path)
    image_files_to_process = []
    valid_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp'] # Case-insensitive check

    if input_path.is_file():
        if input_path.suffix.lower() in valid_extensions:
            image_files_to_process.append(str(input_path))
        else:
            logging.error(f"Input file '{args.input_path}' is not a supported image type ({', '.join(valid_extensions)}).")
            exit(1)
    elif input_path.is_dir():
        logging.info(f"Scanning directory for images: {args.input_path}")
        # Use glob to find files directly
        all_found_files = []
        for ext in valid_extensions:
            all_found_files.extend(input_path.glob(f"*{ext}"))
            all_found_files.extend(input_path.glob(f"*{ext.upper()}")) # Include uppercase extensions

        # Filter unique files and sort
        all_files = sorted(list(set(f for f in all_found_files if f.is_file())))

        if not all_files:
            logging.warning(f"No supported image files found in directory: {args.input_path}")
            exit(0)

        logging.info(f"Found {len(all_files)} potential image files.")

        if args.interactive_select:
            print("\nFound Images:")
            for i, f in enumerate(all_files):
                print(f"  {i+1}: {f.name}")

            while True:
                try:
                    selection = input(f"Enter numbers (1-{len(all_files)}) to process (e.g., 1,3,5-7), or 'all': ").strip().lower()
                    if not selection: continue # Ask again if empty
                    if selection == 'all':
                        image_files_to_process = [str(f) for f in all_files]
                        break
                    else:
                        selected_indices = set()
                        parts = selection.split(',')
                        for part in parts:
                            part = part.strip()
                            if not part: continue
                            if '-' in part:
                                start_str, end_str = part.split('-', 1)
                                start = int(start_str)
                                end = int(end_str)
                                if 1 <= start <= end <= len(all_files):
                                    selected_indices.update(range(start - 1, end)) # end is inclusive here
                                else:
                                    raise ValueError(f"Invalid range '{part}'. Must be between 1 and {len(all_files)}.")
                            else:
                                index = int(part)
                                if 1 <= index <= len(all_files):
                                    selected_indices.add(index - 1)
                                else:
                                     raise ValueError(f"Invalid index '{part}'. Must be between 1 and {len(all_files)}.")
                        if not selected_indices:
                             print("No valid selections made.")
                             continue
                        image_files_to_process = [str(all_files[i]) for i in sorted(list(selected_indices))]
                        break
                except ValueError as e:
                    print(f"Invalid input: {e} Please try again.")
        else:
            image_files_to_process = [str(f) for f in all_files] # Process all if not interactive

    else:
        logging.error(f"Input path '{args.input_path}' is neither a valid file nor a directory.")
        exit(1)

    # --- Processing Loop ---
    total_files = len(image_files_to_process)
    if total_files == 0:
        logging.info("No image files selected for processing.")
        exit(0)

    logging.info(f"Starting processing for {total_files} image(s)...")
    success_count = 0
    skipped_count = 0
    failed_count = 0
    initial_seed = args.seed # Store the original seed argument

    for index, image_path_str in enumerate(image_files_to_process):
        image_path = Path(image_path_str)
        logging.info(f"\n{'='*10} Processing file {index+1}/{total_files}: {image_path.name} {'='*10}")

        # Generate unique job ID (timestamp) for this specific file processing run
        job_id = generate_timestamp() # Get a fresh timestamp
        output_basename = get_output_basename(image_path_str, job_id)

        # Determine the seed for this specific image
        if initial_seed == -1:
             current_seed = np.random.randint(0, 2**32 - 1) # Generate random seed
        else:
             current_seed = initial_seed

        # Check if final output already exists (skip logic)
        if args.skip_existing:
            target_suffix = "_final_loop.mp4" if args.create_loop else "_final.mp4"
            # Look for *any* file starting with the sanitized input name and ending with the target suffix
            sanitized_input_name = sanitize_filename(image_path.name)
            pattern_to_check = f"{sanitized_input_name}*{target_suffix}"
            # Use Path(args.output_dir) for globbing
            existing_files = list(Path(args.output_dir).glob(pattern_to_check))

            if existing_files:
                 logging.info(f"Skipping '{image_path.name}' because existing output file matching pattern '{pattern_to_check}' found: {existing_files[0].name}")
                 skipped_count += 1
                 continue # Skip to the next image

        # --- Run Generation for this image ---
        final_video_path = None # Ensure it's defined
        try:
            # Pass the absolute path string
            final_video_path = generate_video(args, models, device_info, image_path_str, output_basename, current_seed)

            if final_video_path and os.path.exists(final_video_path):
                 logging.info(f"Successfully generated: {Path(final_video_path).name}")
                 success_count += 1
                 # --- Optionally Create Loop ---
                 if args.create_loop:
                      loop_video_path = create_looping_video(final_video_path, output_basename, args)
                      if not loop_video_path:
                          logging.warning(f"Failed to create loop video for {Path(final_video_path).name}, but main generation succeeded.")
            else:
                 logging.error(f"Video generation reported failure for {image_path.name}.")
                 failed_count += 1

        except Exception as e:
             logging.error(f"An unexpected critical error occurred processing {image_path.name}: {e}")
             traceback.print_exc()
             failed_count += 1
             # Optionally, add a small delay before processing next file after an error
             # time.sleep(2)


    # --- Finish ---
    logging.info(f"\n{'='*10} Batch Processing Summary {'='*10}")
    logging.info(f"Total images selected: {total_files}")
    logging.info(f"Successfully processed: {success_count}")
    logging.info(f"Skipped (output found): {skipped_count}")
    logging.info(f"Failed: {failed_count}")
    logging.info(f"{'='*10} Script Finished {'='*10}")
