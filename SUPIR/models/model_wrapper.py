# from mmgp import offload, profile_type
from SUPIR.util import HWC3, upscale_image, convert_dtype, create_SUPIR_model, load_QF_ckpt
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
from moviepy import VideoFileClip, Effect
from PROJ_PTH import OUTPUT_DIR
import gradio as gr
import numpy as np
import os
import torch
import einops
import copy
import time
import random


class SUPIRWrapper:
    def __init__(
        self,
        opt,
        loading_half_params,
        use_tile_vae,
        encoder_tile_size,
        decoder_tile_size,
        use_fp8_unet,
        use_fp8_vae,
    ):
        self.opt = opt
        self.loading_half_params = loading_half_params
        self.use_tile_vae = use_tile_vae
        self.encoder_tile_size = encoder_tile_size
        self.decoder_tile_size = decoder_tile_size
        self.use_fp8_unet = use_fp8_unet
        self.use_fp8_vae = use_fp8_vae
        self.fp8_dtype = torch.float8_e5m2

        if torch.cuda.device_count() >= 1:
            self.SUPIR_device = "cuda:0"
        else:
            raise ValueError("Currently support CUDA only.")

        # load SUPIR
        self.model, self.default_setting = create_SUPIR_model(
            opt, load_default_setting=True
        )

        if self.loading_half_params:
            self.model = self.model.half()
        if self.use_tile_vae:
            self.model.init_tile_vae(
                encoder_tile_size=self.encoder_tile_size,
                decoder_tile_size=self.decoder_tile_size,
            )

        self.model = self.model.to(self.SUPIR_device)
        self.model.first_stage_model.denoise_encoder_s1 = copy.deepcopy(
            self.model.first_stage_model.denoise_encoder
        )

        self.model.current_model = self.default_setting.model_select
        self.ckpt_Q, self.ckpt_F = load_QF_ckpt(opt)
        if self.model.current_model == "v0-Q":
            print("load v0-Q")
            self.model.load_state_dict(self.ckpt_Q, strict=False)
        elif self.model.current_model == "v0-F":
            print("load v0-F")
            self.model.load_state_dict(self.ckpt_F, strict=False)

        # offload.profile(model, profile_type.HighRAM_LowVRAM)

        if self.use_fp8_unet:
            self.model.model.to(self.fp8_dtype)
        if self.use_fp8_vae and not self.use_tile_vae:
            self.model.first_stage_model.to(self.fp8_dtype)

        torch.cuda.set_device(self.SUPIR_device)


    def select_model(self, model_select):
        if model_select != self.model.current_model:
            if model_select == "v0-Q":
                print("load v0-Q")
                self.model.load_state_dict(self.ckpt_Q, strict=False)
                self.model.current_model = "v0-Q"
            elif model_select == "v0-F":
                print("load v0-F")
                self.model.load_state_dict(self.ckpt_F, strict=False)
                self.model.current_model = "v0-F"
            if self.use_fp8_unet:
                self.model.model.to(self.fp8_dtype)


    def process(
        self,
        input_file,
        prompt,
        a_prompt,
        n_prompt,
        num_samples,
        upscale,
        edm_steps,
        s_stage2,
        s_cfg,
        seed,
        s_churn,
        s_noise,
        color_fix_type,
        diff_dtype,
        ae_dtype,
        gamma_correction,
        gaussian_sigma,
        linear_CFG,
        linear_s_stage2,
        spt_linear_CFG,
        spt_linear_s_stage2,
        model_select,
        min_size,
        progress=gr.Progress(),
    ):
        progress(None, desc="Starting")
        id = str(time.time_ns())
        output_name = id + "_" + os.path.basename(input_file) + ".mkv"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        input_clip = VideoFileClip(input_file)

        # Select the SUPIR model
        self.select_model(model_select)

        if seed == -1:
            seed = random.randint(0, 65535)

        def filter(get_frame, t):
            progress((t, input_clip.duration), unit="seconds", desc="Processing")

            # Process the video
            frame = get_frame(t)
            frame = upscale_image(frame, upscale, unit_resolution=32, min_size=min_size)

            if gaussian_sigma > 0:
                frame = gaussian_filter(frame, (gaussian_sigma, gaussian_sigma, 0))

            LQ = np.array(frame) / 255.0
            LQ = np.power(LQ, gamma_correction)
            LQ *= 255.0
            LQ = LQ.round().clip(0, 255).astype(np.uint8)
            LQ = LQ / 255 * 2 - 1
            LQ = (
                torch.tensor(LQ, dtype=torch.float32)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.SUPIR_device)[:, :3, :, :]
            )
            captions = [prompt]

            self.model.ae_dtype = convert_dtype(ae_dtype)
            self.model.model.dtype = convert_dtype(diff_dtype)

            samples = self.model.batchify_sample(
                LQ,
                captions,
                num_steps=edm_steps,
                s_churn=s_churn,
                s_noise=s_noise,
                cfg_scale=s_cfg,
                control_scale=s_stage2,
                seed=seed,
                num_samples=num_samples,
                p_p=a_prompt,
                n_p=n_prompt,
                color_fix_type=color_fix_type,
                use_linear_CFG=linear_CFG,
                use_linear_control_scale=linear_s_stage2,
                cfg_scale_start=spt_linear_CFG,
                control_scale_start=spt_linear_s_stage2,
            )

            x_samples = (
                (einops.rearrange(samples, "b c h w -> b h w c") * 127.5 + 127.5)
                .cpu()
                .numpy()
                .round()
                .clip(0, 255)
                .astype(np.uint8)
            )
            results = [x_samples[i] for i in range(num_samples)]

            return results[0]
        

        # Apply the filter
        try:
            output_clip = input_clip.transform(filter, keep_duration=True)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise

        progress((input_clip.duration, input_clip.duration), desc="Finished")

        output_clip.write_videofile(
            filename=output_path,
            codec="libx264",
            preset="medium",
            ffmpeg_params=["-crf", "1"],
        )

        return [output_path], [output_name]
