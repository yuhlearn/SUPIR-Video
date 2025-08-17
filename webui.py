import gradio as gr
import argparse
from SUPIR.models.model_wrapper import SUPIRWrapper
import os


def gradio_app(wrapper, server_ip, server_port):
    ds = wrapper.default_setting

    def load_and_reset():
        return (
            ds.prompt,
            ds.a_prompt,
            ds.n_prompt,
            ds.num_samples,
            ds.upscale,
            ds.edm_steps,
            ds.s_stage2,
            ds.s_cfg,
            ds.seed,
            ds.s_churn,
            ds.s_noise,
            ds.color_fix_type,
            ds.diff_dtype,
            ds.ae_dtype,
            ds.gamma_correction,
            ds.gaussian_sigma,
            ds.linear_CFG,
            ds.linear_s_stage2,
            ds.spt_linear_CFG,
            ds.spt_linear_s_stage2,
            ds.model_select,
            ds.min_size,
        )

    title_md = """
    # **SUPIR Video: Practicing Model Scaling for Photo-Realistic Video Restoration**
    """

    video_height = 540

    block = gr.Blocks(title="SUPIR Video").queue()
    with block:
        with gr.Row():
            gr.Markdown(title_md)

        with gr.Row():
            with gr.Column():
                with gr.Row(equal_height=True):
                    with gr.Column():
                        gr.Markdown("<center>Input</center>")
                        input_video = gr.Video(
                            sources=["upload"],
                            show_share_button=False,
                            height=video_height,
                            elem_id="video-input",
                        )
                prompt = gr.Textbox(label="Prompt", value="")

                with gr.Accordion("Settings", open=True):
                    min_size = gr.Slider(
                        label="Min size",
                        minimum=128,
                        maximum=1024,
                        value=ds.min_size,
                        step=8,
                    )
                    num_samples = gr.Slider(
                        label="Num Samples",
                        minimum=1,
                        maximum=10,
                        value=ds.num_samples,
                        step=1,
                    )
                    gamma_correction = gr.Slider(
                        label="Gamma Correction",
                        minimum=0.1,
                        maximum=2.0,
                        value=ds.gamma_correction,
                        step=0.1,
                    )
                    gaussian_sigma = gr.Slider(
                        label="Gaussian Blur Sigma",
                        minimum=0.0,
                        maximum=10.0,
                        value=ds.gaussian_sigma,
                        step=0.1,
                    )
                    upscale = gr.Slider(
                        label="Upscale", minimum=1, maximum=8, value=ds.upscale, step=1
                    )
                    edm_steps = gr.Slider(
                        label="Steps",
                        minimum=1,
                        maximum=200,
                        value=ds.edm_steps,
                        step=1,
                    )
                    s_cfg = gr.Slider(
                        label="Text Guidance Scale",
                        minimum=1.0,
                        maximum=15.0,
                        value=ds.s_cfg,
                        step=0.1,
                    )
                    s_stage2 = gr.Slider(
                        label="Guidance Strength",
                        minimum=0.0,
                        maximum=1.0,
                        value=ds.s_stage2,
                        step=0.05,
                    )
                    seed = gr.Slider(
                        label="Seed",
                        minimum=-1,
                        maximum=2147483647,
                        step=1,
                        value=ds.seed,
                    )
                    s_churn = gr.Slider(
                        label="S-Churn", minimum=0, maximum=40, value=ds.s_churn, step=1
                    )
                    s_noise = gr.Slider(
                        label="S-Noise",
                        minimum=1.0,
                        maximum=1.1,
                        value=ds.s_noise,
                        step=0.001,
                    )

                    with gr.Accordion("Default Prompt", open=False):
                        a_prompt = gr.Textbox(
                            label="Default Positive Prompt", value=ds.a_prompt
                        )
                        n_prompt = gr.Textbox(
                            label="Default Negative Prompt", value=ds.n_prompt
                        )

                    with gr.Accordion("CFG Settings", open=False):
                        with gr.Column():
                            linear_CFG = gr.Checkbox(
                                label="Linear CFG", value=ds.linear_CFG
                            )
                            spt_linear_CFG = gr.Slider(
                                label="CFG Start",
                                minimum=1.0,
                                maximum=9.0,
                                value=ds.spt_linear_CFG,
                                step=0.5,
                            )
                        with gr.Column():
                            linear_s_stage2 = gr.Checkbox(
                                label="Linear Stage2 Guidance", value=ds.linear_s_stage2
                            )
                            spt_linear_s_stage2 = gr.Slider(
                                label="Guidance Start",
                                minimum=0.0,
                                maximum=1.0,
                                value=ds.spt_linear_s_stage2,
                                step=0.05,
                            )

                    with gr.Accordion("Model Settings", open=False):
                        with gr.Column():
                            diff_dtype = gr.Radio(
                                ["fp32", "fp16", "bf16"],
                                label="Diffusion Data Type",
                                value=ds.diff_dtype,
                                interactive=True,
                            )
                        with gr.Column():
                            ae_dtype = gr.Radio(
                                ["fp32", "bf16"],
                                label="Auto-Encoder Data Type",
                                value=ds.ae_dtype,
                                interactive=True,
                            )
                        with gr.Column():
                            color_fix_type = gr.Radio(
                                ["None", "AdaIn", "Wavelet"],
                                label="Color-Fix Type",
                                value=ds.color_fix_type,
                                interactive=True,
                            )
                        with gr.Column():
                            model_select = gr.Radio(
                                ["v0-Q", "v0-F"],
                                label="Model Selection",
                                value=ds.model_select,
                                interactive=True,
                            )

                    with gr.Row():
                        with gr.Column():
                            reset_button = gr.Button(value="Reset Parameters", scale=2)

            with gr.Column():
                gr.Markdown("<center>Output</center>")
                processing_time = gr.Textbox(
                    container=True, interactive=False, visible=False
                )
                result_gallery = gr.Gallery(
                    label="Output",
                    show_label=True,
                    height=video_height,
                    elem_id="gallery1",
                )
                with gr.Row():
                    diffusion_button = gr.Button(value="Run")
                    abort_button = gr.Button(value="Abort")


        diffusion_event = diffusion_button.click(
            fn=wrapper.process,
            inputs=[
                input_video,
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
            ],
            outputs=[result_gallery, processing_time],
        )

        abort_button.click(
            cancels=diffusion_event
        )

        reset_button.click(
            fn=load_and_reset,
            inputs=[],
            outputs=[
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
            ],
        )

    block.launch(server_name=server_ip, server_port=server_port, inbrowser=True)


if __name__ == "__main__":
    # Reduce VRAM usage by reducing fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, default="options/SUPIR_v0.yaml")
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default="6688")
    parser.add_argument("--log_history", action="store_true", default=False)
    parser.add_argument("--loading_half_params", action="store_true", default=False)
    parser.add_argument("--use_tile_vae", action="store_true", default=False)
    parser.add_argument("--encoder_tile_size", type=int, default=512)
    parser.add_argument("--decoder_tile_size", type=int, default=64)
    parser.add_argument("--use_fp8_unet", action="store_true", default=False)
    parser.add_argument("--use_fp8_vae", action="store_true", default=False)
    args = parser.parse_args()

    wrapper = SUPIRWrapper(
        args.opt,
        args.loading_half_params,
        args.use_tile_vae,
        args.encoder_tile_size,
        args.decoder_tile_size,
        args.use_fp8_unet,
        args.use_fp8_vae,
    )

    gradio_app(wrapper, args.ip, args.port)
