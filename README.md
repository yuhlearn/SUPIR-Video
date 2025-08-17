## (CVPR2024) Scaling Up to Excellence: Practicing Model Scaling for Photo-Realistic Image Restoration In the Wild

> [[Paper](https://arxiv.org/abs/2401.13627)] &emsp; [[Project Page](http://supir.xpixel.group/)] &emsp; [[Online App]](https://supir.suppixel.ai/home) <br>
> Fanghua, Yu, [Jinjin Gu](https://www.jasongt.com/), Zheyuan Li, Jinfan Hu, Xiangtao Kong, [Xintao Wang](https://xinntao.github.io/), [Jingwen He](https://scholar.google.com.hk/citations?user=GUxrycUAAAAJ), [Yu Qiao](https://scholar.google.com.hk/citations?user=gFtI-8QAAAAJ), [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ) <br>
> Shenzhen Institute of Advanced Technology; Shanghai AI Laboratory; University of Sydney; The Hong Kong Polytechnic University; ARC Lab, Tencent PCG; The Chinese University of Hong Kong <br>


---
## 🔧 Dependencies and Installation

1. Clone repo
    ```bash
    git clone https://github.com/Fanghua-Yu/SUPIR.git
    cd SUPIR
    ```

2. Install dependent packages
    ```bash
    conda create -n SUPIR python=3.10.9 -y
    conda activate SUPIR
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. Download Checkpoints

Store the following in `checkpoints`:
  
* [SDXL base 1.0_0.9vae](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors)
* [Juggernaut_RunDiffusionPhoto2_Lightning_4Steps](https://huggingface.co/RunDiffusion/Juggernaut-XL-Lightning/blob/main/Juggernaut_RunDiffusionPhoto2_Lightning_4Steps.safetensors)

Store the following in `supir_checkpoints`:

* [SUPIR-v0Q_fp16](https://huggingface.co/Kijai/SUPIR_pruned/blob/main/SUPIR-v0Q_fp16.safetensors)
    
    Default training settings with paper. High generalization and high image quality in most cases.

* [SUPIR-v0F_fp16](https://huggingface.co/Kijai/SUPIR_pruned/blob/main/SUPIR-v0F_fp16.safetensors)

    Training with light degradation settings. Stage1 encoder of `SUPIR-v0F` remains more details when facing light degradations.

4. To edit Custom Path for Checkpoints (if desired)
    ```
    * [CKPT_PTH.py] --> SDXL_CLIP1_PATH, SDXL_CLIP2_CACHE_DIR 
    * [options/SUPIR_v0.yaml] --> SDXL_CKPT, SUPIR_CKPT_Q, SUPIR_CKPT_F
    ```
  For users who can connect to huggingface, please leave `SDXL_CLIP1_PATH, SDXL_CLIP2_CKPT_PTH` in `CKPT_PTH.py` as `None`. These CLIPs will be downloaded automatically. 
  
  * [SDXL CLIP Encoder-1](https://huggingface.co/openai/clip-vit-large-patch14)
  * [SDXL CLIP Encoder-2](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
---

## Usage of SUPIR-Video

For limited VRAM, it is recommended to use the `--loading_half_params` and `--use_fp8_unet` flags. The `--use_fp8_vae` works,
but will significantly reduce the quality of the output and is therefore NOT recommended. 
`--use_tile_vae` works well with images, but is very slow for videos.

```Shell
usage: webui.py [-h] [--opt OPT] [--ip IP] [--port PORT] [--log_history] [--loading_half_params] [--use_tile_vae]
                [--encoder_tile_size ENCODER_TILE_SIZE] [--decoder_tile_size DECODER_TILE_SIZE] [--use_fp8_unet]
                [--use_fp8_vae]

options:
  -h, --help            show this help message and exit
  --opt OPT
  --ip IP
  --port PORT
  --log_history
  --loading_half_params
  --use_tile_vae
  --encoder_tile_size ENCODER_TILE_SIZE
  --decoder_tile_size DECODER_TILE_SIZE
  --use_fp8_unet
  --use_fp8_vae
```

### Empirical Hyperparameters Settings
TODO


## BibTeX
    @misc{yu2024scaling,
      title={Scaling Up to Excellence: Practicing Model Scaling for Photo-Realistic Image Restoration In the Wild}, 
      author={Fanghua Yu and Jinjin Gu and Zheyuan Li and Jinfan Hu and Xiangtao Kong and Xintao Wang and Jingwen He and Yu Qiao and Chao Dong},
      year={2024},
      eprint={2401.13627},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }


---
## Non-Commercial Use Only Declaration
The SUPIR ("Software") is made available for use, reproduction, and distribution strictly for non-commercial purposes. For the purposes of this declaration, "non-commercial" is defined as not primarily intended for or directed towards commercial advantage or monetary compensation.

By using, reproducing, or distributing the Software, you agree to abide by this restriction and not to use the Software for any commercial purposes without obtaining prior written permission from Dr. Jinjin Gu.

This declaration does not in any way limit the rights under any open source license that may apply to the Software; it solely adds a condition that the Software shall not be used for commercial purposes.

IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

For inquiries or to obtain permission for commercial use, please contact Dr. Jinjin Gu (jinjin.gu@suppixel.ai).
