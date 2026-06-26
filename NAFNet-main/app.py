
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import numpy as np
import cv2
import torch
from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img
from basicsr.utils.options import parse
from dotenv import load_dotenv

# Tải các biến môi trường từ file .env
load_dotenv()

# ---- Cache models ----
_models_cache = {}


def get_model(model_type):
    """Load and cache model"""
    if model_type in _models_cache:
        return _models_cache[model_type]

    if model_type == "GOPRO (Motion Blur)":
        opt_path = os.getenv("GOPRO_OPT_PATH", "options/test/GoPro/NAFNet-width64.yml")
        model_path = os.getenv("GOPRO_MODEL_PATH", "experiments/pretrained_models/NAFNet-GoPro-width64.pth")
    else:
        opt_path = os.getenv("REDS_OPT_PATH", "options/test/REDS/NAFNet-width64.yml")
        model_path = os.getenv("REDS_MODEL_PATH", "experiments/pretrained_models/NAFNet-REDS-width64.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    opt = parse(opt_path, is_train=False)
    opt["dist"] = False
    opt["num_gpu"] = 0
    opt["path"]["pretrain_network_g"] = model_path

    model = create_model(opt)
    _models_cache[model_type] = model
    return model


def deblur_image(input_image, model_type):
    """Process image through NAFNet"""
    if input_image is None:
        return None, " Vui lòng upload ảnh trước"

    try:
        model = get_model(model_type)

        # Convert to RGB numpy if needed
        if isinstance(input_image, np.ndarray):
            img = input_image.copy()
        else:
            img = np.array(input_image)

        # Gradio gives RGB, we need to handle correctly
        h, w = img.shape[:2]

        # Convert to tensor
        img_float = img.astype(np.float32) / 255.0
        tensor = _img2tensor(img_float, bgr2rgb=False, float32=True)

        # Inference
        model.feed_data(data={"lq": tensor.unsqueeze(dim=0)})
        model.test()
        visuals = model.get_current_visuals()
        result = tensor2img([visuals["result"]], rgb2bgr=False)

        # Calculate stats
        diff = np.abs(img.astype(np.float32) - result.astype(np.float32))
        mean_diff = np.mean(diff)
        pct_changed = 100 * np.sum(diff > 5) / diff.size

        info = (
            f" Deblur hoàn tất!\n"
            f" Kích thước: {w}×{h}\n"
            f" Pixel thay đổi: {pct_changed:.1f}%\n"
            f" Độ khác biệt TB: {mean_diff:.1f}/255"
        )

        return result, info

    except Exception as e:
        return None, f" Lỗi: {str(e)}"


# ---- Build UI ----
with gr.Blocks(title="NAFNet Image Deblurring") as demo:

    # Header
    gr.HTML("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="font-size: 2.5rem; font-weight: 800; margin: 0;" class="title-gradient">
             NAFNet Image Deblurring
        </h1>
        <p style="color: #64748b; font-size: 1.1rem; margin-top: 8px;">
            Khử mờ & khử nhiễu ảnh 
        </p>
    </div>
    """)

    # Main content
    with gr.Row(equal_height=True):
        # Left panel - Input
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Ảnh gốc")
            input_image = gr.Image(
                label="Kéo thả hoặc click để upload",
                type="numpy",
                height=380,
            )

            with gr.Row():
                model_type = gr.Dropdown(
                    choices=["REDS (Blur + JPEG)", "GOPRO (Motion Blur)"],
                    value="REDS (Blur + JPEG)",
                    label=" Chọn model",
                    scale=2,
                )
                process_btn = gr.Button(
                    "🔮 Deblur",
                    variant="primary",
                    scale=1,
                    size="lg",
                )

        # Right panel - Output
        with gr.Column(scale=1):
            gr.Markdown("### ✨ Kết quả")
            output_image = gr.Image(
                label="Ảnh đã deblur",
                type="numpy",
                height=380,
                interactive=False,
            )
            info_text = gr.Textbox(
                label=" Thông tin",
                lines=4,
                interactive=False,
                value=" Upload ảnh và nhấn Deblur để bắt đầu",
            )

    # Process button
    process_btn.click(
        fn=deblur_image,
        inputs=[input_image, model_type],
        outputs=[output_image, info_text],
    )


if __name__ == "__main__":
    print("=" * 50)
    print("  NAFNet Image Deblurring Web UI")
    print("  Pre-loading models...")
    print("=" * 50)

    # Pre-load both models
    try:
        print("  Loading REDS model...")
        get_model("REDS (Blur + JPEG)")
        print("  [OK] REDS model loaded")
    except Exception as e:
        print(f"  [WARN] REDS model: {e}")

    try:
        print("  Loading GoPro model...")
        get_model("GOPRO (Motion Blur)")
        print("  [OK] GoPro model loaded")
    except Exception as e:
        print(f"  [WARN] GoPro model: {e}")

    print("=" * 50)
    print("  Starting web UI at http://127.0.0.1:7860")
    print("=" * 50)

    css = """
    .title-gradient { 
        text-align: center; 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .info-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        margin-top: 8px;
    }
    footer { visibility: hidden }
    """

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css=css,
    )
