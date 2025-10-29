

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

matplotlib.use('QtAgg')
from math import ceil
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

def manual_resize(image: Image.Image, target_width: int, target_height: int, patch_size: int = 16) -> Image.Image:
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Target dimensions must be positive")

    rounded_width = ceil(target_width / patch_size) * patch_size
    rounded_height = ceil(target_height / patch_size) * patch_size

    scale = max(rounded_width / image.width, rounded_height / image.height)
    resized_width = max(patch_size, ceil((image.width * scale) / patch_size) * patch_size)
    resized_height = max(patch_size, ceil((image.height * scale) / patch_size) * patch_size)

    resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    left = max(0, (resized_width - rounded_width) // 2)
    top = max(0, (resized_height - rounded_height) // 2)
    right = left + rounded_width
    bottom = top + rounded_height

    return resized.crop((left, top, right, bottom))


def main():
    #load a random image from /imgs
    image = Image.open("imgs/cats-example.jpg").convert("RGB")
    print("Image size:", image.height, image.width)  # [480, 640]
    processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
    model = AutoModel.from_pretrained(
        "facebook/dinov3-vits16-pretrain-lvd1689m",
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa"
    )
    patch_size = model.config.patch_size
    print("Patch size:", patch_size) # 16
    print("Num register tokens:", model.config.num_register_tokens) # 4

    scale_factor = 2.0
    target_width = int(image.width * scale_factor)
    target_height = int(image.height * scale_factor)
    image = manual_resize(image, target_width, target_height, patch_size=patch_size)
    print("Manually resized image size:", image.height, image.width)

    inputs = processor(images=image, return_tensors="pt", do_resize=False, do_center_crop=False).to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)

    pooled_output = outputs.pooler_output
    print("Pooled output shape:", pooled_output.shape)


    batch_size, _, img_height, img_width = inputs.pixel_values.shape
    num_patches_height, num_patches_width = img_height // patch_size, img_width // patch_size
    num_patches_flat = num_patches_height * num_patches_width

    
    last_hidden_states = outputs.last_hidden_state
    print(last_hidden_states.shape)  # [1, 1 + 4 + 256, 384]
    assert last_hidden_states.shape == (batch_size, 1 + model.config.num_register_tokens + num_patches_flat, model.config.hidden_size)

    patch_features_flat = last_hidden_states[:, 1 + model.config.num_register_tokens:, :].to(torch.float32)
    normalized_flat = torch.nn.functional.normalize(patch_features_flat, dim=-1)
    normalized_flat = normalized_flat.squeeze(0)  # [196, hidden]

    cosine_scores = normalized_flat @ normalized_flat.transpose(0, 1)  # [196, 196]
    cosine_scores = cosine_scores.cpu()
    print("Cosine scores shape:", cosine_scores.shape)
    print("Cosine scores min and max:", cosine_scores.min().item(), cosine_scores.max().item())

    fig, (ax_img, ax_heatmap) = plt.subplots(1, 2, figsize=(12, 6))
    ax_img.imshow(image)
    ax_img.set_title("Input image — click a patch")
    ax_img.axis("off")

    display_patch_width = image.width / num_patches_width
    display_patch_height = image.height / num_patches_height

    # draw patch grid
    for idx in range(num_patches_width + 1):
        ax_img.axvline(idx * display_patch_width, color="white", lw=0.5, alpha=0.3)
    for idx in range(num_patches_height + 1):
        ax_img.axhline(idx * display_patch_height, color="white", lw=0.5, alpha=0.3)

    highlight = Rectangle((0, 0), display_patch_width, display_patch_height, linewidth=1.5, edgecolor="yellow", facecolor="none")
    ax_img.add_patch(highlight)

    heat_data = np.zeros((num_patches_height, num_patches_width), dtype=float)
    heat_plot = ax_heatmap.imshow(heat_data, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax_heatmap.set_title("Cosine similarity heatmap")
    ax_heatmap.axis("off")
    plt.colorbar(heat_plot, ax=ax_heatmap, fraction=0.046, pad=0.04)

    cosine_scores_np = cosine_scores.numpy()

    def update_heatmap(patch_x: int, patch_y: int) -> None:
        if patch_x < 0 or patch_x >= num_patches_width or patch_y < 0 or patch_y >= num_patches_height:
            return
        patch_index = patch_y * num_patches_width + patch_x
        patch_scores = cosine_scores_np[patch_index].reshape(num_patches_height, num_patches_width)
        heat_plot.set_data(patch_scores)
        ax_heatmap.set_title(f"Cosine similarity — patch ({patch_y}, {patch_x})")
        highlight.set_xy((patch_x * display_patch_width, patch_y * display_patch_height))

        if highlight.get_width() != display_patch_width or highlight.get_height() != display_patch_height:
            highlight.set_width(display_patch_width)
            highlight.set_height(display_patch_height)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax_img or event.xdata is None or event.ydata is None:
            return
        patch_x = int(event.xdata // display_patch_width)
        patch_y = int(event.ydata // display_patch_height)
        update_heatmap(patch_x, patch_y)

    default_patch_x = num_patches_width // 2
    default_patch_y = num_patches_height // 2
    update_heatmap(default_patch_x, default_patch_y)

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()