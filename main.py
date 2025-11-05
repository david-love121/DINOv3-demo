import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from functools import partial
from typing import Callable
from matplotlib.backend_bases import MouseEvent
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

matplotlib.use("QtAgg")
from math import ceil
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def manual_resize(
    image: Image.Image, target_width: int, target_height: int, patch_size: int = 16
) -> Image.Image:
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Target dimensions must be positive")

    rounded_width = ceil(target_width / patch_size) * patch_size
    rounded_height = ceil(target_height / patch_size) * patch_size

    scale = max(rounded_width / image.width, rounded_height / image.height)
    resized_width = max(
        patch_size, ceil((image.width * scale) / patch_size) * patch_size
    )
    resized_height = max(
        patch_size, ceil((image.height * scale) / patch_size) * patch_size
    )

    resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    left = max(0, (resized_width - rounded_width) // 2)
    top = max(0, (resized_height - rounded_height) // 2)
    right = left + rounded_width
    bottom = top + rounded_height

    return resized.crop((left, top, right, bottom))


def getCosineScores(
    image: Image.Image, scale_factor: float, model: AutoModel, patch_size
):
    print("Image size:", image.height, image.width)  # [480, 640]
    processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov3-vits16-pretrain-lvd1689m"
    )

    inputs = processor(
        images=image, return_tensors="pt", do_resize=False, do_center_crop=False
    ).to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)

    batch_size, _, img_height, img_width = inputs.pixel_values.shape
    num_patches_height, num_patches_width = (
        img_height // patch_size,
        img_width // patch_size,
    )
    num_patches_flat = num_patches_height * num_patches_width

    last_hidden_states = outputs.last_hidden_state
    print(last_hidden_states.shape)  # [1, 1 + 4 + 256, 384]
    assert last_hidden_states.shape == (
        batch_size,
        1 + model.config.num_register_tokens + num_patches_flat,
        model.config.hidden_size,
    )

    patch_features_flat = last_hidden_states[
        :, 1 + model.config.num_register_tokens :, :
    ].to(torch.float32)
    normalized_flat = torch.nn.functional.normalize(patch_features_flat, dim=-1)
    normalized_flat = normalized_flat.squeeze(0)  # [196, hidden]

    cosine_scores = normalized_flat @ normalized_flat.transpose(0, 1)  # [196, 196]
    cosine_scores = cosine_scores.cpu()
    print("Cosine scores shape:", cosine_scores.shape)
    print(
        "Cosine scores min and max:",
        cosine_scores.min().item(),
        cosine_scores.max().item(),
    )
    return cosine_scores, num_patches_height, num_patches_width


def update_heatmap(
    patch_x: int,
    patch_y: int,
    *,
    num_patches_width: int,
    num_patches_height: int,
    cosine_scores_np: np.ndarray,
    heat_mesh: QuadMesh,
    ax_heatmap: plt.Axes,
    highlight: Rectangle,
    display_patch_width: float,
    display_patch_height: float,
    fig: Figure,
) -> None:
    if (
        patch_x < 0
        or patch_x >= num_patches_width
        or patch_y < 0
        or patch_y >= num_patches_height
    ):
        return

    patch_index = patch_y * num_patches_width + patch_x
    patch_scores = cosine_scores_np[patch_index].reshape(
        num_patches_height, num_patches_width
    )

    heat_mesh.set_array(patch_scores.ravel())
    ax_heatmap.set_title(f"Cosine similarity — patch ({patch_y}, {patch_x})")
    highlight.set_xy((patch_x * display_patch_width, patch_y * display_patch_height))

    if (
        highlight.get_width() != display_patch_width
        or highlight.get_height() != display_patch_height
    ):
        highlight.set_width(display_patch_width)
        highlight.set_height(display_patch_height)

    fig.canvas.draw_idle()


def on_click(
    event: MouseEvent,
    *,
    ax_img: plt.Axes,
    display_patch_width: float,
    display_patch_height: float,
    update_heatmap_func: Callable[[int, int], None],
) -> None:
    if event.inaxes != ax_img or event.xdata is None or event.ydata is None:
        return

    patch_x = int(event.xdata // display_patch_width)
    patch_y = int(event.ydata // display_patch_height)
    update_heatmap_func(patch_x, patch_y)


def main():
    # load a random image from /imgs
    image_path = "imgs/cats-example.jpg"
    image = Image.open(image_path).convert("RGB")
    scale_factor = 4.0
    target_width = int(image.width * scale_factor)
    target_height = int(image.height * scale_factor)
    model = AutoModel.from_pretrained(
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    patch_size = model.config.patch_size
    image = manual_resize(image, target_width, target_height, patch_size=patch_size)

    print("Manually resized image size:", image.height, image.width)

    cosine_scores, num_patches_height, num_patches_width = getCosineScores(
        image, scale_factor, model, patch_size
    )
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

    highlight = Rectangle(
        (0, 0),
        display_patch_width,
        display_patch_height,
        linewidth=1.5,
        edgecolor="yellow",
        facecolor="none",
    )
    ax_img.add_patch(highlight)

    heat_data = np.zeros((num_patches_height, num_patches_width), dtype=float)
    sns_heatmap = sns.heatmap(
        heat_data,
        ax=ax_heatmap,
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        cbar=True,
        square=True,
        xticklabels=False,
        yticklabels=False,
    )
    heat_mesh = sns_heatmap.collections[0]
    ax_heatmap.set_title("Cosine similarity heatmap")
    ax_heatmap.set_aspect("equal")
    sns_heatmap.set_axis_off()
    ax_heatmap.set_xticks([])
    ax_heatmap.set_yticks([])
    ax_heatmap.set_xlabel("")
    ax_heatmap.set_ylabel("")

    cosine_scores_np = cosine_scores.numpy()

    bound_update_heatmap = partial(
        update_heatmap,
        num_patches_width=num_patches_width,
        num_patches_height=num_patches_height,
        cosine_scores_np=cosine_scores_np,
        heat_mesh=heat_mesh,
        ax_heatmap=ax_heatmap,
        highlight=highlight,
        display_patch_width=display_patch_width,
        display_patch_height=display_patch_height,
        fig=fig,
    )

    bound_on_click = partial(
        on_click,
        ax_img=ax_img,
        display_patch_width=display_patch_width,
        display_patch_height=display_patch_height,
        update_heatmap_func=bound_update_heatmap,
    )

    default_patch_x = num_patches_width // 2
    default_patch_y = num_patches_height // 2
    bound_update_heatmap(default_patch_x, default_patch_y)

    fig.canvas.mpl_connect("button_press_event", bound_on_click)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
