import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import ConnectionPatch
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm

# ==============================
# 创建：创建结果保存路径
# ==============================
data_dir_test="./test_images_dense_sparse_matching"
data_dir = "./results"
# 如果目录不存在，则创建
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"📁 已创建目录: {data_dir}")
else:
    print(f"✅ 目录已存在: {data_dir}")
    
    
# ==============================
# 配置：选择你要加载的模型路径
# ==============================

MODEL_PATH = "./allmodels/facebook/dinov3-vitl16-pretrain-lvd1689m"

# ==============================
# 检查模型路径是否存在
# ==============================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"模型路径不存在: {os.path.abspath(MODEL_PATH)}\n"
                            "请检查路径是否正确，或重新下载。")

print(f"✅ 正在从本地加载模型: {os.path.abspath(MODEL_PATH)}")

# ==============================
# 自动选择设备：优先使用 GPU (CUDA)，否则使用 CPU
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {device}")

# ==============================
# 使用 transformers 加载模型
# ==============================
from transformers import AutoModel

# 根据设备决定 dtype
if device.type == "cuda":
    torch_dtype = torch.float16
    use_autocast = True
else:
    torch_dtype = torch.float32
    use_autocast = False

# 加载模型
model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch_dtype,
    output_hidden_states=True
)

model.to(device)
model.eval()
print(f"✅ 模型 {os.path.basename(MODEL_PATH)} 已成功加载！")
print(f"🚀 模型运行设备: {next(model.parameters()).device}")

# ==============================
# 加载数据
# ==============================

image_left_path = os.path.join(data_dir_test, "image_left.jpg")
mask_left_path = os.path.join(data_dir_test, "image_left_fg.png")
image_right_path = os.path.join(data_dir_test, "image_right.jpg")
mask_right_path = os.path.join(data_dir_test, "image_right_fg.png")

def load_image_from_path(path: str) -> Image.Image:
    return Image.open(path).convert("RGBA")

image_left = load_image_from_path(image_left_path)
mask_left = load_image_from_path(mask_left_path)
image_right = load_image_from_path(image_right_path)
mask_right = load_image_from_path(mask_right_path)

# 创建显示图
plt.figure(figsize=(16, 8), dpi=300)

for j, (image, mask) in enumerate([(image_left, mask_left), (image_right, mask_right)]):
    foreground = Image.composite(image, Image.new('RGBA', image.size, (0,0,0,0)), mask)
    mask_np = np.array(mask)
    mask_np[:, :, 3] = 255 - mask_np[:, :, 3]
    mask_bg = Image.fromarray(mask_np)
    background = Image.composite(image, Image.new('RGBA', image.size, (0,0,0,0)), mask_bg)

    data_to_show = [image, mask, foreground, background]
    data_labels = ["Image", "Mask", "Foreground", "Background"]

    for i in range(len(data_to_show)):
        plt.subplot(2, len(data_to_show), 4 * j + i + 1)
        if data_to_show[i].mode == 'RGBA' or data_to_show[i].mode == 'LA':
            plt.imshow(data_to_show[i])
        else:
            plt.imshow(data_to_show[i].convert('RGB'))
        plt.axis('off')
        plt.title(data_labels[i], fontsize=12)

output_path = data_dir+"/result_dense_sparse_matching.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"结果已保存至: {output_path}")

# ==============================
# 数据转换
# ==============================
PATCH_SIZE = 16
IMAGE_SIZE = 768

patch_quant_filter = torch.nn.Conv2d(1, 1, PATCH_SIZE, stride=PATCH_SIZE, bias=False)
patch_quant_filter.weight.data.fill_(1.0 / (PATCH_SIZE * PATCH_SIZE))
patch_quant_filter.to(device)

def resize_transform(
    mask_image: Image,
    image_size: int = IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    w, h = mask_image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))

# ==============================
# 特征提取
# ==============================

MODEL_DINOV3_VITL = "dinov3_vitl16"
MODEL_NAME = MODEL_DINOV3_VITL

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MODEL_TO_NUM_LAYERS = {
    MODEL_DINOV3_VITL: 24,
}

n_layers = MODEL_TO_NUM_LAYERS[MODEL_NAME]

# ==============================
# 特征提取
# ==============================
patch_mask_values = []
patch_features = []
h_patches_list = []  # 保存每个图像的 h_patches
w_patches_list = []  # 保存每个图像的 w_patches

with torch.inference_mode():
    for image, mask in tqdm([(image_left, mask_left), (image_right, mask_right)], desc="Processing images"):
        # 处理 mask
        mask = mask.split()[-1]
        mask_resized = resize_transform(mask)
        mask_resized = mask_resized.to(device)
        mask_quantized = patch_quant_filter(mask_resized).squeeze().detach().cpu()
        patch_mask_values.append(mask_quantized)

        # 处理图像
        image = image.convert('RGB')
        image_resized = resize_transform(image)
        image_resized = TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        image_resized = image_resized.unsqueeze(0).to(device)

        # 计算 h_patches 和 w_patches
        h_patches = int(image_resized.shape[2] // PATCH_SIZE)
        w_patches = int(image_resized.shape[3] // PATCH_SIZE)
        h_patches_list.append(h_patches)
        w_patches_list.append(w_patches)
        num_patches = h_patches * w_patches

        # 使用 autocast（仅 GPU 支持）
        if use_autocast:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(image_resized)
                feats = outputs.hidden_states
        else:
            outputs = model(image_resized)
            feats = outputs.hidden_states

        # 模拟 get_intermediate_layers
        feats = [f.squeeze() for f in feats]
        dim = feats[-1].shape[-1]  # 特征维度（通常为 1024）

        # 跳过 CLS 和 register tokens
        num_extra_tokens = 5  # 1 CLS + 4 register tokens
        feat_last = feats[-1][num_extra_tokens:]

        # 验证 token 数量
        if feat_last.shape[0] != num_patches:
            raise ValueError(f"Token 数量 ({feat_last.shape[0]}) 与预期 patch 数量 ({num_patches}) 不匹配！")

        # Reshape to [num_patches, dim]
        feat_last = feat_last.view(num_patches, dim)
        feat_last = F.normalize(feat_last, dim=-1, p=2)
        feat_last = feat_last.detach().cpu()
        patch_features.append(feat_last)

print("✅ 特征提取完成！")

# 提取左图和右图的 h_patches 和 w_patches
h_patches, h_patches_right = h_patches_list
w_patches, w_patches_right = w_patches_list

# ==============================
# 特征匹配
# ==============================
MASK_FG_THRESHOLD = 0.5

# 再次归一化（与原始代码一致）
patch_features[0] = F.normalize(patch_features[0], p=2, dim=-1)
patch_features[1] = F.normalize(patch_features[1], p=2, dim=-1)

# 计算特征相似度热图
num_patches_right = patch_features[1].shape[0]
if num_patches_right != h_patches_right * w_patches_right:
    w_patches_right = num_patches_right // h_patches_right
    print(f"修正 w_patches_right 为: {w_patches_right}")

heatmaps = torch.einsum(
    "kf,lf->kl",
    patch_features[0],  # [num_patches_left, dim]
    patch_features[1],  # [num_patches_right, dim]
).view(-1, h_patches_right, w_patches_right)  # [num_patches_left, h_patches_right, w_patches_right]

# 计算左图中的 2D patch 位置
n_patches_left = patch_features[0].shape[0]
patch_indices_left = torch.arange(n_patches_left)
locs_2d_left = (
    torch.stack(
        (
            patch_indices_left // w_patches,  # row
            patch_indices_left % w_patches,   # column
        ),
        dim=-1
    ) + 0.5
) * PATCH_SIZE

# 计算右图中对应的 2D patch 位置
patch_indices_right = torch.flatten(heatmaps, start_dim=-2).argmax(dim=-1)
locs_2d_right = (
    torch.stack(
        (
            patch_indices_right // w_patches_right,  # row
            patch_indices_right % w_patches_right,   # column
        ),
        dim=-1
    ) + 0.5
) * PATCH_SIZE

# 前景 patch 选择（左图）
patches_left_fg_selection = (patch_mask_values[0].view(-1) > MASK_FG_THRESHOLD)
# 左图 patch 映射到右图前景 patch 的掩码
patches_right_fg_selection = (patch_mask_values[1].view(-1)[patch_indices_right] > MASK_FG_THRESHOLD)
# 选择左右图均是前景的 patch
patches_fg_selection = patches_left_fg_selection & patches_right_fg_selection

# 提取匹配的前景 patch 坐标
locs_2d_left_fg = locs_2d_left[patches_fg_selection, :]
locs_2d_right_fg = locs_2d_right[patches_fg_selection, :]

print("✅ 特征匹配完成！")
print(f"匹配的前景 patch 数量: {locs_2d_left_fg.shape[0]}")


# ==============================
# PCA 可视化
# ==============================
pca = PCA(n_components=3, whiten=True)
fg_patches_left = patch_features[0][patches_fg_selection]  # 选择前景 patch
pca.fit(fg_patches_left.numpy())  # 确保输入 PCA 的是 numpy 数组

# 获取左图的颜色
num_patches_left = patch_features[0].shape[0]  # [num_patches, dim]
if num_patches_left != h_patches * w_patches:
    w_patches = num_patches_left // h_patches  # 重新计算 w_patches
    print(f"修正 w_patches 为: {w_patches}")

x_left = patch_features[0]  # [num_patches, dim]
projected_image_left = torch.from_numpy(
    pca.transform(x_left.numpy())
).view(h_patches, w_patches, 3)  # reshape 为 [h_patches, w_patches, 3]
projected_image_left = torch.nn.functional.sigmoid(projected_image_left * 2.0).permute(2, 0, 1)

# 获取右图的颜色
num_patches_right = patch_features[1].shape[0]
if num_patches_right != h_patches_right * w_patches_right:
    w_patches_right = num_patches_right // h_patches_right  # 重新计算 w_patches_right
    print(f"修正 w_patches_right 为: {w_patches_right}")

x_right = patch_features[1]  # [num_patches, dim]
projected_image_right = torch.from_numpy(
    pca.transform(x_right.numpy())
).view(h_patches_right, w_patches_right, 3)  # reshape 为 [h_patches_right, w_patches_right, 3]
projected_image_right = torch.nn.functional.sigmoid(projected_image_right * 2.0).permute(2, 0, 1)

# 应用前景掩码
projected_image_left *= (patch_mask_values[0] > MASK_FG_THRESHOLD)[None, :, :]
projected_image_right *= (patch_mask_values[1] > MASK_FG_THRESHOLD)[None, :, :]

# 可视化
plt.figure(figsize=(4, 2), dpi=300)
plt.subplot(1, 2, 1)
plt.imshow(projected_image_left.permute(1, 2, 0))
plt.title("Left Image, Dense Correspondences", fontsize=5)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(projected_image_right.permute(1, 2, 0))
plt.title("Right Image, Dense Correspondences", fontsize=5)
plt.axis('off')
plt.tight_layout()
plt.savefig(data_dir+"/pca_visualization.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ PCA 可视化完成！")

# ==============================
# 分层点选择和可视化
# ==============================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

# 图像缩放比例，从 patch 坐标转换到原始图像坐标
scale_left = image_left.height / IMAGE_SIZE
scale_right = image_right.height / IMAGE_SIZE

STRATIFY_DISTANCE_THRESHOLD = 100.0

def compute_distances_l2(X, Y, X_squared_norm, Y_squared_norm):
    distances = -2 * X @ Y.T
    distances.add_(X_squared_norm[:, None]).add_(Y_squared_norm[None, :])
    return distances

def stratify_points(pts_2d: torch.Tensor, threshold: float = 100.0) -> torch.Tensor:
    # pts_2d: [N, 2]
    n = len(pts_2d)
    max_value = threshold + 1
    pts_2d_sq_norms = torch.linalg.vector_norm(pts_2d, dim=1)
    pts_2d_sq_norms.square_()
    distances = compute_distances_l2(pts_2d, pts_2d, pts_2d_sq_norms, pts_2d_sq_norms)
    distances.fill_diagonal_(max_value)
    distances_mask = torch.empty((n, n), dtype=pts_2d.dtype, device=pts_2d.device)
    torch.le(distances, threshold, out=distances_mask)
    ones_vec = torch.ones(n, device=pts_2d.device, dtype=pts_2d.dtype)
    counts_vec = torch.mv(distances_mask, ones_vec)
    indices_mask = np.ones(n)
    while torch.any(counts_vec).item():
        index_max = torch.argmax(counts_vec).item()
        indices_mask[index_max] = 0
        distances[index_max, :] = max_value
        distances[:, index_max] = max_value
        torch.le(distances, threshold, out=distances_mask)
        torch.mv(distances_mask, ones_vec, out=counts_vec)
    indices_to_exclude = np.nonzero(indices_mask == 0)[0]
    indices_to_keep = np.nonzero(indices_mask > 0)[0]
    return indices_to_exclude, indices_to_keep

print(f"Non-stratified points: {tuple(locs_2d_left_fg.shape)}")

indices_to_exclude, indices_to_keep = stratify_points(locs_2d_left_fg * scale_left, STRATIFY_DISTANCE_THRESHOLD**2)

sparse_points_left_yx = locs_2d_left_fg[indices_to_keep, :].cpu().numpy()
sparse_points_right_yx = locs_2d_right_fg[indices_to_keep, :].cpu().numpy()

print(f"Stratified points: {sparse_points_left_yx.shape}")

# 显示原始左右图像并绘制连接线
fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(121)
ax1.imshow(image_left)
ax1.set_axis_off()
ax2 = fig.add_subplot(122)
ax2.imshow(image_right)
ax2.set_axis_off()

for i, (row_left, col_left), (row_right, col_right) in zip(
    indices_to_keep, sparse_points_left_yx, sparse_points_right_yx
):
    row_left_orig, col_left_orig = locs_2d_left_fg[i]
    # 使用 PCA 可视化的颜色
    color = projected_image_left[
        :,
        int(row_left_orig / PATCH_SIZE),
        int(col_left_orig / PATCH_SIZE)
    ].cpu().numpy()
    con = ConnectionPatch(
        xyA=(col_left * scale_left, row_left * scale_left),
        xyB=(col_right * scale_right, row_right * scale_right),
        coordsA="data",
        coordsB="data",
        axesA=ax1,
        axesB=ax2,
        color=color,
    )
    ax2.add_artist(con)

# 保存可视化结果
output_path = data_dir+"/sparse_matching_visualization.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ 稀疏匹配可视化结果已保存至: {output_path}")
