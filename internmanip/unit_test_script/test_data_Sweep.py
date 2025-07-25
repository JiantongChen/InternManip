import os
import matplotlib.pyplot as plt

from internmanip.dataset.base import LeRobotSingleDataset
from internmanip.dataset.schema import EmbodimentTag
from internmanip.configs.dataset.data_config import DATA_CONFIG_MAP

# 数据路径
REPO_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_PATH, ".internmanip/demo_data/sweep.PickNPlace")
# DATA_PATH = os.path.join(REPO_PATH, "demo_data/robot_sim_converted.PickNPlace")
# DATA_PATH = "/PATH/TO/YOUR/data/Sweep"
print("Loading dataset... from", DATA_PATH)

# 从配置文件获取modality配置
dataset_name = "sweep_joint"

data_config = DATA_CONFIG_MAP[dataset_name]
modality_config = data_config.modality_config()
modality_transform = data_config.transform()
embodiment_tag = EmbodimentTag.GR1

# 创建数据集
dataset = LeRobotSingleDataset(
    dataset_path=DATA_PATH,
    modality_configs=modality_config,
    video_backend="decord",
    video_backend_kwargs=None,
    transforms=None,
    embodiment_tag=embodiment_tag,
)


print("\n"*2)
print("="*100)
print(f"{' Humanoid Dataset ':=^100}")
print("="*100)

# 打印第7个样本
resp = dataset[0]
print("第7个样本类型:", type(resp))
print("包含字段:", resp.keys())
for k, v in resp.items():
    if hasattr(v, 'shape'):
        print(f"{k}: shape={v.shape}")
    else:
        print(f"{k}: type={type(v)}")

# 可视化部分图片
images_list = []
for i in range(100):
    if i % 10 == 0:
        resp = dataset[i]
        img = resp["video.base_view"][0]
        images_list.append(img)

fig, axs = plt.subplots(2, 5, figsize=(20, 10))
for i, ax in enumerate(axs.flat):
    ax.imshow(images_list[i])
    ax.axis("off")
    ax.set_title(f"Image {i}")
plt.tight_layout()
plt.savefig("test_data_Sweep.png")



# 创建带transform的数据集
ddataset = LeRobotSingleDataset(
    dataset_path=DATA_PATH,
    modality_configs=modality_config,
    video_backend="decord",
    video_backend_kwargs=None,
    transforms=modality_transform,
    embodiment_tag=embodiment_tag,
)

# 再次打印第7个样本
resp = dataset[7]
print("应用transform后的第7个样本:")
print("包含字段:", resp.keys())
for k, v in resp.items():
    if hasattr(v, 'shape'):
        print(f"{k}: shape={v.shape}")
    else:
        print(f"{k}: type={type(v)}")