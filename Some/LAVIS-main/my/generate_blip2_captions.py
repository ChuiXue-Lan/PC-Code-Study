import os
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

# 1. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载BLIP-2-opt6.7B模型
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt",
    model_type="pretrain_opt6.7b",
    is_eval=True,
    device=device
)

# 3. 图片文件夹路径
image_folder = "your_image_folder"  # 替换为你的图片文件夹路径
output_file = "captions_result.txt"

# 4. 生成描述的prompt模板
prompts = [
    "请为这张图片生成一条详细描述，突出物体结构。",
    "请为这张图片生成一条详细描述，突出材质。",
    "请为这张图片生成一条详细描述，突出动作。",
    "请为这张图片生成一条详细描述，突出场景。",
    "请为这张图片生成一条详细描述，突出颜色。",
    "请为这张图片生成一条详细描述，突出光影。",
    "请为这张图片生成一条详细描述，突出空间布局。",
    "请为这张图片生成一条详细描述，突出物体关系。",
    "请为这张图片生成一条详细描述，突出细节。",
    "请为这张图片生成一条详细描述，突出整体氛围。"
]

# 5. 遍历图片并生成描述
with open(output_file, "w", encoding="utf-8") as fout:
    for img_name in os.listdir(image_folder):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue
        img_path = os.path.join(image_folder, img_name)
        raw_image = Image.open(img_path).convert("RGB")
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        fout.write(f"图片: {img_name}\n")
        for idx, prompt in enumerate(prompts):
            result = model.generate({"image": image, "prompt": prompt}, use_nucleus_sampling=True)
            fout.write(f"描述{idx+1}: {result[0]}\n")
        fout.write("\n")
        print(f"{img_name} 处理完成")

print(f"全部完成，结果已保存至 {output_file}")