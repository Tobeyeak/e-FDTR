# e-FDTR

基于 [DEIM](https://github.com/ShihuaHuang95/DEIM) / [D-FINE](https://github.com/Peterande/D-FINE) 的目标检测训练框架，面向无人机/空中目标检测场景。

## 改进点

- **FAD_DySample 编码器**：FAD 特征聚合 + DySample 动态上采样，3 层输入（`return_idx: [1,2,3]`，`feat_strides: [8,16,32]`）
- **ACL 平滑渐进训练**（Adaptive Curriculum Learning）：损失权重与匹配器在 epoch 80→100 之间平滑过渡，从"分类+粗定位"逐步切换到"精确定位"，训练更稳定

---

## 目录结构

```
e-FDTR/
├── train.py              # 训练/验证入口
├── export.py             # ONNX / TensorRT 导出
├── engine/               # 核心代码（模型、数据、训练器）
├── configs/
│   ├── dfine/            #   e-FDTR-n.yml（最终配置）等
│   ├── base/             #   模型/优化器/数据加载基础配置
│   └── dataset/          #   数据集定义（custom_detection.yml）
├── tools/inference/      # 推理脚本（torch / onnx / tensorrt）
├── A_pipeline/           # 训练流程脚本（一键执行）
│   ├── train.sh          #   训练
│   ├── val.sh            #   验证
│   ├── export.sh         #   导出 ONNX
│   └── predict.sh        #   推理
├── ckpt/                 # 预训练权重（deim_dfine_hgnetv2_n_coco_160e.pth）
├── weight/hgnetv2/       # HGNetv2 backbone 预训练权重
└── requirements.txt
```

---

## 1. 环境配置

```bash
# 1. 创建环境（Python 3.10）
conda create -n e-fdtr python=3.10 -y
conda activate e-fdtr

# 2. 安装 PyTorch（按你的 CUDA 版本选择，示例为 cu118）
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118

# 3. 安装其余依赖
pip install -r requirements.txt
```

验证环境：

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"
```

> ⚠️ `A_pipeline/` 脚本里写死了 `PYTHON` / `TORCHRUN` 的绝对路径（`/opt/miniforge3/envs/DETR/bin/...`）。如果你用的 conda 环境名不同，需改成你的环境路径。

---

## 2. 数据准备

数据集需为 **COCO 格式**，并修改 `configs/dataset/custom_detection.yml` 中的路径：

```yaml
num_classes: 4            # 你的类别数
train_dataloader:
  dataset:
    img_folder: /path/to/train/images
    ann_file:    /path/to/train/_annotations.coco.json
val_dataloader:
  dataset:
    img_folder: /path/to/val/images
    ann_file:    /path/to/val/_annotations.coco.json
```

目录结构参考：

```
dataset/
├── train/
│   ├── images/          # 训练图
│   └── annotations/     # COCO 标注 json
└── val/
    ├── images/
    └── annotations/
```

---

## 3. 训练

一键训练（`bash A_pipeline/train.sh`）：

```bash
bash A_pipeline/train.sh
```

脚本内部等价于：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 train.py \
    -c configs/dfine/e-FDTR-n.yml \
    --seed=0 \
    --output-dir /home/tione/notebook/tjn/DETR/Z-test/deim \
    --summary-dir /home/tione/notebook/tjn/DETR/Z-test/deim/summary \
    -t ckpt/deim_dfine_hgnetv2_n_coco_160e.pth \
    2>&1 | tee /home/tione/notebook/tjn/DETR/Z-test/deim/train.log
```

### `train.sh` 可调参数

打开 `A_pipeline/train.sh`，修改顶部变量：

| 变量 | 含义 | 怎么调 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | 用哪几张卡 | `0,1,2,3` 用 4 卡；`0` 只用第 0 卡 |
| `NUM_GPUS` | 进程数 | **必须等于** `CUDA_VISIBLE_DEVICES` 里的卡数 |
| `CONFIG_FILE` | 配置文件 | 换模型规模/对照实验时改这里 |
| `PRETRAIN_WEIGHTS` | 预训练权重 | 想从零训练就删掉 `-t $PRETRAIN_WEIGHTS` 这行 |
| `OUTPUT_DIR` | 模型保存目录 | 生成 `last.pth` / `best.pth` 和 `train.log` |
| `SEED` | 随机种子 | 复现实验 |
| `MASTER_PORT` | 分布式端口 | 端口被占用时改 |

### 常见训练调整

**1. 单卡训练**（显存不够 / 只有 1 卡）

改 `train.sh`：
```bash
export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
```
同时减小 batch（`total_batch_size: 64` 是 4 卡设计，每卡 16）。改配置文件 `configs/dfine/e-FDTR-n.yml`：
```yaml
train_dataloader:
  total_batch_size: 8    # 单卡 8
```

**2. 改训练轮数**

配置文件里：
```yaml
epoches: 160
```

**3. 改输入分辨率**

分辨率在 `configs/base/dataloader_dfine.yml` 的 `Resize`：
```yaml
- {type: Resize, size: [1088, 1088], }   # 改成 [640, 640] 等（需为 32 的倍数）
```
同时改 `configs/base/dfine_hgnetv2.yml` 的评估尺寸：
```yaml
eval_spatial_size: [640, 640]
```

**4. 改学习率**

`configs/dfine/e-FDTR-n.yml`（或 base/optimizer.yml）：
```yaml
optimizer:
  lr: 0.0008       # 总学习率
  params:          # 分组学习率
    - params: '^(?=.*backbone)...'  lr: 0.0004   # backbone 组
```

**5. 调整 ACL 平滑策略**（核心改进）

`configs/dfine/e-FDTR-n.yml` 的 `DEIMCriterion` 段：
```yaml
DEIMCriterion:
  transition_mode: smooth     # smooth 平滑过渡 / hard 硬切换
  transition_start: 80        # 过渡开始 epoch
  transition_end: 100         # 过渡结束 epoch
  weight_dict_stage1: {...}   # 阶段1（粗定位）损失权重
  weight_dict_stage2: {...}   # 阶段2（精确定位）损失权重
  matcher_stage1: {...}       # 阶段1 匹配器
  matcher_stage2: {...}       # 阶段2 匹配器
```

---

## 4. 验证（测试）

```bash
bash A_pipeline/val.sh
```

等价于：
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train.py \
    -c configs/dfine/e-FDTR-n.yml \
    --test-only \
    --yolo-metrice \
    --output-dir /home/tione/notebook/tjn/DETR/Z-test/deim3-val \
    -r /home/tione/notebook/tjn/DETR/Z-test/deim3/best_stg1.pth
```

### `val.sh` 可调参数

| 变量 | 含义 |
|---|---|
| `CHECKPOINT` | 要评估的权重（`best.pth` / `best_stg1.pth` / `last.pth`） |
| `OUTPUT_DIR` | 评估结果目录（保存 COCO 预测 json、`yolo_metrics.txt`） |
| `GPU_IDS` / `NUM_GPUS` | 评估用卡（通常单卡即可） |

结果：AP / AP50 / AP75 等 COCO 指标打印在终端。

---

## 5. 导出

### 5.1 导出 ONNX

```bash
bash A_pipeline/export.sh
```

等价于：
```bash
python export.py -c configs/dfine/e-FDTR-n.yml -r /path/to/best.pth --format onnx
```

### `export.sh` 可调参数

| 变量 / 参数 | 含义 |
|---|---|
| `CHECKPOINT` | 要导出的权重 |
| `--format` | `onnx`（默认）/ `engine`（TensorRT） |
| `--img-size` | 导出分辨率，如 `--img-size 640 640`（默认 640） |
| `--batch` | batch size（默认 1） |
| `--half` | engine 时用 FP16 |
| `--dynamic` | ONNX 动态 batch |
| `--simplify` | 用 onnxslim 精简 ONNX（需 `pip install onnxslim`） |
| `--opset` | ONNX opset 版本（默认 17） |

输出文件与 checkpoint 同名（`.onnx` / `.engine`）。

### 5.2 导出 TensorRT Engine

改 `export.sh` 里的导出命令：
```bash
$PYTHON $EXPORT_PY \
    -c $CONFIG_FILE \
    -r $CHECKPOINT \
    --format engine \
    --half \
    --img-size 640 640
```
（需先安装 TensorRT）

---

## 6. 推理

```bash
bash A_pipeline/predict.sh
```

等价于：
```bash
python tools/inference/detect/torch_inf.py \
    -c configs/dfine/e-FDTR-n.yml \
    -r /path/to/best.pth \
    --input /path/to/image_or_dir_or_video \
    --device cuda:0 \
    -o /path/to/output_dir
```

### `predict.sh` 可调参数

| 变量 | 含义 |
|---|---|
| `CHECKPOINT` | 推理用权重 |
| `INPUT_IMAGE` | 输入：单张图片 / 目录 / 视频（`.jpg` `.png` `.bmp` `.mp4` 等） |
| `OUTPUT_DIR` | 可视化结果保存目录 |
| `DEVICE` | `cuda:0` 或 `cpu` |

> 推理置信度阈值在 `tools/inference/detect/torch_inf.py` 的 `-t/--thrh`（默认 0.2），需要改阈值时加 `-t 0.5` 到脚本的推理命令里。

### ONNX / TensorRT 推理

```bash
python tools/inference/detect/onnx_inf.py --onnx model.onnx --input image.jpg
python tools/inference/detect/trt_inf.py  --trt  model.engine --input image.jpg
```

---

## 7. 关键配置速查表

以 `configs/dfine/e-FDTR-n.yml`（最终版）为例：

| 配置项 | 说明 | 位置 |
|---|---|---|
| `DEIM.backbone` | `HGNetv2` | dfine/e-FDTR-n.yml |
| `DEIM.encoder` | `FAD_DySample`（FAD + DySample） | dfine/e-FDTR-n.yml |
| `epoches` | 训练轮数 | dfine/e-FDTR-n.yml |
| `total_batch_size` | 总 batch（÷卡数 = 每卡 batch） | dfine/e-FDTR-n.yml |
| `optimizer.lr` | 学习率 | base/optimizer.yml |
| `Resize size` | 输入分辨率 | base/dataloader_dfine.yml |
| `eval_spatial_size` | 评估尺寸 | base/dfine_hgnetv2.yml |
| `num_classes` | 类别数 | dataset/custom_detection.yml |
| ACL 相关 | `transition_mode` / `start` / `end` / `weight_dict_stage1/2` | dfine/e-FDTR-n.yml |

其他基础配置（不同规模）见 `configs/dfine/dfine_hgnetv2_{n,s,m,l,x}_custom.yml`（基础 HybridEncoder 版），`e-FDTR-n.yml` 是 FAD_DySample 改进版。

---

## 8. 引用

本项目基于以下工作：

```bibtex
@misc{huang2024deim,
      title={DEIM: DETR with Improved Matching for Fast Convergence},
      author={Shihua Huang, Zhichao Lu, Xiaodong Cun, Yongjun Yu, Xiao Zhou, and Xi Shen},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2025},
}
```

[D-FINE](https://github.com/Peterande/D-FINE) · [RT-DETR](https://github.com/lyuwenyu/RT-DETR)
