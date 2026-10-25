# Hand Gesture Recognition

基于 OpenCV、MediaPipe Hands 和 PyTorch 的静态手势识别系统，支持数字 0–8 和 OK 共 10 类手势，提供图片、视频和摄像头演示。

## 方法

- **Plain CNN**：五个卷积阶段组成的基线网络。
- **ResNeXt50**：使用 torchvision 的 ResNeXt-50 32×4d，保留完整骨干。
- **ResNeXt50 + DenseBlocks**：在骨干特征图后增加投影层、两个 DenseBlock 和分类头。每个 DenseBlock 包含 4 层，增长率为 32。
- **MediaPipe 裁剪**：通过手部关键点确定裁剪框；检测失败时使用原图回退，并另报告将漏检视为拒识的结果。
- **规则融合**：根据关键点判断手势，在 CNN 置信度低于阈值时使用有效的规则结果。阈值仅在验证集上选择。

CNN 输入为图像。21 个三维关键点用于手部定位与几何规则，不作为特征向量输入 CNN。视频模式逐帧识别静态手形，不包含动作时序分类。

## 环境

使用 Python 3.11。以下命令面向 Windows 和支持 CUDA 的 NVIDIA 显卡：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B check_environment.py --require-cuda
```

依赖使用 PyTorch 2.7.1、torchvision 0.22.1、CUDA 12.8 构建和 MediaPipe 0.10.21。CPU 运行时可单独安装匹配的 CPU 版 PyTorch/torchvision，再安装 `requirements-gpu.txt` 中的通用依赖，并使用 `--device cpu`。

结构与 CUDA 检查：

```powershell
python verify.py
python verify_gpu.py
```

## 数据

仓库包含 data/dataset_split 中的原始划分图片、weights 中的四组最佳模型权重和 assets 中的演示素材。虚拟环境与预处理缓存不包含在仓库中。数据结构如下：

```text
data/dataset_split/
├── train/
│   ├── 0/ ... 8/
│   └── OK/
├── val/
│   ├── 0/ ... 8/
│   └── OK/
└── test/
    ├── 0/ ... 8/
    └── OK/
```

准备数据并缓存手部裁剪：

```powershell
python prepare_data.py --clean-duplicates
```

也可用 `--data <目录>` 指定数据源。清理只调整训练清单，不删除原图片：像素相同且标签冲突的图片全部排除；同标签重复图片保留一份。逐图排除记录在 `prepared/audit.json`。

当前实验原始图片数为 32,355，清理后为 32,348，划分为训练 19,409、验证 6,470、测试 6,469 张。各类数量不均衡。像素去重不能排除近重复或同一采集者跨集合的问题。

## 训练与评估

一键运行种子 42 的四组训练、融合评估和汇总：

```powershell
python run_all.py --seed 42 --epochs 50 --batch-size 32 --device cuda
```

或者逐项执行：

```powershell
python train.py --model plain_cnn --view raw --seed 42 --device cuda
python train.py --model resnext50 --view raw --seed 42 --device cuda
python train.py --model resnext50_dense --view raw --seed 42 --device cuda
python train.py --model resnext50_dense --view mp --seed 42 --device cuda

python evaluate_pipeline.py --checkpoint runs/resnext50_dense_mp_seed42/best.pt --raw-checkpoint runs/resnext50_dense_raw_seed42/best.pt --output reports/pipeline_seed42 --device cuda
python summarize.py
```

默认训练 50 轮，使用交叉熵损失、Adam、初始学习率 0.001 和 batch size 32，从头训练。图像灰度化后复制为三通道并调整至 64×64；训练增强为 30% 概率水平翻转及 ±10° 旋转。按验证集 Macro-F1 保存最佳权重，测试前重新加载。

`run_all.py` 会准备缺失的数据缓存并跳过设置一致的已完成实验。单独运行 `prepare_data.py` 时不会覆盖已有清单。修改实验设置或重新训练已有模型时，使用新的 `--output`，避免覆盖记录。

`run_experiments.ps1` 可运行多个种子；解释器由已激活环境中的 `python` 或 `-Python` 参数指定。

## 实验结果

以下为种子 42、相同数据划分下的单次测试结果，均为百分比：

| 模型 / 方案 | Accuracy | Macro Precision | Macro Recall | Macro F1 | 分类 mAP |
|---|---:|---:|---:|---:|---:|
| Plain CNN，原图 | 98.75 | 96.58 | 94.76 | 95.61 | 97.15 |
| ResNeXt50，原图 | 98.30 | 93.97 | 93.20 | 93.57 | 96.15 |
| ResNeXt50 + DenseBlocks，原图 | 98.36 | 94.28 | 94.29 | 94.13 | 95.48 |
| ResNeXt50 + DenseBlocks，裁剪及原图回退 | 98.41 | 94.65 | 93.89 | 94.24 | 96.64 |
| CNN 与规则融合，含原图回退 | 98.58 | 94.29 | 94.95 | 94.60 | 94.27 |
| 关键点规则 | 71.57 | 87.93 | 68.87 | 72.75 | 61.70 |

本次实验中 Plain CNN 的整体指标最高。裁剪提高了最终网络的分类 mAP；规则融合提高 Accuracy 和 Macro-F1，但降低分类 mAP。单个随机种子不足以判断差异是否稳定，未进行专门的遮挡或光照分组评估。

Plain CNN 与 ResNeXt50 属于不同骨干的对比；ResNeXt50 与增加 DenseBlocks 的版本用于检验新增模块的影响。

分类 mAP 为逐类 one-vs-rest AP 的均值，不使用检测框 IoU 阈值。规则覆盖时使用 one-hot 分数，其余使用 CNN 概率，融合分数不属于校准置信度。

完整数值与每类结果见 [results](results/)。四组最佳权重保存在 [weights](weights/)。本地训练记录保存在 runs/，流程评估保存在 reports/，不纳入 Git。

## 演示

安装依赖后可直接启动，无需重新训练：

```powershell
python app.py
```

默认加载 weights/resnext50_dense_mp.pt 与 weights/fusion.json（验证集选择阈值 0.95）。点击 Camera、Image 或 Video 选择输入，也可用界面的 Checkpoint 按钮切换模型。

使用自行训练的权重与配置时：

```powershell
python app.py --fusion-config reports/pipeline_seed42/fusion_config.json --device cuda
```

界面支持图片、视频和摄像头。推理在工作线程中执行，展示手部框、关键点、识别结果和真实 CNN 置信度。

## 主要文件

| 文件 | 功能 |
|---|---|
| `models.py` | 三种图像分类网络 |
| `common.py` | 图像预处理和数据加载 |
| `hands.py` | 手部定位、几何归一化和规则 |
| `prepare_data.py` | 数据审计与裁剪缓存 |
| `train.py` | 训练、最佳权重选择与测试 |
| `metrics.py` | 分类指标和混淆矩阵 |
| `inference.py` | 模型加载及融合 |
| `evaluate_pipeline.py` | 阈值选择与完整流程评估 |
| `summarize.py` | 实验结果汇总 |
| `run_all.py` | 实验入口 |
| `app.py` | Qt 演示界面 |

## 参考实现

- [torchvision ResNeXt50](https://docs.pytorch.org/vision/0.19/models/generated/torchvision.models.resnext50_32x4d.html)
- [torchvision DenseNet](https://docs.pytorch.org/vision/0.19/_modules/torchvision/models/densenet.html)
- [MediaPipe Hands](https://chuoling.github.io/mediapipe/solutions/hands.html)
