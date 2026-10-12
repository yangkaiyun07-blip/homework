# 官方 RUBIK 图像匹配实验

当前版本直接依赖 rubik.json：配对、33个难度组、K1/K2内参及rel_pose真值均来自官方清单，不再从test_meta自行生成配对。

主入口 experiment.py，流程为 prepare → fetch → run → analyze。四种方法：sift_nn、superpoint_nn、superpoint_superglue、eloftr（EfficientLoFTR，不是原版LoFTR）。

## 放置与环境

将本文件夹上传到 /data/yangkaiyun/homework/image_matching_project。

```bash
cd /data/yangkaiyun/homework/image_matching_project
conda activate eloftr
```

复用已调好的环境即可，核心依赖是 NumPy、OpenCV、Matplotlib。需要新环境时：

```bash
conda env create -f environment.yml
conda activate matching_project
```

environment.yml 只安装核心依赖，足以跑SIFT/评估/画图。深度模型另需适合GPU的PyTorch 2.x；ELoFTR额外依赖列在requirements-eloftr.txt。已有可工作的环境不要盲目重装，也不要混装多个OpenCV包。安装核心依赖也可用 python -m pip install -r requirements.txt。

## 1. 检查图片并固定官方配对

```bash
python experiment.py prepare \
  --data-root /data/yangkaiyun/datasets/nuscenes \
  --rubik /data/yangkaiyun/homework/RUBIK/rubik.json \
  --per-group 0 \
  --out pairs
```

--per-group 0 使用全部16500对；例如 --per-group 10 从每个组按固定排序均匀选10对，共330对。没有随机抽样，不根据方法表现筛选。

图片按官方文件名在 samples/CAM_*/ 和 sweeps/CAM_*/ 查找。存在v1.0-test/sample_data.json时，辅助使用其中的路径与尺寸；没有元数据时使用原始nuScenes的1600×900尺寸。不要使用已经裁剪或缩放的原始文件。

prepare输出：pairs/rubik_pairs.json、data_check.json、missing_images.txt。只检查所选配对涉及的图片存在性，不因未选中图片缺失而阻止小子集测试。选中图片缺失时明确报错，不静默删除。

运行时才检查解码及尺寸；此时失败的输入仍计入所选配对分母。匹配阶段不会重新选择图像。

prepare是方便检查与抽取子集的步骤，不是必须。也可在run时直接 --pairs /data/yangkaiyun/homework/RUBIK/rubik.json 使用完整原始清单。

## 2. SIFT+NN

```bash
python experiment.py run \
  --methods sift_nn \
  --data-root /data/yangkaiyun/datasets/nuscenes \
  --pairs pairs/rubik_pairs.json \
  --nfeatures 4096 --ratio 0.75 \
  --out results
```

SIFT匹配函数与之前版本逐字保留：SIFT+L2最近邻+0.75比值筛选，默认关闭双向筛选。--mutual启用双向最近邻一致性，--ratio 0使用纯NN。nfeatures是OpenCV请求的上限，相同响应的并列点可能超出。

外围流程统一为800×448图像，内参用实际宽高缩放比例同步调整；几何估计为USAC_MAGSAC，默认1像素阈值、最多10000次迭代。虽然SIFT方法未变，预处理和几何评估与旧版不同，不能直接混用旧成绩。

## 3. SuperPoint / SuperGlue

```bash
python experiment.py fetch
```

下载官方模型代码、superpoint_v1.pth、superglue_outdoor.pth及许可证到models/SuperGluePretrainedNetwork。无需训练；不覆盖现有模型文件。记录下载来源与SHA-256，但并非与预置可信哈希比对。下载失败需完成下载再运行。

来源：https://github.com/magicleap/SuperGluePretrainedNetwork

```bash
python experiment.py run \
  --methods superpoint_nn superpoint_superglue \
  --data-root /data/yangkaiyun/datasets/nuscenes \
  --pairs pairs/rubik_pairs.json \
  --device cuda --nfeatures 1024 \
  --out results
```

两者复用同一SuperPoint，检测阈值默认0.005。NN使用0.75比值筛选；SuperGlue使用outdoor权重、20次Sinkhorn和0.2匹配阈值。ratio/mutual不作用于SuperGlue。示例选1024点以控制显存；若需统一稀疏方法特征上限，SIFT首次运行也选1024，报告必须注明各方法设置。

## 4. ELoFTR

```bash
python experiment.py run \
  --methods eloftr \
  --data-root /data/yangkaiyun/datasets/nuscenes \
  --pairs pairs/rubik_pairs.json \
  --rubik-root /data/yangkaiyun/homework/RUBIK \
  --checkpoint /data/yangkaiyun/homework/RUBIK/detector_free/EfficientLoFTR/weights/eloftr_outdoor.ckpt \
  --device cuda \
  --out results
```

checkpoint替换成真实权重路径。复用rubik-root/detector_free/EfficientLoFTR源码，载入full模型并reparameter，不需要训练。

--methods可以一次选择多个方法。先单独验证各个模型更方便定位环境错误。相同结果目录可补跑其他方法；已有输出的方法拒绝覆盖。修改公共图像对或评估设置需新结果目录。

## 5. 汇总和画图

```bash
python experiment.py analyze --out results
```

自动发现已完成的方法，不要求四种方法全齐。也可加 --methods sift_nn superpoint_nn。无需再运行其他eval.py：run执行逐对评估，analyze汇总。

输出：
- 每个方法目录：results.csv/json、summary.json、config.json、匹配NPZ和前20个可读取图像对的JPG。
- config记录参数、库版本、设备和模型权重SHA-256。
- results/summary.csv/json：总体指标。
- all_pairs.csv：全部方法的逐对记录。
- grouped_results.csv：每个官方难度组的结果。
- difficulty_results.csv：分别沿重叠率、尺度比、视角维度汇总。
- paired_comparisons.csv：success@10°共同成功、各自独有成功、共同失败数。
- figures/overview.png、pose_recall.png、by_group.png、difficulty.png、timing.png。

## 指标与公平比较

主几何真值始终采用rubik.json中的K1/K2和rel_pose；不混用nuScenes原始内参。当前不做两套标定的敏感性对照。

- 位姿误差=max(旋转误差,平移方向误差)，方向按本质矩阵符号歧义取较小角度。
- Success@5/10/20°：误差严格低于阈值的配对比例。
- AUC@5/10/20°：排序误差/召回点梯形积分后除以角度阈值。图显示经验阶梯召回；AUC使用所述插值规则。
- 失败估计和运行时失效输入均留在分母。
- 真值极线精度：用真值F计算Sampson距离（平方Sampson误差的平方根）≤3像素的比例。Macro每对平均，Micro合并全部匹配统计。空匹配的每对精度记0。
- RANSAC内点率和恢复位姿后的正深度内点数分开记录；不能当作真值匹配准确率。
- 匹配数量、Sampson中位距离、旋转角误差、平移方向角误差。
- 不计算米制平移误差，不需要UniDepth，也不是完整官方米制RUBIK成绩。

四种方法共用处理尺寸800×448和相同几何设置。--width/--height可统一更改，但应为32倍数。3像素阈值是在处理图像上定义的。

匹配计时预热3次，GPU同步；读图和统一缩放不计时，特征提取/设备传输/匹配计时，几何单独计时。SIFT在CPU，深度方法可在GPU，图标明设备；不能将跨设备耗时解释为纯算法速度差。

匹配可视化随机固定抽最多100条，绿线是真值Sampson距离合格，红线是其余点。随机种子固定。

## 验证与限制

核心测试命令：

```bash
python -m unittest discover -p 'test_*.py'
```

合成测试覆盖官方格式配对的prepare/run/analyze完整流程、samples/sweeps定位、不修改官方内参/真值、缺图报告、确定性子集、Sampson距离、位姿恢复、AUC与失败分母。

本机没有用户服务器图片和深度模型运行环境，未运行真实深度模型推理。模型适配按之前核对的官方接口编写；首次先跑小子集验证。

partial_results.json是运行中间记录，不是断点续跑保证。模型推理异常会直接报错；重跑同方法请用新目录。最终results.json存在才代表该方法全部图像对处理完毕。
