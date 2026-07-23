# LTX 2.3 A2V LoRA Lab

面向 LTX-2.3 `a2vid_two_stage` 的本地 Web UI。服务进程会保留当前 Pipeline，连续修改 prompt、音频、图片、seed、分辨率或采样参数时不再重新加载 22B 模型。

## 主要能力

- Pipeline 常驻显存/内存；按模型配置指纹判断是否需要重载
- 单 GPU 安全任务队列，关闭浏览器不影响正在执行的任务
- 读取音频实际时长，自动计算不超过音频长度的 `8k+1` 帧数
- 自动检查两阶段分辨率（64 的倍数）、帧数、关键帧位置、文件路径和输出格式
- 支持多个测试 LoRA、Distilled LoRA、FP8、CPU/磁盘 offload、`torch.compile` 与完整 A2V guidance 参数
- CUDA 同步分阶段 profiling：扩散、条件编码、上采样、VAE 解码和 MP4 封装分别计时
- 浏览器上传音频/图片、配置预设、历史任务、生成进度、视频预览/下载

## 安装

推荐把本项目放在已经能执行 `ltx_pipelines.a2vid_two_stage` 的 LTX-2 环境中。假设 LTX-2 在：

```bash
cd /home/us5090/workspace/niro-workspace/LTX-2
uv pip install -e /path/to/Ltx2.3-UI
```

如果 `ltx-pipelines` 尚未安装：

```bash
uv pip install -e packages/ltx-core -e packages/ltx-pipelines
uv pip install -e /path/to/Ltx2.3-UI
```

开发环境（包含测试工具）：

```bash
uv sync --extra dev
```

## 启动

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run ltx23-ui
```

浏览器打开 `http://服务器地址:7860`。通过 SSH 使用时可建立端口转发：

```bash
ssh -L 7860:127.0.0.1:7860 user@server
```

随后在本机打开 `http://127.0.0.1:7860`。

上传文件默认保存到 `~/.ltx23-ui/uploads`；可用环境变量覆盖：

```bash
LTX_UI_DATA_DIR=/data/ltx-ui uv run ltx23-ui
```

单个文件默认最大 2048 MB，可按需要调整：

```bash
LTX_UI_MAX_UPLOAD_MB=4096 uv run ltx23-ui
```

如果上传提示无法连接，请先直接访问 `http://服务器地址:7860/api/health`。返回 JSON
说明后端正常；无法访问则需要检查 `ltx23-ui` 终端日志、SSH 端口转发或反向代理的上传大小限制。

## 模型复用规则

以下字段不变时，下一次任务会直接复用当前 Pipeline：

- checkpoint、Gemma、空间上采样模型
- Distilled LoRA 的路径和强度
- 测试 LoRA 列表、路径和强度
- 量化、offload 与 `torch.compile` 模式

prompt、负向 prompt、音频、图片、输出路径、时长、帧率、分辨率、seed、采样步数和 guidance 参数都不会触发模型重载。

> LTX 官方实现会在 Pipeline 初始化时把 LoRA 交给 DiffusionStage，因此调整 LoRA 强度仍需重建 Pipeline。UI 会在提交前明确显示“将加载新模型配置”或“将复用已加载模型”。

## torch.compile 与性能分析

官方 `a2vid_two_stage` 在不传参数时不会自动启用 `torch.compile`。直接使用 CLI
需要显式添加：

```bash
uv run python -m ltx_pipelines.a2vid_two_stage \
  --compile mode=reduce-overhead \
  ...
```

本 UI 默认选择 `Reduce Overhead`，并把
`CompilationConfig(mode="reduce-overhead")` 传给两个 DiffusionStage。可在“模型”页切换或关闭；
改变编译模式会触发 Pipeline 重载。

高级生成参数里的“记录 CUDA 同步性能报告”默认开启。每个任务结束后，报告会：

- 在服务终端按耗时从高到低输出各阶段、百分比、每步平均/P95/最大耗时及采样循环外开销
- 记录 CUDA 峰值 allocated/reserved 显存
- 出现在任务卡片和右侧生成面板
- 通过 `GET /api/jobs/<job_id>/profile` 返回完整 JSON

`torch.compile` 的第一次生成包含图捕获和编译成本。定位稳态瓶颈时，应保持模型配置、
分辨率、帧数和最大批次不变，连续生成至少两次，并以第二次的“模型复用热运行”报告为准。
性能分析会在阶段边界执行 CUDA 同步；完成定位后可以取消勾选，以测量无 profiling 干扰的最终速度。

CPU offload 下，官方实现的 guidance 最多可把 4 个 pass 合并为一个 batch。
如果显存足够，可把“最大批次”从 1 逐步提高到 4，减少逐层 PCIe 权重搬运；若出现 OOM，
退回 2 或 1。磁盘 offload 通常最慢，只建议在 CPU 内存也不足时使用。

## 帧数说明

当前官方 LTX-2.3 Pipeline 要求帧数满足 `8k+1`。自动模式选择不超过所选音频时长的最大合法值：

```text
16 秒 × 25 FPS = 400 个可用帧 → 393 帧 → 15.72 秒
```

原命令中的 `376` 不满足这个约束，UI 会阻止提交并给出提示。

## 测试

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
```

测试不会加载 LTX 模型或占用 GPU。
