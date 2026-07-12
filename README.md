# LTX 2.3 A2V LoRA Lab

面向 LTX-2.3 `a2vid_two_stage` 的本地 Web UI。服务进程会保留当前 Pipeline，连续修改 prompt、音频、图片、seed、分辨率或采样参数时不再重新加载 22B 模型。

## 主要能力

- Pipeline 常驻显存/内存；按模型配置指纹判断是否需要重载
- 单 GPU 安全任务队列，关闭浏览器不影响正在执行的任务
- 读取音频实际时长，自动计算不超过音频长度的 `8k+1` 帧数
- 自动检查两阶段分辨率（64 的倍数）、帧数、关键帧位置、文件路径和输出格式
- 支持多个测试 LoRA、Distilled LoRA、FP8、CPU/磁盘 offload 与完整 A2V guidance 参数
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
- 量化与 offload 模式

prompt、负向 prompt、音频、图片、输出路径、时长、帧率、分辨率、seed、采样步数和 guidance 参数都不会触发模型重载。

> LTX 官方实现会在 Pipeline 初始化时把 LoRA 交给 DiffusionStage，因此调整 LoRA 强度仍需重建 Pipeline。UI 会在提交前明确显示“将加载新模型配置”或“将复用已加载模型”。

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
