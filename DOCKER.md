# Docker 使用说明

本项目提供了 Dockerfile 用于构建 Docker 容器，容器中已预装了所有依赖、ffmpeg 和模型文件。

## 构建 Docker 镜像

### 本地构建

```bash
docker build -t sensevoice:latest .
```

### 使用 GitHub Actions 构建

项目包含 GitHub Actions 工作流，当推送到 main/master 分支或创建 tag 时会自动构建并推送 Docker 镜像到 GitHub Container Registry (ghcr.io)。

## 运行 Docker 容器

### CPU 模式

```bash
docker run -d \
  --name sensevoice \
  -p 50000:50000 \
  -e SENSEVOICE_DEVICE=cpu \
  sensevoice:latest
```

### GPU 模式（需要 NVIDIA Docker）

```bash
docker run -d \
  --name sensevoice \
  --gpus all \
  -p 50000:50000 \
  -e SENSEVOICE_DEVICE=cuda:0 \
  sensevoice:latest
```

## 使用 API

容器启动后，可以通过以下方式访问 API：

- API 文档: http://localhost:50000/docs
- 健康检查: http://localhost:50000/

### 测试 API

```bash
# 使用 curl 测试
curl -X POST "http://localhost:50000/api/v1/asr" \
  -F "files=@your_audio.wav" \
  -F "lang=auto" \
  -F "keys=test_audio"
```

## 环境变量

- `SENSEVOICE_DEVICE`: 指定运行设备，可选值：
  - `cpu`: 使用 CPU
  - `cuda:0`: 使用第一个 GPU
  - `cuda:1`: 使用第二个 GPU（如果有）

## 注意事项

1. 模型在构建时已预下载，首次启动容器时无需等待模型下载
2. 如果使用 GPU，需要安装 NVIDIA Docker runtime
3. 默认端口为 50000，可以通过 `-p` 参数映射到其他端口

