# 使用官方 Python 基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 复制 requirements.txt 并安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 安装 uvicorn（如果 requirements.txt 中没有）
RUN pip install --no-cache-dir uvicorn

# 复制项目文件
COPY . .

# 预下载模型（在构建时下载，这样容器启动时模型已经就绪）
# 使用环境变量设置设备为 cpu（构建时可能没有 GPU）
# 模型会缓存在 ~/.cache/modelscope/hub/ 目录中
ENV SENSEVOICE_DEVICE=cpu
ENV HF_HOME=/root/.cache/huggingface
ENV MODELSCOPE_CACHE=/root/.cache/modelscope
RUN mkdir -p /root/.cache/modelscope /root/.cache/huggingface && \
    python -c "import os; os.environ['SENSEVOICE_DEVICE']='cpu'; from model import SenseVoiceSmall; print('Downloading model iic/SenseVoiceSmall...'); m, kwargs = SenseVoiceSmall.from_pretrained(model='iic/SenseVoiceSmall', device='cpu'); print('Model downloaded successfully')"

# 暴露端口
EXPOSE 50000

# 设置环境变量（可以在运行时通过 -e 覆盖）
ENV SENSEVOICE_DEVICE=cuda:0

# 设置启动命令
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "50000"]

