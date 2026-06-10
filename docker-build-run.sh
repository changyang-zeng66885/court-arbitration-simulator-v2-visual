#!/usr/bin/env bash
#
# 一键构建 & 运行脚本
# 用法:
#   chmod +x docker-build-run.sh
#   ./docker-build-run.sh                          # 使用默认设置
#   OPENAI_API_KEY=sk-xxx ./docker-build-run.sh    # 指定 API Key
#   OPENAI_BASE_URL=https://api.deepseek.com OPENAI_MODEL=deepseek-v4-flash ./docker-build-run.sh
#

set -euo pipefail

cd "$(dirname "$0")"

IMAGE_NAME="${IMAGE_NAME:-court-arbitration-simulator}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_NAME="${CONTAINER_NAME:-court-arbitration-simulator}"
PORT="${PORT:-8501}"

# ------ 检查 Docker ------
if ! command -v docker &>/dev/null; then
  echo "❌ 未检测到 Docker，请先安装 Docker。"
  echo "   安装指南: https://docs.docker.com/engine/install/"
  exit 1
fi

if ! docker ps &>/dev/null; then
  echo "❌ Docker 守护进程未运行，请先启动 Docker。"
  exit 1
fi

# ------ 停止并清理旧容器 ------
if docker ps -a --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
  echo "🧹 发现旧容器，正在清理..."
  docker stop "${CONTAINER_NAME}" 2>/dev/null || true
  docker rm "${CONTAINER_NAME}" 2>/dev/null || true
fi

# ------ 构建镜像 ------
echo "🔨 正在构建 Docker 镜像: ${IMAGE_NAME}:${IMAGE_TAG} ..."
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .

# ------ 运行容器 ------
echo "🚀 正在启动容器..."
docker run -d \
  --name "${CONTAINER_NAME}" \
  -p "${PORT}:8501" \
  -e OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
  -e OPENAI_BASE_URL="${OPENAI_BASE_URL:-}" \
  -e OPENAI_MODEL="${OPENAI_MODEL:-qwen3.6-flash}" \
  -v "$(pwd)/outputs/streamlit_runs:/app/outputs/streamlit_runs" \
  --restart unless-stopped \
  "${IMAGE_NAME}:${IMAGE_TAG}"

# ------ 等待启动 ------
echo "⏳ 等待服务启动..."
sleep 3

if docker ps --format "{{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
  echo ""
  echo "✅ 容器已成功启动！"
  echo "   访问地址: http://localhost:${PORT}"
  echo ""
  echo "📋 常用命令:"
  echo "   查看日志: docker logs -f ${CONTAINER_NAME}"
  echo "   停止容器: docker stop ${CONTAINER_NAME}"
  echo "   重启容器: docker restart ${CONTAINER_NAME}"
  echo ""
else
  echo "❌ 容器启动失败，请查看日志:"
  docker logs "${CONTAINER_NAME}" 2>/dev/null || true
  exit 1
fi
