# ---- 阶段 1: 安装依赖 ----
FROM python:3.13-alpine AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- 阶段 2: 运行环境 ----
FROM python:3.13-alpine

WORKDIR /app

COPY --from=builder /install /usr/local

COPY *.py ./
COPY routes/ routes/
COPY adapters/ adapters/
COPY utils/ utils/
COPY static/ static/

RUN mkdir -p data

EXPOSE 3029

CMD ["python", "start.py"]
