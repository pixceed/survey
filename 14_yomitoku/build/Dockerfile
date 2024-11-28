# ベースイメージ
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04


# Set proxy
ARG http_proxy=""
ARG https_proxy=""
ENV HTTP_PROXY=${http_proxy} \
    HTTPS_PROXY=${https_proxy} \
    http_proxy=${http_proxy} \
    https_proxy=${https_proxy}
RUN sh -c 'echo Acquire::http::proxy \"${http_proxy}\"\;' >> /etc/apt/apt.conf
RUN sh -c 'echo Acquire::https::proxy \"${https_proxy}\"\;' >> /etc/apt/apt.conf

# 日本語表示用環境設定
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# 環境変数設定
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH "${PYTHONPATH}:/home/ubuntu/workspace/src"

# apt installする際に、対話式の命令をスキップする
ENV DEBIAN_FRONTEND "noninteractive"

# aptモジュールインストール
ARG PYTHON_VERSION=3.9
RUN apt -y update && apt -y upgrade
RUN apt install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    curl \
    wget \
    git \
    ca-certificates \
    poppler-utils \
    libopencv-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt update \
    && apt install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*


# ライブラリインストール
RUN python${PYTHON_VERSION} --version
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1
RUN python -m pip install --upgrade pip
RUN pip install yomitoku


## 権限定義 ##
ARG user_name=ubuntu
ARG user_id=1000
ARG group_name=ubuntu
ARG group_id=1000

RUN groupadd -g ${group_id} ${group_name}
RUN useradd -u ${user_id} -g ${group_id} -d /home/${user_name} --create-home --shell /usr/bin/bash ${user_name}
RUN echo "${user_name} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN chown -R ${user_name}:${group_name} /home/${user_name}

# ワークスペース定義
USER ubuntu
ENV HOME /home/ubuntu
WORKDIR /home/ubuntu/workspace

# # パッケージインストール
# COPY requirements.txt .
# RUN pip install --upgrade pip
# RUN pip install --no-cache-dir -r requirements.txt
