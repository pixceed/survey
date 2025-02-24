# ベースイメージ
FROM ubuntu:24.04

# Set proxy
ARG http_proxy=""
ARG https_proxy=""
ENV HTTP_PROXY=${http_proxy} \
    HTTPS_PROXY=${https_proxy} \
    http_proxy=${http_proxy} \
    https_proxy=${https_proxy}
RUN sh -c 'echo Acquire::http::proxy \"${http_proxy}\"\;' >> /etc/apt/apt.conf
RUN sh -c 'echo Acquire::https::proxy \"${https_proxy}\"\;' >> /etc/apt/apt.conf

# タイムゾーンの設定
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 日本語表示用環境設定
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# 環境変数設定
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH "${PYTHONPATH}:/home/ubuntu/workspace/src"

# aptモジュールインストール
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.11 python3.11-venv
RUN apt-get install -y --no-install-recommends python3-pip
RUN apt-get install -y less vim sudo wget curl git zip unzip
RUN apt-get install -y python-is-python3
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y libx11-6 libx11-dev
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libxrender1
RUN apt-get install -y libxrender-dev
RUN apt-get install -y libxcb-xinerama0 libxcb1 libx11-xcb1 libxcomposite1 libxcursor1 libxrandr2 libxi6 libxrender1
RUN apt-get install -y qt5-default
RUN apt-get install -y libqt5x11extras5

ENV QT_QPA_PLATFORM_PLUGIN_PATH /home/ubuntu/venv/lib/python3.11/site-packages/cv2/qt/plugins


## 権限定義 ##
ARG user_name=ubuntu
ARG user_id=1000
ARG group_name=ubuntu
ARG group_id=1000

# グループが存在しない場合のみ作成
RUN getent group ${group_name} || groupadd -g ${group_id} ${group_name}
# ユーザーが存在しない場合のみ作成
RUN id -u ${user_name} &>/dev/null || useradd -u ${user_id} -g ${group_id} -d /home/${user_name} --create-home --shell /usr/bin/bash ${user_name}
RUN echo "${user_name} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN chown -R ${user_name}:${group_name} /home/${user_name}

# ワークスペース定義
USER ubuntu
ENV HOME /home/ubuntu
WORKDIR /home/ubuntu/workspace

# 仮想環境の作成と有効化
RUN python3.11 -m venv /home/ubuntu/venv
ENV PATH="/home/ubuntu/venv/bin:$PATH"


# パッケージインストール
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# --------------------------- Genesis ----------------------------
RUN pip install --no-cache-dir open3d
RUN git clone https://github.com/Genesis-Embodied-AI/Genesis.git && \
    cd Genesis && \
    pip install . && \
    pip install --no-cache-dir PyOpenGL==3.1.5


# パッケージインストール
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt



