# NVIDIAのCUDA Dockerイメージをベースにする
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu20.04

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    git \
    curl \
    unzip \
    file \
    xz-utils \
    sudo \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# CUDAを使用するための環境変数設定
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

# 日本語フォントのインストール
ENV NOTO_DIR /usr/share/fonts/opentype/notosans
RUN mkdir -p ${NOTO_DIR} && \
    wget -q https://noto-website-2.storage.googleapis.com/pkgs/NotoSansCJKjp-hinted.zip -O noto.zip && \
    unzip ./noto.zip -d ${NOTO_DIR}/ && \
    chmod a+r ${NOTO_DIR}/NotoSans* && \
    rm ./noto.zip

# nvidia-container-runtimeのインストール
RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | apt-key add - \
    && curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | tee /etc/apt/sources.list.d/nvidia-container-runtime.list \
    && apt-get update && apt-get install -y nvidia-container-runtime \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージのインストール
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -U pip setuptools wheel && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt

# コンテナ内で作業するディレクトリの指定
WORKDIR /workspace

# コンテナが起動したときに実行されるコマンドの指定
CMD ["bash"]
