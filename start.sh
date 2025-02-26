#!/bin/bash
set -eo pipefail

# 定义颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # 恢复默认颜色

is_port_used() {
    local port="$1"
    if ss -tuln | awk -v p="$port" '$5 ~ ":"p"$" {exit 0} END {exit 1}'; then
        return 0
    else
        return 1
    fi
}

# 检查并安装 Docker
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${YELLOW}Docker 未安装，正在尝试安装...${NC}"
        curl -fsSL https://get.docker.com | sudo sh
        sudo usermod -aG docker $USER
        echo -e "${GREEN}Docker 安装完成，请重新登录以应用用户组更改${NC}"
        exit 1
    else
        echo -e "${GREEN}Docker 已安装 (版本: $(docker --version | cut -d ' ' -f 3 | tr -d ','))${NC}"
    fi
}

# 检查并安装 Docker Compose
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${YELLOW}Docker Compose 未安装，正在安装...${NC}"
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
            -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
        echo -e "${GREEN}Docker Compose 安装完成 (版本: $(docker-compose --version | cut -d ' ' -f 3))${NC}"
    else
        echo -e "${GREEN}Docker Compose 已安装 (版本: $(docker-compose --version | cut -d ' ' -f 3))${NC}"
    fi
}

# 检查并安装 Go 环境
check_go() {
    if ! command -v go &> /dev/null; then
        echo -e "${YELLOW}Go 未安装，正在安装...${NC}"
        GO_VERSION="1.21.1"
        curl -OL https://golang.org/dl/go${GO_VERSION}.linux-amd64.tar.gz
        sudo tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz
        echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
        source ~/.bashrc
        rm go${GO_VERSION}.linux-amd64.tar.gz
        echo -e "${GREEN}Go 安装完成 (版本: $(go version | cut -d ' ' -f 3))${NC}"
    else
        echo -e "${GREEN}Go 已安装 (版本: $(go version | cut -d ' ' -f 3))${NC}"
    fi
}

main() {
    echo -e "\n=== 环境检查 ==="
    check_docker
    check_docker_compose
    check_go

    echo -e "\n=== 安装依赖 ==="
    go env -w GOPROXY=https://goproxy.cn,direct
    go get github.com/bytedance/godlp@latest
    echo -e "${GREEN}text模块安装成功${NC}"

    echo -e "\n=== 启动图像处理服务 ==="
    cd ./image || { echo -e "${RED}错误：image 目录不存在${NC}"; exit 1; }
    
    # 新增 8000 端口检查
    if is_port_used 8000; then
        echo -e "${YELLOW}8000 端口已被占用，跳过 Docker 服务启动${NC}"
    else
        docker-compose up -d --build
    fi
    docker ps --format "table {{.Names}}\t{{.Status}}"

    echo -e "\n=== 构建文本处理模块 ==="
    cd ../text || { echo -e "${RED}错误：text 目录不存在${NC}"; exit 1; }
    #make
    
    # 新增 8081 端口检查
    if is_port_used 8081; then
        echo -e "${YELLOW}8081 端口已被占用，跳过服务启动${NC}"
    else
	make
        make run &
    fi

    echo -e "\n=== 安装python环境 ==="
    cd .. && pip install -r requirements.txt
    python3 webui.py
}

# 执行主函数
main
