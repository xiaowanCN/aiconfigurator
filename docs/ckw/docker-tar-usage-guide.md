# Docker 容器 tar 包使用指南

> 本文档适用于 `collector-vllm_v0804_1_neimeng.tar`，该文件是 **OCI 格式镜像包**。

---

## 一、文件信息

- **文件路径**: `docker/mat/collector-vllm_v0804_1_neimeng.tar`
- **镜像名称**: `collector-vllm_v0804.1:neimeng`
- **格式**: OCI Image Format（包含 `blobs/sha256/` 分层结构）
- **提取结果**: `docker/mat/workspace-extracted/workspace/`（约 501MB）

---

## 二、提取出的数据结构

```
docker/mat/workspace-extracted/workspace/
├── aic-core/           # aic-core 源码
├── collector/          # Collector 工作目录（主要数据）
│   ├── .collector_checkpoint/
│   ├── cases/          # 测试用例
│   ├── *.parquet       # 采集的性能数据（5个文件）
│   ├── *.log           # 运行日志
│   └── collection_meta.yaml
└── pyarrow-*.whl       # pyarrow 安装包
```

### 主要数据文件

| 文件类型 | 数量 | 说明 |
|---------|------|------|
| `*.parquet` | 5 | 采集的性能数据 |
| `*.log` | 1 | 运行日志 |
| `*.yaml` | 1 | 采集元数据 |

---

## 三、从 OCI tar 包提取 workspace 的方法

由于 OCI 格式是分层存储，不能直接解压。需要识别包含 `/workspace` 的层并逐层提取：

### 步骤 1：识别包含 workspace 的层

```python
import tarfile, json

with open('manifest.json', 'r') as f:
    m = json.load(f)
layers = m[0]['Layers']

main_tar = tarfile.open('collector-vllm_v0804_1_neimeng.tar', 'r')
for i, layer_path in enumerate(layers):
    layer_file = main_tar.extractfile(layer_path)
    layer_tar = tarfile.open(fileobj=layer_file)
    has_workspace = any('workspace' in m.name for m in layer_tar.getmembers())
    if has_workspace:
        print(f'层 {i}: {layer_path}')
    layer_tar.close()
```

### 步骤 2：提取相关层

```python
import tarfile

workspace_layers = [
    'blobs/sha256/f4467534038d00f9003ce4747eb6acb09bb804ce957c507163efd3327232c881',
    'blobs/sha256/a46925b715024d88972793c3678233a183bc4b9d671cf3eee9e1ee9a2ac0120b',
    'blobs/sha256/7b807ba65edcfa7e15441590df83d9d999a686aae7d3016aab00931284b33ac3',
    'blobs/sha256/50d2982edee6b745c241084da4892796e8b968bd2d9f2731d54c25f5744c9a2a',
    'blobs/sha256/451501a88c3825c3b782853adfedeeb380db43c31743345eca8a29a388db813b',
]

main_tar = tarfile.open('collector-vllm_v0804_1_neimeng.tar', 'r')
output_dir = 'workspace-extracted'

for layer_path in workspace_layers:
    layer_file = main_tar.extractfile(layer_path)
    layer_tar = tarfile.open(fileobj=layer_file)
    for member in layer_tar.getmembers():
        if member.name.startswith('workspace/'):
            layer_tar.extract(member, path=output_dir)
    layer_tar.close()

main_tar.close()
```

---

## 四、使用 Docker 加载镜像（完整方式）

如果需要完整的容器环境（包含所有依赖），使用 Docker 加载：

```bash
# 加载镜像（约 20GB，需要时间）
docker load -i docker/mat/collector-vllm_v0804_1_neimeng.tar

# 查看镜像
docker images | grep collector

# 运行容器
docker run -it --gpus all --ipc=host --network=host \
  collector-vllm_v0804.1:neimeng \
  /bin/bash

# 或创建临时容器提取数据
CID=$(docker create collector-vllm_v0804.1:neimeng)
docker cp $CID:/workspace/collector ./collector-data
docker rm $CID
```

---

## 五、两种方式对比

| 方式 | 优点 | 缺点 |
|------|------|------|
| **直接提取层** | 快速，不需要 Docker | 只能获取文件，丢失环境和依赖 |
| **Docker 加载** | 完整环境，可运行 | 需要 Docker，加载时间长 |

---

## 六、提取结果

已成功提取到 `docker/mat/workspace-extracted/workspace/`：
- 采集数据（parquet 文件）
- 运行日志
- Collector 源码和配置
- aic-core 源码
