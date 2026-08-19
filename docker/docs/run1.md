主要改动：

在 docker/Dockerfile 末尾新增了 runtime-with-source 阶段（第188-237行），它会：

1. 从 base 阶段开始（包含 Python 虚拟环境和 uv）
2. 安装 git 用于开发工作流
3. 复制完整的项目源代码到 /workspace/
4. 验证 Git LFS 内容已正确下载
5. 安装 Rust 工具链（编译 aic-core 的 Rust 扩展所需）
6. 以开发模式（editable install）安装项目，这样源代码修改可以立即生效

构建命令：

# 在项目根目录执行
docker build --target runtime-with-source -t aiconfigurator-runtime:v0804.1 -f docker/Dockerfile .

运行容器：

# 交互式运行
docker run -it aiconfigurator-runtime:v0804.1

# 或挂载本地目录
docker run -it -v $(pwd):/workspace aiconfigurator-runtime:v0804.1

容器内项目源代码位于 /workspace/，并且以 editable 模式安装，修改源代码后无需重新安装即可生效。
