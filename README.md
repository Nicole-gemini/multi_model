# 双卡多模态LLM：轻量化训练与对话落地
### 参考项目：https://github.com/wyf3/llm_related/tree/main/train_multimodal_from_scratch； 
### 改进点：项目代码结构调整/动态门控/多轮记忆机制/DPO数据生成与模型训练/DeepSpeed ZeRO-2与混合并行

## 一. 项目简介
• 针对企业多模态大模型落地中高算力依赖与训练流程复杂的痛点，基于个人双卡GPU环境搭建端到端多模态训练pipeline（数据预处理→基础训练→SFT→DPO），为中小企业提供低成本多模态技术落地参考。
• 采用DeepSpeed ZeRO-2与混合并行，降低显存占用近50%。
• 动态门控融合与多轮记忆机制显著提升图文对齐与对话连贯性，图文相关性评分提高29%。

## 二. 环境设置(在featurize租用了4090双卡)
### bash

###### #安装 flashattention
pip install flash-attn==2.3.0 --no-build-isolation

###### #安装 vllm
pip install vllm==0.7.3

pip install fastapi uvicorn

###### #安装modelscope库，国内下载
pip install modelscope

###### #下载文本模型
mkdir /home/featurize/data/multimodal/text_model # 创建并进入目录
cd /home/featurize/data/multimodal/text_model
sudo apt-get install git-lfs
git lfs install
git clone https://www.modelscope.cn/Qwen/Qwen2.5-1.5B-Instruct

###### #下载视觉模型
mkdir /home/featurize/data/multimodal/visual_model # 创建并进入目录
cd /home/featurize/data/multimodal/visual_model

- 方法一：
下载文件：https://hf-mirror.com/google/siglip-base-patch16-224
上传文件到目录中

- 方法二：
sudo apt-get install git-lfs
git lfs install
git clone https://www.modelscope.cn/models/Xenova/siglip-base-patch16-224

###### #降级 peft 版本（llamafactory 需要 peft 版本在 0.11.1 到 0.12.0 之间）
pip install --no-cache-dir "peft>=0.11.1,<0.13.0"

###### #升级 transformers 版本
pip install --no-cache-dir --upgrade "transformers==4.48.2"

###### #降级 tokenizers 到兼容版本
pip install --no-cache-dir "tokenizers<=0.21.0,>=0.19.0"


## 三. 模型
### 运⾏下⾯的命令开始训练（必须按顺序）：
###### #后台挂起终端
###### #继续预训练模型
tmux new-session -d -s mysession "python pretrain.py"
tmux attach-session -t mysession
###### #sft模型
tmux new-session -d -s mysession "python sft_train.py"
tmux attach-session -t mysession
###### #dpo模型
tmux new-session -d -s mysession "python dpo_train.py"
tmux attach-session -t mysession

### 测试
python test.py
