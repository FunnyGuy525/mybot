# mybot
使用聊天记录微调一个“我”

## 项目描述
>  详细的项目描述
1. 预训练模型选用Llama中文社区开源的Llama2-Chinese-13B模型  https://github.com/FlagAlpha/Llama2-Chinese  
   开源模型中llama2能力很强，但其官方模型预训练语料中的中文仅占0.13%，导致其中文能力较弱，所以选择基于200B中文语料训练的Llama2-Chinese-13B模型  
2. 微调方法选择lora，其在原有的模型推理参数旁训练一个旁支，并采用先降维再升维的方式降低参数量，使得训练参数变少。
3. 训练集可以由导出的聊天记录使用DialogPreprocess.ipynb文件进行预先处理


## 运行条件
* 本人使用GPU型号为 **V100 32G** 、单卡
* 建议在Linux系统操作，数据盘至少保留80G

## 运行说明
### 模型部署
1. 首先安装git-lfs(Large File Storage, 用于帮助git管理大文件)
   ```
   curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
   sudo apt-get install git-lfs
   ```
2. 下载模型(可能需要梯子加速)  
   ```git clone https://huggingface.co/RicardoLee/Llama2-chat-13B-Chinese-50W```  
   下载结束进入huggingface项目地址，比对本地项目文件，模型文件较大可能下载失败需要手动下载。
3. 下载并部署gradio
执行 ```pip install -r requirements.txt```   
其中grdio模块安装较慢，耐心等待   
gradio是一种用于构建AI界面的开源库，可以快速构建自己的应用程序并与AI模型进行交互。，这样就不用在黑黢黢的终端窗口使用大模型了。  
4. 运行gradio_demo.py  
将modelpath改为自己刚才下载的模型路径，如/root/Llama2-chat-13B-Chinese-50W   
为避免无法跑通，可以在8bit进行量化    
```python gradio_demo.py --base_model modelpath --tokenizer_path modelpath --load_in_8bit --gpus 0```   
无8bit量化版本   
```python gradio_demo.py --base_model modelpath --tokenizer_path modelpath --gpus 0```  
点击public URL（只有72h有效期）即可进入外部分享链接    


### 模型微调
1. 数据预处理
使用DialogPreprocess.ipynb笔记本 notebook文件  
可以直接用QQ导出聊天文件txt文本，处理后得到训练集json文件train.json
2. 接着notebook文件运行安装程序，使用-U强制升级到最新版本
```
!pip install -q huggingface_hub
!pip install -q -U trl transformers accelerate peft
!pip install -q -U datasets bitsandbytes einops wandb
```
3. 首先需要到：https://huggingface.co/settings/tokens 复制token，需要能访问外网
在setting页面，创建一个新的access token
```
from huggingface_hub import notebook_login
notebook_login()
```
按照提示输入token，如果失败刷新重试              
4. 运行wandb初始化命令
```
import wandb
wandb.init()
```
需要先到：https://wandb.me/wandb-server 注册wandb  
然后到：https://wandb.ai/authorize 复制key出来  
5. 导入相关包，配置参数
```
from datasets import load_dataset
import torch
import einops
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

# 加载训练数据集，'train_data_path' 应为训练数据文件的路径
dataset = load_dataset("json", data_files=train_data_path, split="train")

# 基础模型的名称或路径，例如 "/root/Llama2-chat-13B-Chinese-50W"
base_model_name = modelpath

# 定义 BitsAndBytesConfig 配置，用于量化模型权重
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 使用4位量化
    bnb_4bit_use_double_quant=True,  # 使用双量化以减少参数位数
    bnb_4bit_quant_type="nf4",  # 使用标准化浮点数（normalized float）进行4位量化
    bnb_4bit_compute_dtype=torch.float16,  # 使用半精度浮点数进行计算
)

# 定义设备映射，将模型加载到指定的GPU设备
device_map = {"": 0}
# 如果有多个GPU，可以使用类似以下方式的映射：device_map = {"": [0, 1, 2, 3, ...]}

# 从预训练模型加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,  # 本地模型名称或路径
    quantization_config=bnb_config,  # 量化配置
    device_map=device_map,  # 使用的GPU编号
    trust_remote_code=True,  # 允许远程代码
    use_auth_token=True  # 使用授权令牌（如果需要）
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# 定义 LoraConfig 配置，用于 PeftModel
peft_config = LoraConfig(
    lora_alpha=16,  # LORA alpha 参数
    lora_dropout=0.1,  # LORA dropout 参数
    r=64,  # LORA中嵌入的位置编码数量
    bias="none",  # 不使用任何偏置
    task_type="CAUSAL_LM",  # 任务类型为Causal Language Model
)

# 加载预训练分词器
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # 将 pad_token 设置为 eos_token，以便正确填充数据

# 定义训练输出目录
output_dir = "./results"

# 定义 TrainingArguments 训练配置
training_args = TrainingArguments(
    report_to="wandb",  # 使用wandb进行报告和日志记录
    output_dir=output_dir,  # 训练后的输出目录
    per_device_train_batch_size=4,  # 每个GPU的批次大小
    gradient_accumulation_steps=4,  # 累积梯度的步骤数
    learning_rate=2e-4,  # 初始学习率
    logging_steps=10,  # 每隔多少步记录一次日志
    max_steps=500  # 总训练步数
)
max_seq_length = 512  # 最大序列长度

# 初始化 SFTTrainer
trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",  # 数据集中文本字段的名称
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
)

```
6. 开始训练
```trainer.train()```
如果训练时间过长无法接受，可以降低training_args参数中的max_steps
8. 保存模型
```
import os
output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)
```
8. 模型合并  
设置model_merge.py中相关模型路径并运行 



## 文档说明
* 
