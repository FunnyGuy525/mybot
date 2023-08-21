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
import torch,einops
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

dataset = load_dataset("json",data_files=train_data_path,split="train")
base_model_name = modelpath  # "/root/Llama2-chat-13B-Chinese-50W"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,#在4bit上，进行量化
    bnb_4bit_use_double_quant=True,# 嵌套量化，每个参数可以多节省0.4位
    bnb_4bit_quant_type="nf4",# NF4（normalized float）或纯FP4量化
    bnb_4bit_compute_dtype=torch.float16,
)
device_map = {"": 0}
#有多个gpu时，为：device_map = {"": [0,1,2,3……]}
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,#本地模型名称
    quantization_config=bnb_config,#上面本地模型的配置
    device_map=device_map,#使用GPU的编号
    trust_remote_code=True,
    use_auth_token=True
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

output_dir = "./results"
training_args = TrainingArguments(
    report_to="wandb",
    output_dir=output_dir,#训练后输出目录
    per_device_train_batch_size=4,#每个GPU的批处理数据量
    gradient_accumulation_steps=4,#在执行反向传播/更新过程之前，要累积其梯度的更新步骤数
    learning_rate=2e-4,#超参、初始学习率。太大模型不稳定，太小则模型不能收敛
    logging_steps=10,#两个日志记录之间的更新步骤数
    max_steps=100#要执行的训练步骤总数
)
max_seq_length = 512
#TrainingArguments 的参数详解：https://blog.csdn.net/qq_33293040/article/details/117376382

trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
)
```
6. 开始训练
```trainer.train()```
7. 保存模型
```
import os
output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)
```
8. 模型合并  
设置model_merge.py中相关模型路径并运行 



## 文档说明
* 
