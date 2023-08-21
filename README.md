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
4. ```
import wandb
wandb.init()```
需要先到：https://wandb.me/wandb-server 注册wandb  
然后到：https://wandb.ai/authorize 复制key出来  



## 文档说明
* 
