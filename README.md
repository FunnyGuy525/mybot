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
   
* 修改requrement.txt里的torch版本，可以修改为自己本地已有版本，若无建议下载最新版本  
执行 ```pip install -r requirements.txt```

### 模型微调
1. 数据预处理
可以直接用QQ导出聊天比较多的好友

## 文档说明
* 
