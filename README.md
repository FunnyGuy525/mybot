# mybot
使用聊天记录微调一个“我”

## 项目描述
>  详细的项目描述
1. 预训练模型选用Llama中文社区开源的Llama2-Chinese-13B模型  https://github.com/FlagAlpha/Llama2-Chinese
   开源模型中llama2能力很强，但其官方模型预训练语料中的中文仅占0.13%，导致其中文能力较弱，所以选择基于200B中文语料训练的Llama2-Chinese-13B模型
2. 微调方法选择lora，其在原有的模型推理参数旁训练一个旁支，并采用先降维再升维的方式降低参数量，使得训练参数变少。
3. 训练集可以由导出的聊天记录使用DialogPreprocess.ipynb文件进行预先处理


## 运行说明
* 



## 文档说明
* 

## 测试说明
* 
