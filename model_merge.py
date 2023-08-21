from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 设置原模型地址
model_name_or_path = ''  # 在这里填入原始模型的地址

# 设置微调后模型地址
adapter_name_or_path = ''  # 在这里填入微调后模型的地址

# 设置合并后模型导出地址
save_path = '/root/autodl-tmp/new_model'  # 新模型导出地址

# 从预训练模型中加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True
)

# 从预训练模型中加载Causal Language Model
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map='auto'
)
print("load model success")

# 从预训练模型中加载PeftModel适配器
model = PeftModel.from_pretrained(model, adapter_name_or_path)
print("load adapter success")

# 合并Peft适配器和模型权重，并释放适配器的内存
model = model.merge_and_unload()
print("merge success")

# 将分词器的配置保存到新模型导出地址
tokenizer.save_pretrained(save_path)

# 将合并后的模型保存到新模型导出地址
model.save_pretrained(save_path)
print("save done.")
