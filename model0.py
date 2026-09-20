import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

    # 加载分词器和权重
MODEL = "Qwen/Qwen3-0.6B"
t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype = torch.bfloat16).to("cuda")
torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"加载耗时 {t1 - t0:.2f} 秒")      # 记录加载耗时

    # 把自然语言问题转化为模型可理解的输入
messages = [{"role": "user", "content": "用一句话解释什么是显存"}]
text = tok.apply_chat_template(messages,tokenize = False,
                               add_generation_prompt = True, enable_thinking = False)
inputs = tok(text, return_tensors = "pt").to("cuda")
torch.cuda.reset_peak_memory_stats()    # 重置显存峰值记录
t0 = time.perf_counter()
out = model.generate(**inputs, max_new_tokens = 100)
torch.cuda.synchronize()
t1 = time.perf_counter()
new_tokens = out.shape[-1] - inputs["input_ids"].shape[-1]
dt = t1 - t0
print(f"生成 {new_tokens} 个 token,耗时 {dt:.2f} 秒，每秒 {new_tokens / dt:.1f} 个")
print(f"峰值显存 {torch.cuda.max_memory_allocated() / 1048576:.1f} MB")
print("--- 模型输出 ---")
print(tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
