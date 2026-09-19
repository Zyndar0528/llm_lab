import torch
import time
print (torch.__version__)
print (torch.cuda.get_device_name(0))
a = torch.randn(4096,4096,device='cuda')
b = torch.randn(4096,4096,device='cuda')
t0 = time.perf_counter()
for i in range(200):
    c = a @ b
torch.cuda.synchronize()
t1 = time.perf_counter()
t = t1 - t0
peak = torch.cuda.max_memory_allocated()
print (f"峰值显存{peak / 1048576:.1f} MB")
print (f"总时长{t:.3f} 秒,平均每次时长{t / 20:.3f} 秒")