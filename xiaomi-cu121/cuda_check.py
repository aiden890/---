import json
import torch
x=torch.arange(256,device='cuda',dtype=torch.float32).reshape(16,16)
y=x@x.T
torch.cuda.synchronize()
assert torch.allclose(y.cpu(),x.cpu()@x.cpu().T)
print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'bf16_supported':torch.cuda.is_bf16_supported(),'matmul':'passed'}),flush=True)
