import torch
from qtorch.quant import *
from qtorch import BlockFloatingPoint



if __name__ == "__main__":
    a = torch.randint(1000, (8,8)).float()
    quant_base = lambda x: block_quantize(x, wl=3, dim=-1, rounding='nearest')
    quant_check = lambda x: block_quantize(x, wl=3, dim=8, rounding='nearest')
    out_base = quant_base(a)
    out_check = quant_check(a)
    print(a)
    print('output baseline is : \n ', out_base)
    print('output checkpoint is : \n', out_check)