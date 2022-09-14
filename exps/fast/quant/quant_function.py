import math 

import torch
import torch.nn.functional as F
import numpy as np
from qtorch.number import FAST_BFP, Number


__all__ = ['fast_bfp_quant', "quantizer"]


def logr2(data):
    const = torch.zeros_like(data) + 2**(-0.5)
    return torch.log(data)/torch.log(const)

def r2(data):
    return (2**(-0.5))**(data)

def add_r_(data):
    r = torch.rand_like(data)
    data.add_(r)

def _calc_padding(fold, dim):
    times = math.ceil(dim/fold)
    return times*fold - dim 

def block_design(data, number, func):
    _dim = data.size()

    data = data.view(data.size()[0], -1)
    dim = data.size()
    num_pad = _calc_padding(number.group, data.size()[1])
    padding = tuple([0,num_pad])
    data = F.pad(input=data, pad=padding, mode='constant', value=0)
    dim_pad = data.size()

    data_unfold = data.unfold(1, number.group, number.group)
    data_f = data_unfold.contiguous().view([data_unfold.size(0), data_unfold.size(1), -1])
    max_entry = func(torch.abs(data_f), 2)
    shift_exponent = torch.ceil(torch.log2(max_entry+1e-28))
    shift_exponent = torch.clamp(shift_exponent, -2**(number.exp-1), 2**(number.exp-1)-1) # 当然要约束在一定的范围之内
    # print(shift_exponent)
    shift_exponent = shift_exponent.repeat(1, number.group)
    shift_exponent = shift_exponent.view(dim_pad)
    shift_exponent = shift_exponent[:dim[0], :dim[1]]

    return shift_exponent.view(_dim)


def fast_bfp_quant(x, number, rounding="stochastic"):

    assert isinstance(x, torch.Tensor), "x is not a single precision Floating Point Tensor"

    # shared exponent
    # mean_func = lambda x, dim: torch.mean(x, dim)
    max_func = lambda x, dim: torch.max(x, dim)[0]

    max_exponent = block_design(x, number, max_func) # 找到了最大的exp，然后需要进行移位操作

    # fast bfp
    offset = max_exponent - number.man # 除去相应内容之外，剩余的偏移 
    # shared exponent shifting
    shift = 2**(-offset)
    i = x * shift # 通过shift 操作得到了一系列的整数 （这里还应该取整） 
    i.floor_()

    mval = 2**(offset)
    mnumber = 2**(number.man)

    # sgn = torch.sign(i) # 首先求得相应的符号 (似乎不需要求绝对值)
    # i = torch.abs(i) # 求得其绝对值
    i.clamp_(1-mnumber, mnumber-1)


    # rounding on frac
    if rounding == "stochastic":
        r = torch.randint_like(i, 1-mnumber, mnumber)
        i.add_(r).floor_()
    # else:
    #     i.mul_(mval)
    # sign magnitude multiplication for subnormal and normal
    # k = torch.where(i<esbn, clipped, me+clipped)
    i.clamp_(1-mnumber, mnumber-1)
    out = i * mval
    return out


def quantizer(forward_number=None, backward_number=None,
              forward_rounding="stochastic", backward_rounding="stochastic",
              clamping_grad_zero=False, backward_hooks=[]):
    """
    Creates a quantization function to support quantizing forward and backward process differently.

    Args:
        - :param: forward_number (qtorch.Number, optional) : the number format used for forward quantization.
                  if is None, the quantization would be a identity mapping.
        - :param: backward_number (qtorch.Number, optional) : the number format used for backward quantization.
                  if is None, the quantization would be a identity mapping.
        - :param: forward_rounding (string) : rounding mode, \"stochastic\" or \"nearest\" (default: \"stochastic\")
        - :param: backward_rounding (string) : rounding mode, \"stochastic\" or \"nearest\" (default: \"stochastic\")
        - :param: clamping_grad_zero (bool) : zero out the gradient of numbers that are being clamped during forward propagation.
                  currently requires forward_number to be a fixed point number.
        - :param: backward_hooks (iterable) : iterable of functions that will be applied to gradients before backward quantization.
                  For example, this can be used to support custom scaling.

    Returns:
        A quantization function as specified (torch.Tensor -> torch.Tensor)
    """
    if forward_number is not None:
        if forward_number.exp == -1 or forward_number.man == -1:
            forward_number = None
    if backward_number is not None:
        if backward_number.exp == -1 or backward_number.man == -1:
            backward_number = None


    for rounding in [forward_rounding, backward_rounding]:
        assert rounding in ["stochastic", "nearest"], "invalid rounding type {:s}".format(rounding)
    for num in [forward_number, backward_number]:
        if num != None: assert isinstance(num, Number)

   
    # forward and backward quantisation functions
    tensor_type = "w" if backward_number is None else "x"
    forward_quant = lambda x, num, rd: fast_bfp_quant(x, num, rd)
    backward_quant = lambda x, num, rd: fast_bfp_quant(x, num, rd)  


    class Rounding(torch.autograd.Function):
        @staticmethod
        def forward(self, x):
            if forward_number==None: return x

            out = forward_quant(x.contiguous(), forward_number, forward_rounding)

            return out.clone()

        @staticmethod
        def backward(self, grad_output):
            if self.needs_input_grad[0]:
                if backward_number == None:
                    grad_input = grad_output
                else:
                    grad_input = backward_quant(grad_output.contiguous(), backward_number, 
                        backward_rounding)
            else:
                grad_input = None

            return grad_input.clone()

    return Rounding.apply

if __name__ == "__main__":
    num_type = FAST_BFP(exp=3, man=4, group=16)

    quantizer = quantizer(forward_number=num_type, forward_rounding="stochastic")

    x = torch.randn((16, 64, 28, 28))
    print(x)

    tmp = quantizer(x)

    print(tmp)
