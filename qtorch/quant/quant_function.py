import torch
from qtorch import Number, FixedPoint, BlockFloatingPoint, FloatingPoint
import quant_cuda
import quant_cpu
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['fixed_point_quantize', 'block_quantize', 'float_quantize', "quantizer"]
def assert_wl_fl(wl, fl, stage):
    if wl == -1 and fl != -1:
        raise ValueError("fixed point {} wl {}, fl {}".format(stage, wl, fl))

class FixedPointRounding(torch.autograd.Function):

    @staticmethod
    # def forward(self, x, forward_wl=-1, forward_fl=-1, backward_wl=-1, backward_fl=-1,
    #             forward_rounding="stochastic", backward_rounding="stochastic"):
    def forward(self, x, 
                forward_number=None, backward_number=None,
                forward_rounding="stochastic", backward_rounding="stochastic"):
        for num in [forward_number, backward_number]:
            if num != None: assert isinstance(num, FixedPoint), "Number must be FixedPoint"

        if x.is_cuda:
            quant_module = quant_cuda
        else:
            quant_module = quant_cpu

        self.backward_number = backward_number
        self.backward_rounding = backward_rounding

        assert forward_rounding in ["stochastic", "nearest"]
        assert backward_rounding in ["stochastic", "nearest"]

        if forward_number == type(None): return x

        if forward_rounding=="nearest":
            out = quant_module.fixed_point_quantize_nearest(x, forward_number.wl, forward_number.fl)
        elif forward_rounding=="stochastic":
            out = quant_module.fixed_point_quantize_stochastic(x, forward_number.wl, forward_number.fl)
        return out

    @staticmethod
    def backward(self, grad_output):
        if grad_output.is_cuda:
            quant_module = quant_cuda
        else:
            quant_module = quant_cpu

        grad_input = None

        if self.needs_input_grad[0]:
            if not self.backward_number == None:
                if self.backward_rounding=="nearest":
                    grad_input = quant_module.fixed_point_quantize_nearest(grad_output,
                                                                           self.backward_number.wl,
                                                                           self.backward_number.fl)
                elif self.backward_rounding=="stochastic":
                    grad_input = quant_module.fixed_point_quantize_stochastic(grad_output,
                                                                              self.backward_number.wl,
                                                                              self.backward_number.fl)
            else:
                grad_input = grad_output

        return grad_input, None, None, None, None, None, None

class BlockRounding(torch.autograd.Function):
    @staticmethod
    def forward(self, x, 
                forward_number=None, backward_number=None, 
                forward_rounding="stochastic", backward_rounding="stochastic"):
        for num in [forward_number, backward_number]:
            if num != None: assert isinstance(num, BlockFloatingPoint), "Number must be BlockFloatingPoint"

        if x.is_cuda:
            quant_module = quant_cuda
        else:
            quant_module = quant_cpu

        self.backward_number = backward_number
        self.backward_rounding = backward_rounding

        assert forward_rounding in ["stochastic", "nearest"]
        assert backward_rounding in ["stochastic", "nearest"]

        if forward_number == None: return x
        if forward_rounding=="nearest":
            out = quant_module.block_quantize_nearest(x, forward_number.wl)
        elif forward_rounding=="stochastic":
            out = quant_module.block_quantize_stochastic(x, forward_number.wl)

        return out

    @staticmethod
    def backward(self, grad_output):
        if grad_output.is_cuda:
            quant_module = quant_cuda
        else:
            quant_module = quant_cpu

        if self.needs_input_grad[0]:
            if not self.backward_number == None:
                if self.backward_rounding=="nearest":
                    grad_input = quant_module.block_quantize_nearest(grad_output, self.backward_number.wl)
                elif self.backward_rounding=="stochastic":
                    grad_input = quant_module.block_quantize_stochastic(grad_output, self.backward_number.wl)
            else:
                grad_input = grad_output
        return grad_input, None, None, None, None

class FloatRounding(torch.autograd.Function):

    @staticmethod
    def forward(self, x, 
                forward_number=None, backward_number=None,
                forward_rounding="stochastic", backward_rounding="stochastic"
                ):
        for num in [forward_number, backward_number]:
            if num != None: assert isinstance(num, FloatingPoint), "Number must be FloatingPoint"
        if x.is_cuda:
            quant_module = quant_cuda
        else:
            quant_module = quant_cpu
        self.backward_number = backward_number
        self.backward_rounding = backward_rounding

        assert forward_rounding in ["stochastic", "nearest"]
        assert backward_rounding in ["stochastic", "nearest"]

        if forward_number == None: return x

        if forward_rounding=="nearest":
            out = quant_cuda.float_quantize_nearest(x, forward_number.man, forward_number.exp)
        elif forward_rounding=="stochastic":
            out = quant_cuda.float_quantize_stochastic(x, forward_number.man, forward_number.exp)
        return out

    @staticmethod
    def backward(self, grad_output):
        grad_input = None
        if grad_output.is_cuda:
            quant_module = quant_cuda
        else:
            quant_module = quant_cpu

        if self.needs_input_grad[0]:
            if self.backward_number == None:
                if self.backward_rounding=="nearest":
                    grad_input = quant_cuda.float_quantize_nearest(grad_output,
                                                                   self.backward_number.man,
                                                                   self.backward_number.exp)
                elif self.backward_rounding=="stochastic":
                    grad_input = quant_cuda.float_quantize_stochastic(grad_output,
                                                                      self.backward_number.man,
                                                                      self.backward_number.exp)
            else:
                grad_input = grad_output

        return grad_input, None, None, None, None, None, None

def get_module(x):
    if x.is_cuda:
        quant_module = quant_cuda
    else:
        quant_module = quant_cpu
    return quant_module

def quantizer(forward_number=None, backward_number=None,
              forward_rounding="stochastic", backward_rounding="stochastic",
              clamping_grad_zero=False):
    for rounding in [forward_rounding, backward_rounding]:
        assert rounding in ["stochastic", "nearest"], "invalid rounding type {:s}".format(rounding)
    for num in [forward_number, backward_number]:
        if num != None: assert isinstance(num, Number)

    if clamping_grad_zero==False:
        if forward_rounding=="nearest":
            if type(forward_number)==BlockFloatingPoint:
                forward_quant = lambda x, quant_module: quant_module.block_quantize_nearest(x, forward_number.wl)
            elif type(forward_number)==FixedPoint:
                forward_quant = lambda x, quant_module: quant_module.fixed_point_quantize_nearest(x, forward_number.wl, forward_number.fl)
            elif type(forward_number)==FloatingPoint:
                forward_quant = lambda x, quant_module: quant_module.float_quantize_nearest(x, forward_number.man, forward_number.exp)
        elif forward_rounding=="stochastic":
            if type(forward_number)==BlockFloatingPoint:
                forward_quant = lambda x, quant_module: quant_module.block_quantize_stochastic(x, forward_number.wl)
            elif type(forward_number)==FixedPoint:
                forward_quant = lambda x, quant_module: quant_module.fixed_point_quantize_stochastic(x, forward_number.wl, forward_number.fl)
            elif type(forward_number)==FloatingPoint:
                forward_quant = lambda x, quant_module: quant_module.float_quantize_stochastic(x, forward_number.man, forward_number.exp)
    else:
        if type(forward_number)==FixedPoint:
            if forward_rounding=="nearest":
                forward_quant = lambda x, quant_module: quant_module.fixed_point_quantize_nearest_mask(x, forward_number.wl, forward_number.fl)
            elif forward_rounding=="stochastic":
                forward_quant = lambda x, quant_module: quant_module.fixed_point_quantize_stochastic_mask(x, forward_number.wl, forward_number.fl)
        else:
            raise ValueError("zeroing clamping gradient only support fixed point.")

    if backward_rounding=="nearest":
        if type(backward_number)==BlockFloatingPoint:
            backward_quant = lambda x, quant_module: quant_module.block_quantize_nearest(x, backward_number.wl)
        elif type(backward_number)==FixedPoint:
            backward_quant = lambda x, quant_module: quant_module.fixed_point_quantize_nearest(x, backward_number.wl, backward_number.fl)
        elif type(backward_number)==FloatingPoint:
            backward_quant = lambda x, quant_module: quant_module.float_quantize_nearest(x, backward_number.man, backward_number.exp)
    elif backward_rounding=="stochastic":
        if type(backward_number)==BlockFloatingPoint:
            backward_quant = lambda x, quant_module: quant_module.block_quantize_stochastic(x, backward_number.wl)
        elif type(backward_number)==FixedPoint:
            backward_quant = lambda x, quant_module: quant_module.fixed_point_quantize_stochastic(x, backward_number.wl, backward_number.fl)
        elif type(backward_number)==FloatingPoint:
            backward_quant = lambda x, quant_module: quant_module.float_quantize_stochastic(x, backward_number.man, backward_number.exp)

    if clamping_grad_zero == False:
        class Rounding(torch.autograd.Function):
            @staticmethod
            def forward(self, x):
                if forward_number==None: return x

                quant_module = get_module(x)
                out = forward_quant(x, quant_module)

                return out

            @staticmethod
            def backward(self, grad_output):
                if self.needs_input_grad[0]:
                    if backward_number == None:
                        grad_input = grad_output
                    else:
                        quant_module = get_module(grad_output)
                        grad_input = backward_quant(grad_output, quant_module)
                else:
                    grad_input = None

                return grad_input
    else:
        class Rounding(torch.autograd.Function):
            @staticmethod
            def forward(self, x):
                if forward_number==None:
                    return x
                else:
                    quant_module = get_module(x)
                    out, mask = forward_quant(x, quant_module)
                    self.mask = mask

                return out

            @staticmethod
            def backward(self, grad_output):
                if self.needs_input_grad[0]:
                    if backward_number == None:
                        grad_input = grad_output
                    else:
                        quant_module = get_module(grad_output)
                        grad_input = backward_quant(grad_output.masked_fill_(mask, 0), quant_module)
                else:
                    grad_input = None

                return grad_input

    return Rounding.apply


def fixed_point_quantize(x,
                         forward_number=None, backward_number=None,
                         forward_rounding="stochastic", backward_rounding="stochastic"):
    return FixedPointRounding.apply(x,
                                    forward_number, backward_number,
                                    forward_rounding, backward_rounding)

def block_quantize(x,
                   forward_number=None, backward_number=None,
                   forward_rounding="stochastic", backward_rounding="stochastic"):
    return BlockRounding.apply(x,
                               forward_number, backward_number,
                               forward_rounding, backward_rounding)

def float_quantize(x,
                   forward_number=None, backward_number=None,
                   forward_rounding="stochastic", backward_rounding="stochastic"):
    return FloatRounding.apply(x,
                               forward_number, backward_number,
                               forward_rounding, backward_rounding)