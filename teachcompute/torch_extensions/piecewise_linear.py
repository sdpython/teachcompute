import torch
from torch.autograd import Function
from .piecewise_linear_c import (
    piecewise_linear_forward,
    piecewise_linear_backward,
    piecewise_linear_forward_better,
    piecewise_linear_backward_better,
)


class PiecewiseLinearFunction(Function):
    """
    Implements a function similar to a piecewise linear function.
    It multiplies by different coefficient on negative and positive
    number. It takes a tensor of shape (N, 1).
    """

    @staticmethod
    def forward(ctx, x, alpha_neg, alpha_pos):
        sign = (x >= 0).to(torch.float32)
        weight = sign * alpha_pos + (-sign + 1) * alpha_neg
        ctx.save_for_backward(x, sign, weight)
        output = x * weight
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, sign, weight = ctx.saved_tensors
        grad_x = weight
        grad_alpha_neg = (x * grad_output * (-sign + 1)).sum(dim=0, keepdim=True)
        grad_alpha_pos = (x * grad_output * sign).sum(dim=0, keepdim=True)
        return grad_x, grad_alpha_neg, grad_alpha_pos


class PiecewiseLinearFunctionC(Function):
    """
    Same function as :class:`PiecewiseLinearFunction
    <teachcompute.torch_extensions.piecewise_linear.PiecewiseLinearFunction>`
    but the implementation of forward and backward functions
    are done in C. See :func:`piecewise_linear_forward
    <teachcompute.torch_extensions.piecewise_linear_c.piecewise_linear_forward>` and
    :func:`piecewise_linear_backward
    <teachcompute.torch_extensions.piecewise_linear_c.piecewise_linear_backward>`.
    It follows the tutorial :epkg:`Custom C++ and CUDA Extensions`.
    """

    @staticmethod
    def forward(ctx, x, alpha_neg, alpha_pos):
        outputs = piecewise_linear_forward(x, alpha_neg, alpha_pos)
        ctx.save_for_backward(*outputs[1:])
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output):
        x, sign, weight = ctx.saved_tensors
        weight, grad_alpha_neg, grad_alpha_pos = piecewise_linear_backward(
            grad_output, x, sign, weight
        )
        return weight, grad_alpha_neg, grad_alpha_pos


class PiecewiseLinearFunctionCBetter(Function):
    """
    Same function as :class:`PiecewiseLinearFunctionC
    <teachcompute.torch_extensions.piecewise_linear.PiecewiseLinearFunctionC>`,
    the implementation of forward and backward are is reducing the
    memory allocations.
    """

    @staticmethod
    def forward(ctx, x, alpha_neg, alpha_pos):
        outputs = piecewise_linear_forward_better(x, alpha_neg, alpha_pos)
        ctx.save_for_backward(*outputs[1:])
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output):
        x, sign, weight = ctx.saved_tensors
        weight, grad_alpha_neg, grad_alpha_pos = piecewise_linear_backward_better(
            grad_output, x, sign, weight
        )
        return weight, grad_alpha_neg, grad_alpha_pos
