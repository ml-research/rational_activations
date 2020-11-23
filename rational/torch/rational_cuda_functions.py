import torch
from rational.cuda import *


class Rational_CUDA_A_F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight_numerator, weight_denominator, training):
        ctx.save_for_backward(input, weight_numerator, weight_denominator)
        x = forward_A_5_4(input, weight_numerator, weight_denominator)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, w_numerator, w_denominator = ctx.saved_tensors
        d_x, d_weight_numerator, d_weight_denominator = backward_A_5_4(grad_output, x, w_numerator, w_denominator)
        return d_x, d_weight_numerator, d_weight_denominator, None


class Rational_CUDA_B_F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight_numerator, weight_denominator, training):
        ctx.save_for_backward(input, weight_numerator, weight_denominator)
        x = forward_B_5_4(input, weight_numerator, weight_denominator)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, w_numerator, w_denominator = ctx.saved_tensors
        d_x, d_weight_numerator, d_weight_denominator = backward_B_5_4(grad_output, x, w_numerator, w_denominator)
        return d_x, d_weight_numerator, d_weight_denominator, None


class Rational_CUDA_C_F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight_numerator, weight_denominator, training):
        ctx.save_for_backward(input, weight_numerator, weight_denominator)
        x = forward_C_5_4(input, weight_numerator, weight_denominator)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, w_numerator, w_denominator = ctx.saved_tensors
        d_x, d_weight_numerator, d_weight_denominator = backward_C_5_4(grad_output, x, w_numerator, w_denominator)
        return d_x, d_weight_numerator, d_weight_denominator, None


class Rational_CUDA_D_F(torch.autograd.Function):
    cnt = 0

    @staticmethod
    def forward(ctx, input, w_numerator, w_denominator, training):
        local_cnt = Rational_CUDA_D_F.cnt

        ctx.save_for_backward(input, w_numerator, w_denominator, torch.tensor(local_cnt, dtype=torch.long))

        Rational_CUDA_D_F.cnt += 1
        x = forward_D_5_4(training, local_cnt, input, w_numerator, w_denominator)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # if not grad_output.is_contiguous():  # TODO this check is necessary if efficientnet is used
        #    grad_output = grad_output.contiguous()

        x, weight_numerator, weight_denominator, local_cnt = ctx.saved_tensors
        d_x, d_weight_numerator, d_weight_denominator = backward_D_5_4(True,
                                                                       local_cnt,
                                                                       grad_output,
                                                                       x,
                                                                       weight_numerator,
                                                                       weight_denominator)

        return d_x, d_weight_numerator, d_weight_denominator, None
