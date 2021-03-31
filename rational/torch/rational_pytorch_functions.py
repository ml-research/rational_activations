import torch

def Rational_PYTORCH_A_F(x, weight_numerator, weight_denominator, training):
    # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X^n /
    #               1 + | b_1 * X | + | b_2 * X^2| + ... + | b_m * X ^m|

    z = x.view(-1)
    len_num, len_deno = len(weight_numerator), len(weight_denominator)
    xps = torch.vander(z, max(len_num, len_deno), increasing=True)
    numerator = xps.mul(weight_numerator).sum(1)
    expanded_dw = torch.cat([torch.tensor([1.]), weight_denominator, \
                             torch.zeros(len_num - len_deno - 1)])
    denominator = xps.mul(expanded_dw).abs().sum(1)
    return numerator.div(denominator).view(x.shape)


def Rational_PYTORCH_B_F(x, weight_numerator, weight_denominator, training):
    # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X^n /
    #               1 + |b_1 * X + b_1 * X^2 + ... + b_m * X^m|
    z = x.view(-1)
    len_num, len_deno = len(weight_numerator), len(weight_denominator)
    xps = torch.vander(z, max(len_num, len_deno), increasing=True)
    numerator = xps.mul(weight_numerator).sum(1)
    denominator = xps[:, 1:len_deno+1].mul(weight_denominator).sum(1).abs()
    return numerator.div(1 + denominator).view(x.shape)


def Rational_PYTORCH_C_F(x, weight_numerator, weight_denominator, training):
    # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X^n /
    #               eps + |b_0 + b1 * X + b_2 * X^2 + ... + b_m*X^m|
    z = x.view(-1)
    len_num, len_deno = len(weight_numerator), len(weight_denominator)
    xps = torch.vander(z, max(len_num, len_deno), increasing=True)
    numerator = xps.mul(weight_numerator).sum(1)
    denominator = xps[:, :len_deno].mul(weight_denominator).sum(1).abs()
    return numerator.div(0.1 + denominator).view(x.shape)


def Rational_PYTORCH_D_F(x, weight_numerator, weight_denominator, training, random_deviation=0.1):
    # P(X)/Q(X) = noised(a_0) + noised(a_1) * X +noised(a_2) * X^2 + ... + noised(a_n) * X^n /
    #     #                1 + |noised(b_1) * X + noised(b_2) * X^2 + ... + noised(b_m)*X^m|
    #     # Noised parameters have uniform noise to be in range [(1-random_deviation)*parameter,(1+random_deviation)*parameter].
    if not training:
        # do not add noise
        return Rational_PYTORCH_B_F(x, weight_numerator, weight_denominator, training)
    z = x.view(-1)
    len_num, len_deno = len(weight_numerator), len(weight_denominator)
    xps = torch.vander(z, max(len_num, len_deno), increasing=True)
    numerator = xps.mul(weight_numerator.mul(
        torch.FloatTensor(len_num).uniform_(1-random_deviation,
                                            1+random_deviation))
                       ).sum(1)
    denominator = xps[:, 1:len_deno+1].mul(weight_denominator).sum(1).abs()
    return numerator.div(1 + denominator).view(x.shape)
