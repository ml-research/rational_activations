import torch


def get_xps(weight_denominator, weight_numerator, z):
    xps = list()
    xps.append(z)
    for _ in range(max(len(weight_numerator), len(weight_denominator))):
        xps.append(xps[-1].mul(z))
    xps.insert(0, torch.ones_like(z))
    return xps


def PAU_PYTORCH_A_F(x, weight_numerator, weight_denominator, training):
    # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X ^ n /
    #               1 + | b_0 * X | + | b_1 * X | ^ 2 + ... + | b_i * X | ^ {i + 1}

    z = x.view(-1)

    xps = get_xps(weight_denominator, weight_numerator, z)

    numerator = torch.FloatTensor([0])
    for i, w_n in enumerate(weight_numerator):
        numerator = numerator + w_n.mul(xps[i])

    denominator = torch.FloatTensor([1.0])
    for j, w_d in enumerate(weight_denominator):
        denominator = denominator + w_d.mul(xps[j + 1]).abs()

    return numerator.div(denominator).view(x.shape)


def PAU_PYTORCH_B_F(x, weight_numerator, weight_denominator, training):
    # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X ^ n /
    #               1 + |b_0*X + b_1*X^2 + ... + b_{n-1}*X^n|

    z = x.view(-1)

    xps = get_xps(weight_denominator, weight_numerator, z)

    numerator = torch.FloatTensor([0])
    for i, w_n in enumerate(weight_numerator):
        numerator = numerator + w_n.mul(xps[i])

    denominator = torch.FloatTensor([0])
    for j, w_d in enumerate(weight_denominator):
        denominator = denominator + w_d.mul(xps[j + 1])

    return numerator.div((1.0 + denominator.abs())).view(x.shape)

def PAU_PYTORCH_C_F(x, weight_numerator, weight_denominator, training):
    # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X ^ n /
    #               eps + |b_1*X + b_1*X^2 + ... + b_{n-1}*X^n|

    z = x.view(-1)

    xps = get_xps(weight_denominator, weight_numerator, z)

    numerator = torch.FloatTensor([0])
    for i, w_n in enumerate(weight_numerator):
        numerator = numerator + w_n.mul(xps[i])

    denominator = torch.FloatTensor([0])
    for j, w_d in enumerate(weight_denominator):
        denominator = denominator + w_d.mul(xps[j])

    return numerator.div((0.1 + denominator.abs())).view(x.shape)


def PAU_PYTORCH_D_F(x, weight_numerator, weight_denominator, training, random_deviation=0.1):
    # P(X)/Q(X) = noised(a_0) + noised(a_1)*X +noised(a_2)*X^2 + ... + noised(a_n)*X^n /
    #                1 + |noised(b_0)*X + noised(b_1)*X^2 + ... + noised(b_{n-1})*X^n|
    # Noised parameters have uniform noise to be in range [(1-random_deviation)*parameter,(1+random_deviation)*parameter].

    if not training:
        # do not add noise
        return PAU_PYTORCH_B_F(x, weight_numerator, weight_denominator, training)

    z = x.view(-1)

    xps = get_xps(weight_denominator, weight_numerator, z)

    numerator = torch.FloatTensor([0])
    for i, w_n in enumerate(weight_numerator):
        w_n_noised = w_n.mul(torch.FloatTensor(
            z.shape).uniform_(1-random_deviation, 1+random_deviation))
        numerator = numerator + w_n_noised.mul(xps[i])

    denominator = torch.FloatTensor([0])
    for j, w_d in enumerate(weight_denominator):
        w_d_noised = w_d.mul(torch.FloatTensor(
            z.shape).uniform_(1-random_deviation, 1+random_deviation))
        denominator = denominator + w_d_noised.mul(xps[j + 1])

    return numerator.div((1 + denominator.abs())).view(x.shape)
