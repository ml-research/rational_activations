import json
import numpy as np
from utils import fit_pau_to_base_function, plot_result, append_to_config_file, parser
from paus_py import PAU_version_A, PAU_version_B, PAU_version_C
import torch.nn.functional as F
import torch


def main():
    args = parser.parse_args()


    # To be changed by the function you want to approximate
    approx_name = "leaky_relu"
    def function_to_approx(x):
        # return np.heaviside(x, 0)
        x = torch.tensor(x)
        return F.leaky_relu(x)

    degrees = (args.nd, args.dd)

    step = (args.ub - args.lb) / 1000000
    x = np.arange(args.lb, args.ub, step)
    if args.version == 'A':
        pau = PAU_version_A
    elif args.version == 'B':
        pau = PAU_version_B
    elif args.version == 'C':
        pau = PAU_version_C
    elif args.version == 'D':
        pau = PAU_version_B

    w_params, d_params = fit_pau_to_base_function(pau, function_to_approx, x,
                                                  degrees=degrees, version=args.version)
    print(f"Found coeffient :\nP: {w_params}\nQ: {d_params}")
    if args.plot:
        plot_result(x, pau(x, w_params, d_params), function_to_approx(x), approx_name)

    append_to_config_file(args, approx_name, w_params, d_params)

if __name__ == '__main__':
    main()
