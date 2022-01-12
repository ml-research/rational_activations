import torch
from termcolor import colored
from rationals import Rational
import matplotlib.pyplot as plt


def recover_from_load(path):
    loaded_model = torch.load(path)
    rationals_rec = []
    last_nume_base_name = None
    for param_name, tensor_value in loaded_model.items():
        if param_name[-9:] == "numerator":
            last_nume_base_name = ".".join(param_name.split(".")[:-1])
            rat_rec = RationalRecovery(last_nume_base_name, tensor_value)
        elif param_name[-11:] == "denominator":
            # asserts numer and deno from same rat
            deno_base_name = ".".join(param_name.split(".")[:-1])
            if deno_base_name != last_nume_base_name:
                msg = f"Found numerator with name {last_nume_base_name}\n"
                msg += f"and next denominator with name {deno_base_name}"
                print(colored(msg, "red"))
                exit(1)
            rat_rec.denominator_tensor = tensor_value
            if rat_rec not in rationals_rec:
                rationals_rec.append(rat_rec)
                print(f"Added {rat_rec}")
            else:
                print(f"Found {rat_rec.base_name}, a shared rational already retrieved.\n")
    return rationals_rec


class RationalRecovery():
    def __init__(self, base_name, numerator=None, denominator=None):
        self.base_name = base_name
        self.numerator_tensor = numerator
        self.denominator_tensor = denominator

    def __repr__(self):
        string = f"Rational Recovery from {self.base_name}:\n"
        string += f"\t numerator   : {self.numerator_tensor}\n"
        string += f"\t denominator : {self.denominator_tensor}"
        return string

    def __eq__(self, o):
        if not isinstance(o, RationalRecovery):
            return False
        return all(self.numerator_tensor == o.numerator_tensor) and \
                all(self.denominator_tensor == o.denominator_tensor)


if __name__ == '__main__':
    recovs = recover_from_load('pytorch_adapter.bin')
    for rat_recov in recovs:
        rat = Rational()
        rat.func_name = rat_recov.base_name
        rat.numerator = torch.nn.Parameter(rat_recov.numerator_tensor)
        rat.denominator = torch.nn.Parameter(rat_recov.denominator_tensor)
    for rat in Rational.list:
        rat.func_name = rat.func_name.replace(".output.adapters.pfeiffer_rational_one", "")
        rat.func_name = rat.func_name.replace("non_linearity.f", "")
        rat.func_name = rat.func_name.replace("bert.encoder.", "")
        rat.func_name = rat.func_name.replace(".", " ")
    fig = Rational.show_all(display=False)
    fig.suptitle("Pfeiffer Rational one adapter on COLA", y=1., fontsize=12)
    plt.show()
    # fig = plt.gcf()
