from rational.torch import Rational

rat_l = Rational("leaky_relu")
rat_s = Rational("sigmoid")
rat_i = Rational("identity")

rat_l.show()

print(Rational.list)
# [Rational Activation Function A) of degrees (5, 4) running on cuda 0x7f778678b700
# , Rational Activation Function A) of degrees (5, 4) running on cuda 0x7f778678b1c0
# , Rational Activation Function A) of degrees (5, 4) running on cuda 0x7f77851fb5b0
# ]

Rational.show_all()

print(rat_l.snapshot_list)
rat_l.capture(name="Leaky init :)")
print(rat_l.snapshot_list)
# []
# [Snapshot (Leaky init :))]

import torch
rat_l.snapshot_list[0].show(other_func=[torch.sin, torch.tanh])

Rational.show_all(other_func=[torch.sin, torch.tanh])
