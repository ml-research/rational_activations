import numpy as np


class Histogram():
    """
    Input Histograms, used to retrieve the input of Rational Activations
    """
    def __init__(self, bin_size=0.1, random_select=False):
        self.bins = np.array([])
        self.weights = np.array([], dtype=np.int)
        self.bin_size = bin_size
        self._empty = True
        self._rd = int(np.log10(1./bin_size).item())

    def fill_n(self, input):
        self._update_hist(input.numpy())

    def _update_hist(self, new_input):
        range_ext = np.around(new_input.min() - self.bin_size / 2, 1), \
                    np.around(new_input.max() + self.bin_size / 2, 1)
        bins_array = np.arange(range_ext[0], range_ext[1] + self.bin_size,
                               self.bin_size)
        weights, bins = np.histogram(new_input, bins_array)
        if self._empty:
            self.weights, self.bins = weights, bins[:-1]
            self._empty = False
        else: #  update the hist
            self.weights, self.bins = concat_hists(self.weights, self.bins,
                                                   weights, bins[:-1],
                                                   self.bin_size, self._rd)

    def __repr__(self):
        if self._empty:
            return "Empty Histogram"
        return f"Histogram on range {self.bins[0]}, {self.bins[-1]}, of " + \
               f"bin_size {self.bin_size}, with {self.weights.sum()} total" + \
               f"elements"

    def normalize(self, numpy=True):
            return self.weights / self.weights.sum(), self.bins

    def _from_physt(self, phystogram):
        if (phystogram.bin_sizes == phystogram.bin_sizes[0]).all():
            self.bin_size = phystogram.bin_sizes[0]
        self.bins = np.array(phystogram.bin_left_edges)
        self.weights = np.array(phystogram.frequencies)
        return self


def concat_hists(weights1, bins1, weights2, bins2, bin_size, rd):
    min1, max1 = np.around(bins1[0], rd), np.around(bins1[-1], rd)
    min2, max2 = np.around(bins2[0], rd), np.around(bins2[-1], rd)
    mini, maxi = min(min1, min2), max(max1, max2)
    new_bins = np.arange(mini, maxi + bin_size*0.9, bin_size)  # * 0.9 to avoid unexpected random inclusion of last element
    if min1 - mini != 0 and maxi - max1 != 0:
        ext1 = np.pad(weights1, (np.int(np.around((min1 - mini) / bin_size)),
                                 np.int(np.around((maxi - max1) / bin_size))),
                      'constant', constant_values=0)
    elif min1 - mini != 0:
        ext1 = np.pad(weights1, (np.int(np.around((min1 - mini) / bin_size)),
                                 0), 'constant', constant_values=0)
    elif maxi - max1 != 0:
        ext1 = np.pad(weights1, (0,
                                 np.int(np.around((maxi - max1) / bin_size))),
                      'constant', constant_values=0)
    else:
        ext1 = weights1
    if min2 - mini != 0 and maxi - max2 != 0:
        ext2 = np.pad(weights2, (np.int(np.around((min2 - mini) / bin_size)),
                                 np.int(np.around((maxi - max2) / bin_size))),
                      'constant', constant_values=0)
    elif min2 - mini != 0:
        ext2 = np.pad(weights2, (np.int(np.around((min2 - mini) / bin_size)),
                                 0), 'constant', constant_values=0)
    elif maxi - max2 != 0:
        ext2 = np.pad(weights2, (0,
                                 np.int(np.around((maxi - max2) / bin_size))),
                      'constant', constant_values=0)
    else:
        ext2 = weights2
    new_ext = ext1 + ext2
    return new_ext, new_bins
