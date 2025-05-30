from tqdm.auto import tqdm
from collections import OrderedDict
from numbers import Number
import numpy as np


class Coloredtqdm(tqdm):
    def set_postfix(self, ordered_dict=None, refresh=True, round=4, **kwargs):
        postfix = OrderedDict([] if ordered_dict is None else ordered_dict)

        for key in sorted(kwargs.keys()):
            postfix[key] = kwargs[key]

        for key in postfix.keys():
            if isinstance(postfix[key], Number):
                postfix[key] = self.format_num_to_k(np.round(postfix[key], round), k=round + 1)
            if isinstance(postfix[key], str):
                postfix[key] = str(postfix[key])
            if len(postfix[key]) != round:
                postfix[key] += (round - len(postfix[key])) * " "

            self.postfix = ""

        self.postfix += ", ".join(key + "=" + postfix[key] for key in postfix.keys())

        if refresh:
            self.refresh()

    def format_num_to_k(self, seq, k=4):
        seq = str(seq)
        length = len(seq)
        out = seq + " " * (k - length) if length < k else seq
        return out if length < k else seq[:k]
