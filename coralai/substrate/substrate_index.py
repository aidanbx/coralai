import numpy as np

class SubstrateIndex:
    def __init__(self, index_tree):
        self.index_tree = index_tree

    def _get_tuple_indices(self, key_tuple):
        chid = key_tuple[0]
        subchid = key_tuple[1]
        if isinstance(subchid, list):
            indices = []
            for subchid_single in key_tuple[1]:
                indices += self.index_tree[chid]["subchannels"][subchid_single][
                    "indices"
                ]
        else:
            indices = self.index_tree[chid]["subchannels"][subchid]["indices"]
        return indices


    def __getitem__(self, key):
        if isinstance(key, tuple):
            return np.array(self._get_tuple_indices(key))

        elif isinstance(key, list):
            indices = []
            for chid in key:
                if isinstance(chid, tuple):
                    indices += self._get_tuple_indices(chid)
                else:
                    indices += self.index_tree[chid]["indices"]
            return np.array(indices)
        else:
            return np.array(self.index_tree[key]["indices"])

    def __setitem__(self, key, value):
        raise ValueError(
            f"World: World indices are read-only. Cannot set index {key} to {value} - get/set to the world iteself"
        )
        