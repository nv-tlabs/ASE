import torch
from rl_games.common import datasets

class AMPDataset(datasets.PPODataset):
    def __init__(self, batch_size, minibatch_size, is_discrete, is_rnn, device, seq_len):
        super().__init__(batch_size, minibatch_size, is_discrete, is_rnn, device, seq_len)
        self._idx_buf = torch.randperm(batch_size)
        return
    
    def update_mu_sigma(self, mu, sigma):	  
        raise NotImplementedError()
        return

    def _get_item(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        sample_idx = self._idx_buf[start:end]

        input_dict = {}
        for k,v in self.values_dict.items():
            if k not in self.special_names and v is not None:
                input_dict[k] = v[sample_idx]
                
        if (end >= self.batch_size):
            self._shuffle_idx_buf()

        return input_dict

    def _shuffle_idx_buf(self):
        self._idx_buf[:] = torch.randperm(self.batch_size)
        return