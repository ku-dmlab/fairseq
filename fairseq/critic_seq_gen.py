
import copy
import torch
import torch.nn as nn
from fairseq.sequence_generator import SequenceGenerator


class CriticSequenceGenerator(SequenceGenerator):
    def __init__(self, *args, **kwargs):
        vf = kwargs.pop("vf")
        critic_mix_ratio = kwargs.pop("critic_mix_ratio")
        super().__init__(*args, **kwargs)
        self.vf = vf
        self.critic_mix_ratio = critic_mix_ratio

    def _get_lprobs(self, token, encoder_outs, incremental_states):
        cur_states = copy.deepcopy(incremental_states)
        lprobs, _ = super()._get_lprobs(token, encoder_outs, incremental_states)

        out_features, _ = self.model.models[0].decoder.forward(
            token,
            encoder_out=encoder_outs[0],
            incremental_state=cur_states[0],
            features_only=True
        )

        values = self.vf(out_features)[:, -1]
        lprobs += self.critic_mix_ratio * nn.functional.log_softmax(values, dim=-1)
        return lprobs, None
