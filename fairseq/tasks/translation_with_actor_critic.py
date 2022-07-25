
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import II

from dataclasses import dataclass, field
from fairseq.tasks import FairseqTask
from . import register_task
from .translation import TranslationTask, TranslationConfig
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq.critic_seq_gen import CriticSequenceGenerator
from fairseq.checkpoint_utils import load_checkpoint_to_cpu, convert_namespace_to_omegaconf

class ValueEstimator(nn.Module):

    def __init__(self, input_size, output_size):
        super(ValueEstimator, self).__init__()
        self.ff1 = nn.Linear(input_size, input_size * 4)
        self.ff2 = nn.Linear(input_size * 4, input_size * 4)
        self.ff3 = nn.Linear(input_size * 4, output_size)
        
    def forward(self, input):

        out = self.ff1(input)
        out = F.relu(out)
        out = self.ff2(out)
        out = F.relu(out)
        out = self.ff3(out)
        return out


@dataclass
class TranslationWithActorCriticConfig(TranslationConfig):
    base_model_path: str = field(
        default="", metadata={"help": "path to model for sequence generation"}
    )
    use_critic_generator: bool = field(default=False)
    critic_mix_ratio: float = field(default=0.5)
    decoder_embed_d: int = field(default=512)
    subtract_max: bool = field(default=False)


@register_task("translation_with_actor_critic", dataclass=TranslationWithActorCriticConfig)
class TranslationWithActorCritic(TranslationTask):

    def __init__(self, cfg, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.base_model = None
        self.cfg = cfg

    def build_model(self, cfg):
        if self.cfg.use_critic_generator:
            state = load_checkpoint_to_cpu(self.cfg.base_model_path)
            if "args" in state and state["args"] is not None:
                base_cfg = convert_namespace_to_omegaconf(state["args"])
            elif "cfg" in state and state["cfg"] is not None:
                base_cfg = state["cfg"]
            if self.cfg.use_critic_generator:
                self.base_model = FairseqTask.build_model(self, base_cfg.model)
                self.base_model.load_state_dict(state["model"], model_cfg=base_cfg.model)
                self.base_model.cuda()
        model = super().build_model(cfg)
        return model

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample, update_num)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None, prefix_allowed_tokens_fn=None,
    ):
        if self.cfg.use_critic_generator and not getattr(args, "sampling", False):
            if self.base_model is not None:
                print("USING BASE MODEL")
                base_models = [self.base_model.cuda()]
            else:
                base_models = models
            if extra_gen_cls_kwargs is None:
                extra_gen_cls_kwargs = {}
            extra_gen_cls_kwargs["vf"] = models
            extra_gen_cls_kwargs["critic_mix_ratio"] = self.cfg.critic_mix_ratio
            extra_gen_cls_kwargs["subtract_max"] = self.cfg.subtract_max
            return super().build_generator(base_models, args, CriticSequenceGenerator, extra_gen_cls_kwargs, prefix_allowed_tokens_fn)
        else:
            return super().build_generator(models, args, seq_gen_cls, extra_gen_cls_kwargs, prefix_allowed_tokens_fn)
