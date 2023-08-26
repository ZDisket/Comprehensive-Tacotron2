import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from numba import jit


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        b, c, h = y.size()
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps**2))
        return loss/(c*b*h)


class BinLoss(torch.nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()

    @property
    def input_types(self):
        return {
            "hard_attention": NeuralType(('B', 'S', 'T_spec', 'T_text'), ProbsType()),
            "soft_attention": NeuralType(('B', 'S', 'T_spec', 'T_text'), ProbsType()),
        }

    @property
    def output_types(self):
        return {
            "bin_loss": NeuralType(elements_type=LossType()),
        }


    def forward(self, hard_attention, soft_attention):
        log_sum = torch.log(torch.clamp(soft_attention[hard_attention == 1], min=1e-12)).sum()
        return -log_sum / hard_attention.sum()


class ForwardSumLoss(torch.nn.modules.loss._Loss):
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.ctc_loss = torch.nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    @property
    def input_types(self):
        return {
            "attn_logprob": NeuralType(('B', 'S', 'T_spec', 'T_text'), LogprobsType()),
            "in_lens": NeuralType(tuple('B'), LengthsType()),
            "out_lens": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "forward_sum_loss": NeuralType(elements_type=LossType()),
        }

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(input=attn_logprob, pad=(1, 0), value=self.blank_logprob)

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[: query_lens[bid], :, : key_lens[bid] + 1]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            total_loss += loss

        total_loss /= attn_logprob.shape[0]
        return total_loss

class Tacotron2Loss(nn.Module):
    """ Tacotron2 Loss """

    def __init__(self, preprocess_config, model_config, train_config):
        super(Tacotron2Loss, self).__init__()
        self.n_frames_per_step = model_config["decoder"]["n_frames_per_step"]
        self.use_cac = train_config["optimizer"]["use_cac"]
        self.use_guided_attn_loss = train_config["optimizer"]["guided_attn"] and not self.use_cac # CAC turns off guided attention loss
        self.cac_hard_mul = train_config["optimizer"]["cac_loss_hard_mul"]
        self.bin_loss_start_epoch = train_config["optimizer"]["bin_loss_start_epoch"]
        self.bin_loss_warmup_epochs = train_config["optimizer"]["bin_loss_warmup_epochs"]

    

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        if self.use_cac:
            self.forward_sum = ForwardSumLoss()
            self.bin_loss = BinLoss()
            self.charb_loss = CharbonnierLoss()
            print("Using Convolutional Attention Consistency")
        
        if self.use_guided_attn_loss:
            self.guided_attn_loss = GuidedAttentionLoss(
                sigma=train_config["optimizer"]["guided_sigma"],
                alpha=train_config["optimizer"]["guided_lambda"],
            )
            print("Using diagonal guided attention loss")

    def forward(self, inputs, predictions, epoch):
        mel_target, input_lengths, output_lengths, r_len_pad, gate_target \
                                = inputs[6], inputs[4], inputs[7], inputs[9], inputs[10]
        mel_out, mel_out_postnet, gate_out, alignments, attn_logprob, attn_hard, conv_att_soft = predictions
        

        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        gate_out = gate_out.view(-1, 1)
        mel_loss = self.mse_loss(mel_out, mel_target) + \
            self.mse_loss(mel_out_postnet, mel_target)
        gate_loss = self.bce_loss(gate_out, gate_target)

        total_loss = mel_loss + gate_loss

        if self.use_cac:            
            al_forward_sum = self.forward_sum(attn_logprob=attn_logprob, in_lens=input_lengths, out_lens=output_lengths)
            soft_cac_loss = self.charb_loss(alignments, # alignments =  [b, x, y]
                                     conv_att_soft.squeeze()) # conv_att_soft = [b, 1, x, y] -> [b, x, y]
            
            hard_cac_loss = self.charb_loss(alignments, # alignments =  [b, x, y]
                                     attn_hard.squeeze()) # attn_hard = [b, 1, x, y] -> [b, x, y]
            
            cac_loss = soft_cac_loss + hard_cac_loss * self.cac_hard_mul
            total_attn_loss = al_forward_sum + cac_loss
            
            if epoch > self.bin_loss_start_epoch:
                bin_loss_scale = min((epoch - self.bin_loss_start_epoch) / self.bin_loss_warmup_epochs, 1.0)
                al_match_loss = self.bin_loss(hard_attention=attn_hard, soft_attention=conv_att_soft) * bin_loss_scale
                total_attn_loss += al_match_loss
    
            total_loss += total_attn_loss
            
            return total_loss, mel_loss, gate_loss, al_forward_sum, total_attn_loss 
        elif self.use_guided_attn_loss:
            attn_loss = self.guided_attn_loss(alignments, input_lengths, \
                                (output_lengths + r_len_pad)//self.n_frames_per_step)
            total_loss += attn_loss
            return total_loss, mel_loss, gate_loss, torch.tensor([0.], device=mel_target.device), attn_loss
        else:
            return total_loss, mel_loss, gate_loss, torch.tensor([0.], device=mel_target.device), torch.tensor([0.], device=mel_target.device)
            
    


class GuidedAttentionLoss(nn.Module):
    """Guided attention loss function module.
    See https://github.com/espnet/espnet/blob/e962a3c609ad535cd7fb9649f9f9e9e0a2a27291/espnet/nets/pytorch_backend/e2e_tts_tacotron2.py#L25
    This module calculates the guided attention loss described
    in `Efficiently Trainable Text-to-Speech System Based
    on Deep Convolutional Networks with Guided Attention`_,
    which forces the attention to be diagonal.
    .. _`Efficiently Trainable Text-to-Speech System
        Based on Deep Convolutional Networks with Guided Attention`:
        https://arxiv.org/abs/1710.08969
    """

    def __init__(self, sigma=0.4, alpha=1.0, reset_always=True):
        """Initialize guided attention loss module.
        Args:
            sigma (float, optional): Standard deviation to control
                how close attention to a diagonal.
            alpha (float, optional): Scaling coefficient (lambda).
            reset_always (bool, optional): Whether to always reset masks.
        """
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.reset_always = reset_always
        self.guided_attn_masks = None
        self.masks = None

    def _reset_masks(self):
        self.guided_attn_masks = None
        self.masks = None

    def forward(self, att_ws, ilens, olens):
        """Calculate forward propagation.
        Args:
            att_ws (Tensor): Batch of attention weights (B, T_max_out, T_max_in).
            ilens (LongTensor): Batch of input lenghts (B,).
            olens (LongTensor): Batch of output lenghts (B,).
        Returns:
            Tensor: Guided attention loss value.
        """
        if self.guided_attn_masks is None:
            self.guided_attn_masks = self._make_guided_attention_masks(ilens, olens).to(
                att_ws.device
            )
        if self.masks is None:
            self.masks = self._make_masks(ilens, olens).to(att_ws.device)
        losses = self.guided_attn_masks * att_ws
        loss = torch.mean(losses.masked_select(self.masks))
        if self.reset_always:
            self._reset_masks()
        return self.alpha * loss

    def _make_guided_attention_masks(self, ilens, olens):
        n_batches = len(ilens)
        max_ilen = max(ilens)
        max_olen = max(olens)
        guided_attn_masks = torch.zeros((n_batches, max_olen, max_ilen))
        for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(
                ilen, olen, self.sigma
            )
        return guided_attn_masks

    @staticmethod
    def _make_guided_attention_mask(ilen, olen, sigma):
        """Make guided attention mask.
        Examples:
            >>> guided_attn_mask =_make_guided_attention(5, 5, 0.4)
            >>> guided_attn_mask.shape
            torch.Size([5, 5])
            >>> guided_attn_mask
            tensor([[0.0000, 0.1175, 0.3935, 0.6753, 0.8647],
                    [0.1175, 0.0000, 0.1175, 0.3935, 0.6753],
                    [0.3935, 0.1175, 0.0000, 0.1175, 0.3935],
                    [0.6753, 0.3935, 0.1175, 0.0000, 0.1175],
                    [0.8647, 0.6753, 0.3935, 0.1175, 0.0000]])
            >>> guided_attn_mask =_make_guided_attention(3, 6, 0.4)
            >>> guided_attn_mask.shape
            torch.Size([6, 3])
            >>> guided_attn_mask
            tensor([[0.0000, 0.2934, 0.7506],
                    [0.0831, 0.0831, 0.5422],
                    [0.2934, 0.0000, 0.2934],
                    [0.5422, 0.0831, 0.0831],
                    [0.7506, 0.2934, 0.0000],
                    [0.8858, 0.5422, 0.0831]])
        """
        grid_x, grid_y = torch.meshgrid(torch.arange(olen), torch.arange(ilen))
        grid_x, grid_y = grid_x.float().to(olen.device), grid_y.float().to(ilen.device)
        return 1.0 - torch.exp(
            -((grid_y / ilen - grid_x / olen) ** 2) / (2 * (sigma ** 2))
        )

    def _make_masks(self, ilens, olens):
        """Make masks indicating non-padded part.
        Args:
            ilens (LongTensor or List): Batch of lengths (B,).
            olens (LongTensor or List): Batch of lengths (B,).
        Returns:
            Tensor: Mask tensor indicating non-padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)
        Examples:
            >>> ilens, olens = [5, 2], [8, 5]
            >>> _make_mask(ilens, olens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]],
                    [[1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]], dtype=torch.uint8)
        """
        in_masks = self.make_non_pad_mask(ilens)  # (B, T_in)
        out_masks = self.make_non_pad_mask(olens)  # (B, T_out)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)  # (B, T_out, T_in)


    def make_non_pad_mask(self, lengths, xs=None, length_dim=-1):
        return ~self.make_pad_mask(lengths, xs, length_dim)


    def make_pad_mask(self, lengths, xs=None, length_dim=-1):
        if length_dim == 0:
            raise ValueError("length_dim cannot be 0: {}".format(length_dim))

        if not isinstance(lengths, list):
            lengths = lengths.tolist()
        bs = int(len(lengths))
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)

        seq_range = torch.arange(0, maxlen, dtype=torch.int64)
        seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
        seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
        mask = seq_range_expand >= seq_length_expand

        if xs is not None:
            assert xs.size(0) == bs, (xs.size(0), bs)

            if length_dim < 0:
                length_dim = xs.dim() + length_dim
            # ind = (:, None, ..., None, :, , None, ..., None)
            ind = tuple(
                slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
            )
            mask = mask[ind].expand_as(xs).to(xs.device)
        return mask
