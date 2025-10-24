import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- DKD util -------------------------------------------------
def _mask_by_class(logits: torch.Tensor, target: torch.Tensor, invert: bool = False):

    if target.dim() == 2:                # soft label → hard
        target = target.argmax(dim=1)
    target = target.to(torch.long).view(-1, 1)   # ★ 转 int64

    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(1, target, True)               # index ok
    return ~mask if invert else mask

def _gather_two_probs(prob, m_y, m_not):
    p_y   = (prob * m_y).sum(1, keepdim=True)
    p_not = (prob * m_not).sum(1, keepdim=True)
    return torch.cat([p_y, p_not], dim=1).clamp_min(1e-12)


def dkd_loss(student_logits: torch.Tensor,
             teacher_logits: torch.Tensor,
             target: torch.Tensor,
             alpha: float = 1.0,
             beta: float  = 2.0,
             T: float     = 4.0) -> torch.Tensor:
    """DKD: α·TCKD + β·NCKD  (已乘 T²)"""
    m_y      = _mask_by_class(student_logits, target, invert=False)
    m_not_y  = ~m_y

    p_s = F.softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)

    # --- TCKD ---
    p_s_2 = _gather_two_probs(p_s, m_y, m_not_y)
    p_t_2 = _gather_two_probs(p_t, m_y, m_not_y)
    tckd  = F.kl_div(p_s_2.log(), p_t_2, reduction='batchmean')

    # --- NCKD ---
    p_s_not = (p_s * m_not_y).clamp_min(1e-12)
    p_t_not = (p_t * m_not_y).clamp_min(1e-12)
    p_s_not = p_s_not / p_s_not.sum(1, keepdim=True)
    p_t_not = p_t_not / p_t_not.sum(1, keepdim=True)
    nckd    = F.kl_div(p_s_not.log(), p_t_not, reduction='batchmean')

    return (alpha * tckd + beta * nckd) * (T * T)


# ---------- Distillation wrapper -------------------------------------
class DistillationLoss(nn.Module):

    def __init__(self,
                 base_criterion,
                 teacher_model,
                 distillation_type,
                 alpha,
                 tau,
                 beta = 2.0,):
        super().__init__()
        assert distillation_type in ['none', 'soft', 'hard', 'dkd']
        self.base_criterion  = base_criterion
        self.teacher_model   = teacher_model
        self.distillation_type = distillation_type

        if self.distillation_type == 'dkd':
            self.alpha, self.beta, self.tau = 1.0, 4.0, 4.0
        else:
            self.alpha, self.beta, self.tau = alpha, beta, tau

        if self.distillation_type != 'none':
            # Teacher 只推理
            for p in self.teacher_model.parameters():
                p.requires_grad_(False)
            self.teacher_model.eval()

    def forward(self, inputs, outputs, labels):
        """
        outputs :  Tensor                        – 单 head
               或 (logits, logits_kd)           – 双 head
        """
        if isinstance(outputs, torch.Tensor):
            outputs, outputs_kd = outputs, None
        else:
            outputs, outputs_kd = outputs   # tuple

        if outputs_kd is None:
            outputs_kd = outputs

        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        with torch.no_grad():
            teacher_logits = self.teacher_model(inputs)
            if isinstance(teacher_logits, (list, tuple)):
                teacher_logits = teacher_logits[0]

        if self.distillation_type == 'soft':
            T = self.tau
            distill_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.softmax     (teacher_logits / T, dim=1),
                reduction='batchmean') * (T * T)

            total_loss = (1 - self.alpha) * base_loss + self.alpha * distill_loss

        elif self.distillation_type == 'hard':
            distill_loss = F.cross_entropy(
                outputs_kd, teacher_logits.argmax(dim=1))
            total_loss = (1 - self.alpha) * base_loss + self.alpha * distill_loss

        elif self.distillation_type == 'dkd':
            distill_loss = dkd_loss(outputs_kd, teacher_logits,
                                    labels,
                                    alpha=self.alpha,  # = α_TCKD
                                    beta =self.beta,
                                    T    =self.tau)
            total_loss = base_loss + distill_loss          # DKD 推荐 CE 权重 = 1

        return total_loss
