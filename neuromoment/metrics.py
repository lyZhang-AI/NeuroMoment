import torch


def retrieval_metrics(scores, labels, vocabulary_size=None):
    """Macro metrics over observed classes, with explicit coverage for interpretation."""
    size = scores.shape[1] if vocabulary_size is None else vocabulary_size
    if scores.shape[1] != size or labels.numel() == 0:
        raise ValueError("Invalid candidate pool or empty evaluation")
    order = scores.argsort(dim=-1, descending=True, stable=True)
    ranks = (order == labels[:, None]).long().argmax(-1) + 1
    classes = labels.unique()
    def macro(values):
        return torch.stack([values[labels == cls].float().mean() for cls in classes]).mean().item()
    result = {f"balanced_top{k}": macro((ranks <= min(k, size)).float()) for k in [1, 5, 10]}
    result.update(mrr=(1 / ranks.float()).mean().item(), balanced_mrr=macro(1 / ranks.float()),
                  classes_evaluated=len(classes), candidates=size, coverage=len(classes) / size,
                  events=len(labels), chance_top10=min(10, size) / size)
    return result
