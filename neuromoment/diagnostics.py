"""Localization diagnostics on absolute time, including boundary mass outside the overlap."""
import torch


def absolute_equivariance_ms(original, shifted, times, delta):
    """Exact W1 between discrete token masses at tau and tau+delta, on their union support.

    A positive event-anchor shift moves a predicted relative token to a later absolute
    coordinate. No cropping/renormalization can hide boundary collapse. Fractional
    token shifts retain the discretization error of the coarse coordinate grid.
    """
    delta=torch.as_tensor(delta,device=times.device,dtype=times.dtype)
    if delta.ndim==0:delta=delta.expand(original.shape[0])
    points=torch.cat([times[None].expand(original.shape[0],-1),times[None]+delta[:,None]],-1)
    locations,order=points.sort(-1)
    signed=torch.cat([original,-shifted],-1).gather(1,order)
    cdf=signed.cumsum(-1)
    return (cdf[:,:-1].abs()*(locations[:,1:]-locations[:,:-1])).sum(-1)*1000


def union_distributions(original,shifted,times,delta):
    """Place both views on an extended absolute-time grid, preserving boundary mass."""
    from torch.nn import functional as F
    if len(times)<2:raise ValueError('Need >=2 temporal tokens')
    delta=torch.as_tensor(delta,device=times.device,dtype=times.dtype)
    if delta.ndim==0:delta=delta.expand(original.shape[0])
    offsets=delta/(times[1]-times[0])
    pad=int(offsets.detach().abs().max().ceil().item())+1
    length=original.shape[-1]
    p=F.pad(original,(pad,pad))
    query=torch.arange(length+2*pad,device=times.device,dtype=times.dtype)[None]-pad-offsets[:,None]
    lo=query.floor().long();hi=lo+1;fraction=query-lo
    q=shifted.gather(1,lo.clamp(0,length-1))*(1-fraction)*((lo>=0)&(lo<length))
    q=q+shifted.gather(1,hi.clamp(0,length-1))*fraction*((hi>=0)&(hi<length))
    return p,q


def full_support_js(original,shifted,times,delta):
    p,q=union_distributions(original,shifted,times,delta)
    mean=(p+q)/2
    def kl(x):return (x*(x.clamp_min(1e-12).log()-mean.clamp_min(1e-12).log())).sum(-1)
    return ((kl(p)+kl(q))/2).mean()
