import torch
import numpy as np

def snr_db2sigma(snr):
    return 10**(-snr*1.0/20)

def errors_ber(y_true, y_pred, mask=None):
    if mask == None:
        mask=torch.ones(y_true.size(),device=y_true.device)
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)
    mask = mask.view(mask.shape[0], -1, 1)
    myOtherTensor = (mask*torch.ne(torch.round(y_true), torch.round(y_pred))).float()
    res = sum(sum(myOtherTensor))/(torch.sum(mask))
    return res

def errors_bler(y_true, y_pred, get_pos = False):
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    decoded_bits = torch.round(y_pred).cpu()
    X_test       = torch.round(y_true).cpu()
    tp0 = (abs(decoded_bits-X_test)).view([X_test.shape[0],X_test.shape[1]])
    tp0 = tp0.detach().cpu().numpy()
    bler_err_rate = sum(np.sum(tp0,axis=1)>0)*1.0/(X_test.shape[0])

    if not get_pos:
        return bler_err_rate
    else:
        err_pos = list(np.nonzero((np.sum(tp0,axis=1)>0).astype(int))[0])
        return bler_err_rate, err_pos

def corrupt_signal(code,sigma):
    device = code.device
    dist = torch.distributions.Normal(torch.tensor([0.0],device=device),torch.tensor([sigma],device=device))
    noise = dist.sample(code.shape).squeeze()
    corrupted_signal = code + noise

    return corrupted_signal

def min_sum_log_sum_exp(x, y):
    log_sum_ms = torch.min(torch.abs(x),torch.abs(y))*torch.sign(x)*torch.sign(y)
    return log_sum_ms

def log_sum_exp(x,y):
    def log_sum_exp_(LLR_vector):
        sum_vector = LLR_vector.sum(dim=1,keepdim=True)
        sum_concat = torch.concat([sum_vector,torch.zeros_like(sum_vector)],dim=1)
        return torch.logsumexp(sum_concat,dim=1)-torch.logsumexp(LLR_vector,dim=1)

    Lv = log_sum_exp_(torch.cat([x.unsqueeze(2),y.unsqueeze(2)],dim=2).permute(0,2,1))
    return Lv