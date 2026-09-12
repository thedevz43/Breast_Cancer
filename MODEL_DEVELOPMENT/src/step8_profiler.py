import time
import torch
import torch.nn as nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / (1024 ** 2)
    return round(size_all_mb, 2)

def estimate_flops(model, input_tensor):
    # Standardized FLOPs estimation
    params = count_parameters(model)
    # Approximate MACs for typical 2D CNNs
    estimated_macs = params * 2
    return round(estimated_macs / 1e6, 2) # MFLOPs

def measure_latency_ms(model, sample_input, device, n_runs=10):
    if hasattr(model, 'eval'):
        model.eval()
    sample_input = sample_input.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(sample_input)
            
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(sample_input)
    end = time.perf_counter()
    
    avg_latency_ms = ((end - start) / float(n_runs)) * 1000.0
    return round(avg_latency_ms, 2)
