
import logging
import platform

import torch

def select_device(min_memory = 2048):
    logger = logging.getLogger(__name__)
    if torch.cuda.is_available():
        available_gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free_memory = props.total_memory - torch.cuda.memory_reserved(i)
            available_gpus.append((i, free_memory))
        selected_gpu, max_free_memory = max(available_gpus, key=lambda x: x[1])
        device = torch.device(f'cuda:{selected_gpu}')
        free_memory_mb = max_free_memory / (1024 * 1024)
        if free_memory_mb < min_memory:
            logger.warning('GPU %s has %.2f MB memory left.', selected_gpu, free_memory_mb)
            device = torch.device('cpu')
    elif (
        platform.system() == 'Darwin'
        and hasattr(torch.backends, 'mps')
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        device = torch.device('mps')
        logger.info('Using Apple Silicon GPU via MPS backend.')
    else:
        logger.warning('未找到GPU，使用CPU模式')
        device = torch.device('cpu')
    
    return device
