from functools import wraps
from torch.amp import autocast

def mixed_precision_decorator(use_amp):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Only extract 'self' if it's a method (optional for clarity)
            if args and hasattr(args[0], '__class__'):
                self = args[0]
            
            if use_amp:
                # Use autocast for mixed precision only when required
                with autocast(device_type='cuda'):
                    return func(*args, **kwargs)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator
