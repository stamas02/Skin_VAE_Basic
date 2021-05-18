import torch

msg = "Cuda device available: {}"
print(msg.format(torch.cuda.is_available()))

msg = "Cuda device ID: {}"
print(msg.format(torch.cuda.current_device()))

msg = "Cuda device Address: {}"
print(msg.format(torch.cuda.device(0)))

msg = "Cuda device Name: {}"
print(msg.format(torch.cuda.get_device_name(0)))

