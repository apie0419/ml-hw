from model      import cnn_model
from config     import cfg
from datasets   import make_test_loader

import torch, os
import numpy as np

model = cnn_model()

weight_path = cfg.MODEL.OUTPUT_PATH
use_cuda    = cfg.DEVICE.CUDA
gpu_id      = cfg.DEVICE.GPU

weight = torch.load(weight_path)
model.load_state_dict(weight)

if use_cuda:
    torch.cuda.set_device(gpu_id)
    model.cuda()

test_loader = make_test_loader(cfg)

model.eval()

test_loss = 0.
correct = 0.
total = 0.

with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))

        pred = output.data.max(1, keepdim=True)[1]
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

    print('Test Loss: {:.6f}'.format(test_loss))

    print('Test Accuracy: %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))