import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


@torch.enable_grad()
def grad_cam(model, x, hooks, cls_idx=None):
    """ cf CheXpert: Test Results / Visualization; visualize final conv layer, using grads of final linear layer as weights,
    and performing a weighted sum of the final feature maps using those weights.
    cf Grad-CAM https://arxiv.org/pdf/1610.02391.pdf """
    model.eval()
    model.zero_grad()

    # register backward hooks
    conv_features, linear_grad = [], []
    forward_handle = hooks['forward'].register_forward_hook(
        lambda module, in_tensor, out_tensor: conv_features.append(out_tensor))
    backward_handle = hooks['backward'].register_backward_hook(
        lambda module, grad_input, grad_output: linear_grad.append(grad_input))

    # run model forward and create a one hot output for the given cls_idx or max class
    outputs = model(x)
    if not cls_idx: cls_idx = outputs.argmax(1)
    one_hot = F.one_hot(cls_idx, outputs.shape[1]).float().requires_grad_(True)

    # run model backward
    one_hot.mul(outputs).sum().backward()

    # compute weights; cf. Grad-CAM eq 1 -- gradients flowing back are global-avg-pooled to obtain the neuron importance weights
    weights = linear_grad[0][2].mean(1).view(1, -1, 1, 1)
    # compute weighted combination of forward activation maps; cf Grad-CAM eq 2; linear combination over channels
    cam = F.relu(torch.sum(weights * conv_features[0], dim=1, keepdim=True))

    # normalize each image in the minibatch to [0,1] and upscale to input image size
    cam = cam.clone()  # avoid modifying tensor in-place

    def norm_ip(t, min, max):
        t.clamp_(min=min, max=max)
        t.add_(-min).div_(max - min + 1e-5)

    for t in cam:  # loop over mini-batch dim
        norm_ip(t, float(t.min()), float(t.max()))

    cam = F.interpolate(cam, x.shape[2:], mode='bilinear', align_corners=True)

    # cleanup
    forward_handle.remove()
    backward_handle.remove()
    model.zero_grad()

    return cam


def visualize(model, dataloader, grad_cam_hooks, save_dir):
    # 1. run through model to compute logits and grad-cam
    for batch_idx, (x, target) in enumerate(dataloader):
        imgs, labels, scores, masks = [], [], [], []
        imgs += [x]
        labels += [target]
        x = x.cuda()
        scores += [model(x).cpu()]
        masks += [grad_cam(model, x, grad_cam_hooks).cpu()]
        imgs, labels, scores, masks = torch.cat(imgs), torch.cat(labels), torch.cat(scores), torch.cat(masks)

        # For each image, get the greyscale of the image and the mask and plot mask over the greyscale image
        for img_id, (img, mask) in enumerate(zip(imgs, masks)):
            fig, ax = plt.subplots(1, 2)
            print_label = labels[img_id]
            ax[0].set_title('Label: {0:1.0f}'.format(print_label), fontsize=10)
            ax[0].imshow(np.transpose(img.cpu().detach().numpy(), [1, 2, 0]), cmap='gray')
            ax[1].set_title('Prediction: {0:1.0f}'.format(np.argmax(scores[img_id].cpu().detach().numpy())))
            ax[1].imshow(np.transpose(img.squeeze().cpu().detach().numpy(), [1, 2, 0]), cmap='gray')
            ax[1].imshow(mask.squeeze().cpu().detach().numpy(), cmap='jet', alpha=0.5)
            ax[1].axis('off')
            ax[0].axis('off')
            plt.savefig(os.path.join(save_dir, '{}-{}.png'.format(batch_idx, img_id)), dpi=300, bbox_inches='tight')
