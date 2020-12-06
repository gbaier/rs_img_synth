import torch
import torch.nn as nn


class HingeDiscriminator(nn.Module):
    """ Hinge loss for discriminator

    [1] Jae Hyun Lim, Jong Chul Ye, "Geometric GAN", 2017
    [2] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida,
        "Spectral normalization for generative adversarial networks", 2018

    """

    def __init__(self):
        super().__init__()

    def forward(self, disc_real_output, disc_fake_output):
        """

        Args:
        disc_real_output: the discriminators output for a real sample
        disc_fake_output: the discriminators output for a fake sample

        """

        loss = -torch.mean(
            torch.min(disc_real_output - 1, torch.zeros_like(disc_real_output))
        )
        loss -= torch.mean(
            torch.min(-disc_fake_output - 1, torch.zeros_like(disc_fake_output))
        )

        return loss


class HingeGenerator(nn.Module):
    """ Hinge loss for discriminator

    [1] Jae Hyun Lim, Jong Chul Ye, "Geometric GAN", 2017
    [2] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida,
        "Spectral normalization for generative adversarial networks", 2018

    """

    def __init__(self):
        super().__init__()

    def forward(self, disc_fake_output):
        return -torch.mean(disc_fake_output)


def iou(pr, gt, eps=1e-7, axis=(0, 2, 3)):
    """
    intersection over union loss

    Source:
        https://github.com/catalyst-team/catalyst/
        https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/utils/functions.py
        https://discuss.pytorch.org/t/iou-score-for-muilticlass-segmentation/89350

    Args:
        pr (torch.Tensor): A list of predicted elements as softmax
        gt (torch.Tensor):  A list of elements that are to be predicted as one hot encoded
        eps (float): epsilon to avoid zero division
    Returns:
        float: IoU (Jaccard) score
    """

    intersection = torch.sum(gt.float() * pr.float(), axis)
    union = torch.sum(gt, axis).float() + torch.sum(pr, axis).float() - intersection + eps

    asdf = intersection / union

    return torch.mean(asdf)

def pixel_acc(pred, target):
    """ pixel accuracy 

    Args:
        pred (torch.Tensor): predicted classes. Not one-hot encoded
        targer (torch.Tensor):correct classes. Not one-hot encoded

    """

    corr_pix = torch.sum(pred == target).float()

    return corr_pix / pred.nelement()
