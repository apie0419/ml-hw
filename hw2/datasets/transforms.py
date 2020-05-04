from torchvision import transforms

def build_transform(cfg):

    transform = transforms.Compose([transforms.Resize(cfg.DATA.RESIZE),
                                     transforms.ToTensor(),
                                     transforms.Normalize(cfg.DATA.PIXEL_MEAN,
                                                          cfg.DATA.PIXEL_STD)])
    return transform