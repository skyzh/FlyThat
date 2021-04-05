from torchvision import transforms


def feature_transformer(enable_normalize=False):
    composed = []
    # h*w*c转变为c*h*w，正则化为0-1之间
    composed.append(transforms.ToTensor())
    if enable_normalize:
        # 正则化为-1到1之间
        composed.append(transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),)
    return transforms.Compose(composed)
