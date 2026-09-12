import torchvision.transforms as T

def get_transforms(is_train=True, img_size=224):
    if is_train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_raw_transforms(img_size=224):
    # Unnormalized 0-1 float array for classical ML models
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor()
    ])
