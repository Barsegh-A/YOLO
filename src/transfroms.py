import albumentations

transforms_alb = {
    'train': albumentations.Compose(
        [
        albumentations.HorizontalFlip(p=0.5),
        albumentations.Affine(
            scale=(0.8, 1.2),
            translate_percent=(-0.2, 0.2),
            p=0.5
        ),
        albumentations.ColorJitter(
            brightness=(0.8, 1.2),
            contrast=(0.8, 1.2),
            saturation=(0.8, 1.2),
            hue=(-0.2, 0.2),
            p=0.5,
        ),
        albumentations.Resize(224, 224)
        ],
        bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['labels'])
    ),
    'val': albumentations.Compose(
        [    
        albumentations.Resize(224, 224)
        ],
        bbox_params=albumentations.BboxParams(format='pascal_voc', label_fields=['labels'])
    )
    
}