import segmentation_models_pytorch as smp


def efficientnetb4_unet(classes: list = ['target_class'], activation: str | None = None):
    """
    Создание UNet с EfficientNet-B4
    decoder_channels: каналы в декодере. Для B4 обычно используют уменьшенные каналы для экономии памяти.
    """
    model = smp.Unet(
        encoder_name='efficientnet-b4',
        encoder_weights='imagenet',
        decoder_attention_type="scse",
        classes=len(classes),
        activation=activation,
        decoder_channels=(256, 128, 64, 32, 16),
    )
    return model


def efficientnetb4_unetpp(classes: list = ['target_class'], activation: str | None = None):
    """
    Создание UNet++ с EfficientNet-B4
    decoder_channels: каналы в декодере. Для B4 обычно используют уменьшенные каналы для экономии памяти.
    """
    model = smp.UnetPlusPlus(
        encoder_name='efficientnet-b4',
        encoder_weights='imagenet',
        decoder_attention_type="scse",
        classes=len(classes),
        activation=activation,
        decoder_channels=(256, 128, 64, 32, 16),
    )
    return model
