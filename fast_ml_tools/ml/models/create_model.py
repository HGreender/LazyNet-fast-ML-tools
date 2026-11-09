import segmentation_models_pytorch as smp


def efficientnetb4_unet_base(
        classes=1, 
        in_channels=1, 
        activation: str | None = None
    ):
    return smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        decoder_attention_type="scse",
        classes=classes,
        activation=activation,
        in_channels=in_channels,
        decoder_channels=(256, 128, 64, 32, 16),
    )

def efficientnetb4_unetpp_base(
        classes=1, 
        in_channels=1, 
        activation: str | None = None
    ):
    """
    Создание UNet++ с EfficientNet-B4
    decoder_channels: каналы в декодере. Для B4 обычно используют уменьшенные каналы для экономии памяти.
    """
    return smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        decoder_attention_type="scse",
        classes=classes,
        activation=activation,
        in_channels=in_channels,
        decoder_channels=(256, 128, 64, 32, 16),
    )


def efficientnetb4_unet(classes: list = ['target_class'], activation: str | None = None):
    """
    Создание UNet с EfficientNet-B4
    decoder_channels: каналы в декодере. Для B4 обычно используют уменьшенные каналы для экономии памяти.
    """
    return efficientnetb4_unet_base(
        classes=len(classes),
        in_channels=3,
        activation=activation,
    )


def efficientnetb4_unet_1_channel(classes: list = ['target_class'], activation: str | None = None):
    """
    Создание UNet с EfficientNet-B4, одноканальная версия для Gray Scaled
    """
    return efficientnetb4_unet_base(
        classes=len(classes),
        in_channels=1,
        activation=activation,
    )

def efficientnetb4_unetpp(classes: list = ['target_class'], activation: str | None = None):
    """
    Создание UNet++ с EfficientNet-B4
    decoder_channels: каналы в декодере. Для B4 обычно используют уменьшенные каналы для экономии памяти.
    """
    return efficientnetb4_unetpp_base(
        classes=len(classes),
        in_channels=3,
        activation=activation,
    )


def efficientnetb4_unetpp_1_channel(classes: list = ['target_class'], activation: str | None = None):
    """
    Создание UNet++ с EfficientNet-B4
    decoder_channels: каналы в декодере. Для B4 обычно используют уменьшенные каналы для экономии памяти.
    """
    return efficientnetb4_unetpp_base(
        classes=len(classes),
        in_channels=1,
        activation=activation,
    )