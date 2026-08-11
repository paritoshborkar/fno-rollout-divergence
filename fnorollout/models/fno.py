from neuralop.models import FNO


def build_fno2d(
    in_channels: int,
    out_channels: int,
    n_modes: list[int],
    hidden_channels: int,
    n_layers: int,
) -> FNO:
    """
    Instantiates and returns a FNO model from the neuralop package
    """
    return FNO(
        n_modes=tuple(n_modes),
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
    )
