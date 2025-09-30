from pathlib import Path

import torch


def export_to_onnx(
    model: torch.nn.Module,
    resolution: int,
    output: Path,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)

    # Remember original training state
    was_training = model.training

    # Switch to eval for export
    model.eval()
    dummy_input = torch.randn(1, 3, resolution, resolution)

    torch.onnx.export(
        model,
        dummy_input,
        str(output),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # Restore original state
    if was_training:
        model.train()

    return output
