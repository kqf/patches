from pathlib import Path

import onnx

from patches.export import export_to_onnx
from patches.train import build_model


def test_export_to_onnx_preserves_state(tmp_path: Path):
    model = build_model()
    model.train()  # explicitly set training mode

    export_path = tmp_path / "resnet50.onnx"
    result_path = export_to_onnx(
        model,
        resolution=128,
        output=export_path,
    )

    # File exists
    assert result_path.exists()

    # Model state is preserved (train mode in this case)
    assert model.training is True

    # ONNX file is valid
    onnx_model = onnx.load(str(result_path))
    onnx.checker.check_model(onnx_model)
