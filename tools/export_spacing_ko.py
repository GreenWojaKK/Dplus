"""Exports dokey's Korean spacing model to ONNX, and checks the export against dokey's own forward pass.

Usage: export_spacing_ko.py RESOURCES_DIR OUTPUT_DIR
RESOURCES_DIR holds spacer3.npz and spacer-vocab.txt.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime
import torch
import torch.nn.functional as F

KERNELS = (5, 7, 11, 5, 3)
PAD_INDEX = 0


def load(resources):
    raw = np.load(resources / "spacer3.npz")
    tensors = {}
    for name in raw.files:
        array = raw[name]
        if array.ndim == 3:  # TF conv kernels are (kernel, in, out); torch wants (out, in, kernel).
            array = array.transpose(2, 1, 0)
        tensors["/".join(name.split("/")[1:]).removesuffix(":0")] = torch.from_numpy(array.copy())
    tensors["embedding/embeddings"][PAD_INDEX].zero_()
    return tensors


class Spacer(torch.nn.Module):
    def __init__(self, tensors):
        super().__init__()
        self.names = {key: key.replace("/", "_") for key in tensors}
        for key, tensor in tensors.items():
            self.register_buffer(self.names[key], tensor)

    def weight(self, key):
        return getattr(self, self.names[key])

    def convolve(self, x, slot, kernel):
        total = kernel - 1
        padded = F.pad(x, (total // 2, total - total // 2))
        return F.conv1d(padded, self.weight(f"{slot}/kernel"), self.weight(f"{slot}/bias"))

    def forward(self, indices):
        x = F.embedding(indices, self.weight("embedding/embeddings")).transpose(1, 2)
        branches = [x]
        for slot, kernel in zip(("conv1d", "conv1d_1", "conv1d_2"), KERNELS[:3]):
            branches.append(F.relu(self.convolve(x, slot, kernel)))
        merged = F.relu(self.convolve(torch.cat(branches, dim=1), "conv1d_3", KERNELS[3]))
        return torch.sigmoid(self.convolve(merged, "conv1d_4", KERNELS[4])[:, 0, :])


def main():
    resources, output = Path(sys.argv[1]), Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=True)
    model = Spacer(load(resources)).eval()
    sample = torch.randint(2, 500, (2, 37), dtype=torch.long)
    target = output / "model.onnx"
    torch.onnx.export(
        model, (sample,), str(target), input_names=["indices"], output_names=["probabilities"],
        dynamic_axes={"indices": {0: "batch", 1: "length"}, "probabilities": {0: "batch", 1: "length"}},
        opset_version=17, dynamo=False,
    )
    shutil.copyfile(resources / "spacer-vocab.txt", output / "vocab.txt")
    session = onnxruntime.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    for length in (1, 2, 9, 37, 512):
        probe = torch.randint(0, 2000, (1, length), dtype=torch.long)
        with torch.no_grad():
            expected = model(probe).numpy()
        actual = session.run(None, {"indices": probe.numpy()})[0]
        difference = float(np.abs(expected - actual).max())
        print(f"length {length}: largest difference {difference:.2e}")
        if difference > 1e-5:
            raise SystemExit("the export does not match the source model")
    print("wrote", target)


if __name__ == "__main__":
    main()
