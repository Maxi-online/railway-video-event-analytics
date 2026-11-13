# Runtime and configuration

The full stack uses PyTorch/torchvision, Ultralytics, GPU ONNX Runtime, OpenCV, Hydra and Kafka. Review the preserved implementation guide before selecting compatible Torch/CUDA builds. Set authorized local weight/video paths in `src/config/config.yaml`; the live entry point is `python main.py`. The Compose entry point invokes this script. Hydra environment references use `${oc.env:NAME}`; they are not Docker Compose substitution syntax.

Use the component navigation and validation boundaries in the root [README](../README.md). Runtime services and external providers are not started by the offline replay.
