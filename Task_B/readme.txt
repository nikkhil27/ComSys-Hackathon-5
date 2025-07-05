1. Requirements (TensorFlow version, etc.)
2. Command to train: python3 Task_B.py
3. Command to evaluate: python3 evaluate.py
4. Ensure test/ folder is provided in expected structure.
5. Reproducibility and Platform Notes
    This model was trained on macOS using the MPS (Metal Performance Shaders) backend via TensorFlow. Due to low-level differences in hardware acceleration (MPS vs CUDA), training the same model on a different backend (e.g., CUDA on NVIDIA GPUs) may produce slightly different weights, even with the same random seed and data.

    To ensure strict reproducibility:
    1. You are recommended to evaluate using the provided saved model weights (model1.h5, model2.h5, model3.h5) instead of re-training.
    2. Alternatively, if retraining is required, it should be done using TensorFlow with MPS enabled on macOS to minimize deviation.
6. Model Architecture - EfficientNet-Based

