# Model Placement

The deployed model is `best_finetuned.keras`. The API loads this exact file by
default; it is not retrained, reconstructed, or replaced by the application.

The original training data folders were sorted by Keras as `normal=0` and
`pneumonia=1`, so the saved sigmoid output is interpreted as the pneumonia
probability. Inference converts images to RGB, resizes to 224 × 224, converts
to float32, and applies `1./255` scaling.

You can override the path using the `MODEL_PATH` environment variable.
