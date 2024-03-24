from os import path

model_name = "my_model.keras"
model_windows_name = "keras_model_dir"
model_dir_name = "current_model_cnn_lstm_25_epochs"
# model_dir_name = "current_model_cnn_100_epochs"


windows_base_path = path.join("O:", "neural_networks")
windows_training_directory = path.join(windows_base_path, "recordings", "training")
windows_testing_directory = path.join(windows_base_path, "recordings", "testing")
windows_model_location = path.join(windows_base_path, "models")

linux_base_path = path.join("/media", "dj", "Games Drive", "neural_networks", )
linux_training_directory = path.join(linux_base_path, "recordings", "training")
linux_testing_directory = path.join(linux_base_path, "recordings", "testing")
linux_model_location = path.join(linux_base_path, "models")

### Operating modes for different settings.
# Whether to convert images to greyscale. Most models were trained in greyscale.
grayscale = True
color_mode = "grayscale"

# Number of images to batch to the GPU. gpu_batch will need to be changed depending on model size.
gpu_batch_size = 32
