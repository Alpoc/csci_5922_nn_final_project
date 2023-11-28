from os import path

model_name = "current_model.keras"

windows_base_path = path.join("O:", "neural_networks")
windows_training_directory = path.join(windows_base_path, "recordings", "training")
windows_testing_directory = path.join(windows_base_path, "recordings", "testing")
windows_model_location = path.join(windows_base_path, "models")

linux_base_path = path.join("/media", "dj", "Games Drive", "neural_networks",)
linux_training_directory = path.join(linux_base_path, "recordings", "training")
linux_testing_directory = path.join(linux_base_path, "recordings", "testing")
linux_model_location = path.join(linux_base_path, "models")
