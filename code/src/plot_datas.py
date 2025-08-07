import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_datas(data_dict, logger, log='train'):
    base_dir = f"../logs/{logger.name}_v{logger.version}/"
    os.makedirs(base_dir, exist_ok=True)
    
    json_path = os.path.join(base_dir, f"{log}_{logger.name}_v{logger.version}.json")
    
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            saved_data = json.load(f)
    else:
        saved_data = {}

    for key, value in data_dict.items():
        if isinstance(value, (list, tuple)):
            processed_value = float(np.mean(value))
        else:
            processed_value = value
        if key not in saved_data:
            saved_data[key] = []
        saved_data[key].append(processed_value)


    with open(json_path, "w") as f:
        json.dump(saved_data, f, indent=4)

    for metric, values in saved_data.items():
        plt.figure()
        epochs = list(range(1, len(values) + 1))
        plt.plot(epochs, values, marker='o', linestyle='-')
        plt.title(f"{metric} over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.grid(True)
        plt.xticks(epochs)

        plot_path = os.path.join(base_dir, f"{metric}_{logger.name}_v{logger.version}.png")
        plt.savefig(plot_path)
        plt.close()
