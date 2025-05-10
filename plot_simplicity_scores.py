import json
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Plot simplicity scores from a JSON file.")
    parser.add_argument(
        "--json_file",
        type=str,
        required=True,
        help="Path to the input JSON file containing simplicity scores.",
    )
    return parser.parse_args()

args = parse_args()
json_file = args.json_file
json_file = f"output/{json_file}"

if "kaiming" in json_file:
    weight_mode = "Kaiming"
elif "xavier" in json_file:
    weight_mode = "Xavier"
elif "orthogonal" in json_file:
    weight_mode = "Orthogonal"

if "TD3" in json_file:
    policy_name = "TD3"
elif "SAC" in json_file:
    policy_name = "SAC"

# Read data from the JSON file
with open(json_file, "r") as f:
    data = json.load(f)

# Initialize a dictionary to store the flattened values
filtered_data = {}

# Process each key in the JSON data
for key, value in data.items():
    # Flatten the list of lists and exclude 1000
    flattened_values = [val for val in value if val != 1000]
    filtered_data[key] = flattened_values

# Prepare data for plotting
plot_data = []
labels = []
for key, values in filtered_data.items():
    plot_data.append(values)
    labels.append(key)

# Create the box plot
plt.figure(figsize=(6, 4))
sns.boxplot(data=plot_data, orient="h", palette="pastel", showfliers=False)

# Add labels and title
plt.yticks(ticks=range(len(labels)), labels=labels)
plt.xlabel(f"{policy_name} Simplicity Score (↑)")
# plt.xlim(1.5, 5.5)
plt.title(f"with {weight_mode} Normal Initialization", fontsize=12)

# Show the plot
plt.tight_layout()
plt.savefig(f"{json_file.split('.')[0]}.png", dpi=300, bbox_inches="tight")