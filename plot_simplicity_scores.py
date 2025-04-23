import json
import matplotlib.pyplot as plt
import seaborn as sns

# Filepath to the JSON file
input_min, input_max = -100, 100
# input_min, input_max = -1, 1
json_file = f"simplicity_scores_({input_min},{input_max}).json"

# Read data from the JSON file
with open(json_file, "r") as f:
    data = json.load(f)

# Initialize a dictionary to store the flattened values
filtered_data = {}

# Process each key in the JSON data
for key, value in data.items():
    # Flatten the list of lists and exclude 1000
    flattened_values = [score for sublist in value for score in sublist if score != 1000]
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
plt.xlabel("Simplicity Score (↑)")
plt.title("Simplicity Score Distribution")

# Show the plot
plt.tight_layout()
plt.savefig(f"simplicity_scores_boxplot_({input_min},{input_max}).png", dpi=300, bbox_inches="tight")