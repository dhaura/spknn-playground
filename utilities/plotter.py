import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description="Plot parameters from SPKNN results CSV.")
parser.add_argument("-input_file", type=str, required=True, help="Path to the SPKNN results CSV file.")
parser.add_argument("-output_file", type=str, help="Path to save the output PDF with plots.")
args = parser.parse_args()

# Load the CSV file
df = pd.read_csv(f"~/repos/SpKNN/spknn-playground/results/{args.input_file}")

# Drop unnamed index column if present
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Parameters to plot against Recall
parameters = [
    "Single Query Time (microseconds)",
    "Searching Time (Seconds)",
    "QPS",
    "RR@10"
]

# Create a PDF to save all plots
with PdfPages(args.output_file) as pdf:
    # Line plots for each parameter vs Recall
    for param in parameters:
        plt.figure(figsize=(10, 6))
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model]
            plt.plot(model_data['Recall'], model_data[param], marker='o', label=model)
        plt.xlabel("Recall")
        plt.ylabel(param)
        plt.title(f"{param} vs Recall")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # Bar chart for Indexing Time per model (constant per model)
    plt.figure(figsize=(10, 6))
    indexing_times = df.groupby('Model')["Indexing Time"].first()
    indexing_times.plot(kind='bar', color='skyblue')
    plt.ylabel("Indexing Time")
    plt.title("Indexing Time per Model")
    plt.xticks(rotation=45)
    plt.tight_layout()
    pdf.savefig()
    plt.close()
