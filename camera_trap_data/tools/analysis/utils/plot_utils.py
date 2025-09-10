import matplotlib.pyplot as plt
import numpy as np

def plot_piechart(class_counts, title, colors, output_path, show_all_legend=True, top_n=5):
    """
    Plot and save a pie chart.

    Args:
        class_counts (dict): Dictionary of class counts {class_name: count}.
        title (str): Title of the pie chart.
        colors (dict): Dictionary mapping class names to colors.
        output_path (str): Path to save the pie chart.
        show_all_legend (bool): Whether to show all legends (True for full-train).
        top_n (int): Number of top classes to highlight (others will be black).
    """
    # Sort class counts by value (descending)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

    # Align colors with the sorted class order
    top_colors = [colors[cls] for cls, _ in sorted_classes]

    # Prepare data for the pie chart
    labels = [f"Class {cls}" for cls, _ in sorted_classes]
    sizes = [count for _, count in sorted_classes]

    # Plot the pie chart
    fig, ax = plt.subplots()
    wedges, texts = ax.pie(
        sizes,
        labels=None,  # Hide labels on the pie chart itself
        colors=top_colors,
        autopct=None,  # Remove percentages
        startangle=90,
        textprops=dict(color="w"),
    )

    # Set title
    ax.set_title(title)

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_multiple_piecharts(piechart_data, output_path, title, colors, top_n=5):
    """
    Plot and save multiple pie charts in a grid (4 per row).

    Args:
        piechart_data (list): List of tuples [(class_counts, ckp_title), ...].
        output_path (str): Path to save the pie chart group.
        title (str): Title of the group of pie charts.
        colors (dict): Color mapping for the dataset {class_name: color}.
        top_n (int): Number of top classes to highlight (others will be black).
    """
    num_charts = len(piechart_data)
    num_rows = (num_charts + 3) // 4  # 4 pie charts per row

    fig, axes = plt.subplots(num_rows, 4, figsize=(16, 4 * num_rows))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, (class_counts, ckp_title) in enumerate(piechart_data):
        # Sort class counts by value (descending) and get top N classes
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        top_classes = sorted_classes[:top_n]
        other_classes = sorted_classes[top_n:]

        # Prepare data for the pie chart
        labels = [f"Class {cls}" for cls, _ in top_classes]
        sizes = [count for _, count in top_classes]
        top_colors = [colors[cls] for cls, _ in top_classes]

        # Add "non-top-5" for other classes
        if other_classes:
            labels.append("non-top-5")
            sizes.append(sum(count for _, count in other_classes))
            top_colors.append("black")  # Use black for non-top-5

        # Plot the pie chart
        ax = axes[i]
        wedges, texts = ax.pie(  # Only unpack wedges and texts since autopct=None
            sizes,
            labels=None,  # Hide labels on the pie chart itself
            colors=top_colors,
            autopct=None,  # Remove percentages
            startangle=90,
            textprops=dict(color="w"),
        )
        ax.set_title(ckp_title)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Set the title for the entire group
    fig.suptitle(title, fontsize=16)

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)