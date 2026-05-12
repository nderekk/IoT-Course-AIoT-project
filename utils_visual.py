import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def plot_instance_time_domain(df: pd.DataFrame, title = "Time Domain Plot", axes = ['acc_x', 'acc_y', 'acc_z']):
    """Visualizes the movement instance to a plot in time domain.

    Args:
        df: The DataFrame to be visualized in time domain.
        title: The title of the plot.

    Returns:

    """
    df[axes].plot(figsize=(20, 10), linewidth=2, fontsize=20).legend(fontsize=18)
    plt.title(title, fontsize=20)

    plt.xlabel('Sample', fontsize=20)
    plt.ylabel('Axes', fontsize=20)
    plt.show()


def plot_instance_3d(
        df: pd.DataFrame,
        axes: tuple = ("acc_x", "acc_y", "acc_z"),
        title: str = "3d plot",
):
    """Plots a 3-axes DataFrame in 3D graph.

    Args:
        df: The DataFrame to be plotted in 3D.
        axes_list: Tuple with the 3-axis values. For gyroscope axes should
            be: ("gyr_x", "gyr_y", "gyr_z")

    Returns:

    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # print the plot in 3D

    xs = df[axes[0]]
    ys = df[axes[1]]
    zs = df[axes[2]]

    ax.scatter(xs, ys, zs, color='green', s=50, alpha=0.6, edgecolors='w')

    ax.set_xlabel(axes[0])
    ax.set_ylabel(axes[1])
    ax.set_zlabel(axes[2])


def plot_np_instance(
        np_array: np.ndarray,
        columns_list: list
):
    """Plot NumPy instance using DataFrames (pandas). It first transforms the
        array into
    DataFrame with its corresponding columns naming, and then, it plots the
        DataFrame in time domain.

    Args:
        np_array: The NumPy array to be transformed.
        columns_list: The columns list that the DataFrame and the plot will
            have.

    Returns:

    """
    df = pd.DataFrame(np_array, columns=columns_list)
    df.plot(figsize=(20, 10), linewidth=2, fontsize=20)
    plt.xlabel('Sample', fontsize=20)
    plt.ylabel('Axes', fontsize=20)


def plot_heatmap(df: pd.DataFrame):
    """Visualizes the heatmap of the DataFrame's values.

    Args:
        df: A DataFrame.

    Returns:

    """
    plt.figure(figsize=(14, 6))
    sns.heatmap(df, cmap='plasma')


def plot_scatter_pca(
        df: pd.DataFrame,
        c_name: str,
        cmap_set: str = "plasma"
):
    """Visualizes the values of the component columns of the DataFrame
    according to its column that includes the labels.

    Args:
        df: The DataFrame that contains the transformed data after the PCA
            procedure.
        c_name: The name of the column that includes the labels.
        cmap_set: The format of the plot.

    Returns:

    """
    if len(df.columns) == 3:
        plt.style.use('classic')
        plt.figure(figsize=(16, 8))
        plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df[c_name], cmap=cmap_set)
        plt.xlabel('First principal component')
        plt.ylabel('Second Principal Component')
    elif len(df.columns) == 4:
        plt.style.use('classic')
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], c=df[c_name], cmap=cmap_set)
        ax.set_xlabel('First principal component')
        ax.set_ylabel('Second Principal Component')
        ax.set_zlabel('Third Principal Component')
    else:
        print("The DataFrame has more than 4 columns.")
        
        
def plot_bar(df: pd.DataFrame, title: str):
    """Visualizes the bar plot of the DataFrame's values.

    Args:
        df: A DataFrame.
        title: The title of the plot.

    Returns:
        void
    """
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    barplot = sns.barplot(data=df, x=df.columns[0], y=df.columns[1], palette='viridis')

    # Add labels and title
    plt.title(title, fontsize=16)
    plt.xlabel(df.columns[0], fontsize=12)
    plt.ylabel(df.columns[1], fontsize=12)
    plt.xticks(rotation=45)

    # Optional: Add duration values on top of the bars
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.1f'),
                        (p.get_x() + p.get_width() / 2.,
                            p.get_height()),
                        ha = 'center', va = 'center',
                        xytext = (0, 9),
                        textcoords = 'offset points')

    plt.tight_layout()
    plt.show()


def plot_box(df: pd.DataFrame, title: str):
    """Visualizes the boxplot of the DataFrame's values.

    Args:
        df: A DataFrame.
        title: The title of the plot.

    Returns:

    """
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    sns.boxplot(data=df)
    plt.title(title, fontsize=16)
    plt.ylabel('Values', fontsize=12)
    plt.xlabel('Axes', fontsize=12)
    plt.show()
    
    
def plot_distribution_analysis(df: pd.DataFrame, column_name: str):
    """
        Visualizes the distribution of a specific column in the DataFrame using
        a histogram with KDE and a Q-Q plot.
        
        Args:
            df: A DataFrame.
            column_name: The name of the column to analyze.

        Returns:
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 1. Histogram + KDE
    sns.histplot(df[column_name], kde=True, ax=axes[0], color='skyblue', edgecolor='black')
    axes[0].set_title(f'Histogram & KDE of {column_name}')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')

    # 2. Q-Q Plot
    # stats.probplot calculates the quantiles and plots against a normal distribution
    stats.probplot(df[column_name], dist="norm", plot=axes[1])
    axes[1].set_title(f'Q-Q Plot of {column_name}')
    
    plt.tight_layout()
    plt.show()