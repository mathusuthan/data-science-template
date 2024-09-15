from pathlib import Path
from typing import List

import pandas as pd
import typer
from loguru import logger
from sklearn.inspection import PartialDependenceDisplay
from tqdm import tqdm
import seaborn as sns
from utils.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
        # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
        input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
        output_path: Path = FIGURES_DIR / "plot.png",
        # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()


def barplot(x: str, y: str, data: pd.DataFrame, title: str = None, xlabel: str = None, ylabel: str = None):
    """
    Generate a barplot
    :param x: the x-axis variable
    :param y: the y-axis variable
    :param data: the data
    :param title: the title of the plot
    :param xlabel: the x-axis label
    :param ylabel: the y-axis label
    :return:
    """
    if not xlabel:
        xlabel = x

    if not ylabel:
        ylabel = y

    if not title:
        title = f"{y} by {x}"

    return (sns.barplot(x=x, y=y, data=data)
            .set(title=title, xlabel=xlabel, ylabel=ylabel))


def boxplot(x: str, y: str, data: pd.DataFrame, title: str = None, xlabel: str = None, ylabel: str = None):
    """
    Generate a barplot
    :param x: the x-axis variable
    :param y: the y-axis variable
    :param data: the data
    :param title: the title of the plot
    :param xlabel: the x-axis label
    :param ylabel: the y-axis label
    :return: Seaborn boxplot object
    """
    if not xlabel:
        xlabel = x

    if not ylabel:
        ylabel = y

    if not title:
        title = f"Distribution of {y} by {x}"

    return (sns.boxplot(x=x, y=y, data=data)
            .set(title=title, xlabel=xlabel, ylabel=ylabel))


def histogram(x: str, data: pd.DataFrame, title: str = None, xlabel: str = None, ylabel: str = None):
    """
    Generate a histogram
    :param x: the variable to plot
    :param data: the data
    :param title: the title of the plot
    :param xlabel: the x-axis label
    :param ylabel: the y-axis label
    :return: Seaborn histogram object
    """
    if not xlabel:
        xlabel = x

    if not title:
        title = f"Distribution of {x}"

    return (sns.histplot(data=data, x=x)
            .set(title=title, xlabel=xlabel, ylabel=ylabel))


def set_plot_size(width: int = 10, height: int = 6):
    """
    Set the size of the plot
    :param width: width, default is 10 inches
    :param height: height default is 6 inches
    :return: Seaborn object
    """
    sns.set(rc={'figure.figsize': (width, height)})
    return sns


def pdp(model, X, features: List[str], categorical_features=None):
    """ Generate a partial dependence plot for a feature
    :param model: A trained model
    :param X: The input data
    :param features: List of features to plot
    :param categorical_features: Optional list of categorical features in the feature list
    :return: PartialDependenceDisplay object
    """

    return PartialDependenceDisplay.from_estimator(model, X, features, categorical_features=categorical_features)
