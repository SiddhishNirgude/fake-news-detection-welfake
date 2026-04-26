"""
utils.py
--------
Shared utility functions for the WELFake fake news detection project.
Includes timing helpers, pretty printing, path management,
and reproducibility setup.

Author: Siddhish Nirgude
Course: CMSE 928 - Applied Machine Learning
"""

import functools
import os
import random
import time

import numpy as np


def set_seed(seed=42):
    """
    Set random seeds for reproducibility across numpy and random.
    Should be called at the start of every notebook and training script.

    Parameters
    ----------
    seed : int
        Random seed value. Default 42.

    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    print(f"Random seed set to {seed}")


def timer(func):
    """
    Decorator that measures and prints the execution time of a function.
    Useful for benchmarking training and feature extraction steps.

    Parameters
    ----------
    func : callable
        Function to be timed.

    Returns
    -------
    callable
        Wrapped function that prints elapsed time after execution.

    Examples
    --------
    @timer
    def train_model():
        ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} completed in {elapsed:.2f} seconds")
        return result

    return wrapper


def print_section(title):
    """
    Print a formatted section header to terminal output.
    Used to visually separate pipeline stages in notebook output.

    Parameters
    ----------
    title : str
        Section title to display.

    Returns
    -------
    None

    Examples
    --------
    print_section("TF-IDF Feature Engineering")

    Output::

        ============================================================
        TF-IDF Feature Engineering
        ============================================================
    """
    separator = "=" * 60
    print(separator)
    print(title)
    print(separator)


def ensure_dirs(path_list):
    """
    Create multiple directories if they do not already exist.
    Used at the start of notebooks to guarantee output folders exist.

    Parameters
    ----------
    path_list : list of str
        List of directory paths to create.

    Returns
    -------
    None
    """
    for path in path_list:
        os.makedirs(path, exist_ok=True)
        print(f"Directory ready: {path}")


def get_file_size_mb(filepath):
    """
    Return the size of a file in megabytes.
    Used to verify saved model and data file sizes.

    Parameters
    ----------
    filepath : str
        Path to the file.

    Returns
    -------
    float
        File size in MB rounded to 2 decimal places.
        Returns 0.0 if file does not exist.
    """
    if not os.path.exists(filepath):
        print(f"Warning: file not found: {filepath}")
        return 0.0
    return round(os.path.getsize(filepath) / (1024 * 1024), 2)
