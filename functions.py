import numpy as np
import pandas as pd
import datetime

from datascience import *


import matplotlib.pyplot as plots

# helpers

def original_value(original_values, sd_value ):
    '''convert a standard unit value to it's original value'''

def standard_units(any_numbers):
    "Convert any array of numbers to standard units."
    return (any_numbers - np.mean(any_numbers)) / np.std(any_numbers)


def correlation(t, x, y):
    return np.mean(standard_units(t.column(x)) * standard_units(t.column(y)))


def slope(table, x, y):
    r = correlation(table, x, y)
    return r * np.std(table.column(y)) / np.std(table.column(x))


def intercept(table, x, y):
    a = slope(table, x, y)
    return np.mean(table.column(y)) - a * np.mean(table.column(x))


def fit(table, x, y):
    a = slope(table, x, y)
    b = intercept(table, x, y)
    return a * table.column(x) + b


def residual(table, x, y):
    return table.column(y) - fit(table, x, y)


def scatter_fit(table, x, y):
    plots.scatter(table.column(x), table.column(y), s=20)
    plots.plot(table.column(x), fit(table, x, y), lw=2, color='gold')
    plots.xlabel(x)
    plots.ylabel(y)


# Helpers Functions

def print_stats(data):
    '''Prints common stats for a data array'''

    data_mean = np.mean(data)
    data_std = np.std(data)
    data_min = min(data)
    data_max = max(data)

    percent_5 = percentile(5, data)
    percent_95 = percentile(95, data)
    percent_1 = percentile(1, data)
    percent_99 = percentile(99, data)

    percent_25 = percentile(25, data)
    percent_50 = percentile(50, data)
    percent_75 = percentile(75, data)

    print("Avg:", data_mean, "\tStd:", data_std, "\tMin:", data_min, "\tMax:", data_max)
    print(" 5%:", percent_5, "\t95%:", percent_95)
    print(" 1%:", percent_1, "\t99%:", percent_99)
    print("25%:", percent_25, "\t50%:", percent_50, '\t75%', percent_75)


def print_col_stats(table, col_name):
    ''' Print the stats For column named'''

    print(col_name, "Stats")
    data = table.column(col_name)
    print_stats(data)


def draw_hist(table: Table, col_name, offset_percent=0):
    ''' Draw a histogram for table with an additional offset percent'''
    data = table.column(col_name)
    offset_start = percentile(offset_percent, data)
    offset_end = percentile(100 - offset_percent, data)
    table.hist(col_name, bins=np.arange(offset_start, offset_end, (offset_end - offset_start) / 20))


def col_stats(table, col_name):
    ''' Prints state for a column in table'''
    print_col_stats(table, col_name)
    draw_hist(table, col_name)
