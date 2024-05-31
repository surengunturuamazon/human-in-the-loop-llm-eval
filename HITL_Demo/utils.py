import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
import streamlit as st


def get_unique_settings(df, model_col_name, temperature_col_name):
    """
    Obtain all unique settings in the model_parameter logs,
        example, all unique model_names and temperatures.
    Args: 
        df(Pandas Dataframe): model_parameter list in Pandas
        model_col_name(str): column name that represents the model
        temperature_col_name(str): column name that represents the temp
    Return: 
        Pandas DataFrame: unique model/temperature combos in a df
    """
    settings = df.drop_duplicates(subset=[model_col_name,
                                          temperature_col_name]).\
        to_dict("records")
    return settings


def generate_buttons_for_setting_selection_window(df, model_col_name,
                                                  temperature_col_name):
    """
    Generate model and parameter drowdown windows for selection. 
    Args: 
        df(Pandas Dataframe): model_parameter list in Pandas
        model_col_name(str): column name that represents the model
        temperature_col_name(str): column name that represents the temp
    Return: 
        list: setting names
    """
    settings = get_unique_settings(df, model_col_name, temperature_col_name)
    setting_names = []
    for setting in settings:
        model_id = setting[model_col_name]
        temperature = setting[temperature_col_name]
        setting_names.append(f'Model: {model_id}, Temperature: {temperature}')
    return ["None"] + setting_names


def get_model_dataset_temperature(experiment_setting_string):
    """
    Parse the selected model and parameter settings and save into a list. 
    Args: 
        experiment_setting_string(str): selected user model and temperature
    Return: 
        list: the selected model and parameter
    """
    values = []
    matches = re.findall(r": (.*?)(?:,|\s|$)", experiment_setting_string)

    for match in matches:
        values.append(match.strip())
    return values


def parse_model_setting_string(model_setting_string, prompt_1, prompt_2):
    """
    Parse model and parameter settings into independent strings
    Args:
        model_setting_string(str): model/param settings selected by the user
        prompt_1(str): instruction 1 to test
        prompt_2(str): instruction 2 to test
    Returns: 
        dict: prompts and model/param result
    """
    pattern = r"(\w+)\s*:\s*(.*?)(?:,|$)"
    matches = re.findall(pattern, model_setting_string)
    result = {}
    for match in matches:
        key = match[0]
        value = match[1]
        result[key] = value

    result['prompt A'] = prompt_1
    result['prompt B'] = prompt_2
    return result


def draw_bar_chart(label_dict, title, y_label):
    """
    Draw bar chart for label_dict keys and values
    Args: 
        label_dict(dict): dictionary of assigned labels
        title(str): title of the bar chart
        y_label(str): y-axis label
    Return: 
        None
    """
    # draw bar chart for assigned labels
    categories = list(label_dict.keys())
    values = list(label_dict.values())

    # Create the bar chart
    fig, ax = plt.subplots()
    ax.bar(categories, values)

    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Format y-axis tick labels as integers
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xticks(rotation=90)
    st.pyplot(fig)


def draw_pie_chart(label_dict, title=None):
    """
    Draw pie chart for label_dict keys and values
    Args:
        label_dict(dict): dictionary of assigned labels
        title(str): title of the pie chart
    Return:
        None
    """
    # draw pie chart for assigned labels
    labels = list(label_dict.keys())
    sizes = list(label_dict.values())
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    st.pyplot(fig)