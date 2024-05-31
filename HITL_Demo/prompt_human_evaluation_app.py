import boto3
import json
import streamlit as st
import pandas as pd
from collections import Counter
from utils import generate_buttons_for_setting_selection_window, \
                get_model_dataset_temperature, parse_model_setting_string, \
                draw_bar_chart, draw_pie_chart
from config import MODEL_COL_NAME, TEMPERATURE_COL_NAME, LABEL_LIST, \
                INPUT_PATH, Experiment_Path, OUTPUT_PATH, Prompt_Ids, \
                max_tokens_to_sample_claude

# Bedrock initialization and call
bedrock_client = boto3.client(service_name='bedrock-runtime')

def claude_llm_response(prompt, modelId, temp):
    """
    Use Claude to get an LLM response
    Args: 
        prompt(str): llm prompt
        modelId(str): anthropic claude v2 or claude instant
        temp(str): parameter for the claude model
    Return: 
        str: claude llm output
    """
    body = json.dumps(
        {
            "prompt": prompt,
            "max_tokens_to_sample": max_tokens_to_sample_claude,
            "temperature": temp
        }
    )
    response = bedrock_client.invoke_model(body=body, modelId=modelId)
    response_body = json.loads(response.get("body").read())
    return response_body.get("completion")

# Initializing session state variables

# Human in the Loop Resulting Dataframe
if "result_df" not in st.session_state:
    st.session_state.result_df = \
        pd.DataFrame(columns=['question', 'output', 'labels', 'comment'])

# Accumulated score of whether the first output is better
if 'first_score' not in st.session_state:
    st.session_state.first_score = 0

# Accumulated score of whether the second output is better
if 'sec_score' not in st.session_state:
    st.session_state.sec_score = 0

# Row index from new_prompts.csv
if 'row_index' not in st.session_state:
    st.session_state.row_index = 0

# Feedback from the user
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

# First Prompt Approval Rate
if 'first_prompt_score' not in st.session_state:
    st.session_state.first_prompt_score = 0

# Second Prompt Approval Rate
if 'sec_prompt_score' not in st.session_state:
    st.session_state.sec_prompt_score = 0

# Dictionary of options for the first output (ex: Correct, hallucination, etc..)
if 'first_label_map' not in st.session_state:
    first_counter = Counter(LABEL_LIST)
    st.session_state.first_label_map = \
        Counter({key: value - 1 for key, value in first_counter.items()})

# Dictionary of options for the second output (ex: Correct, hallucination, etc..)
if 'sec_label_map' not in st.session_state:
    sec_counter = Counter(LABEL_LIST)
    st.session_state.sec_label_map = \
        Counter({key: value - 1 for key, value in sec_counter.items()})

# load input json file
df_labels = pd.read_csv(INPUT_PATH)
df_embedding_and_tox = pd.read_json(Experiment_Path)

# create model and parameter selection buttons
modelOptions = generate_buttons_for_setting_selection_window(
    df_embedding_and_tox, MODEL_COL_NAME, TEMPERATURE_COL_NAME)
prompt_ids = pd.read_csv(Prompt_Ids)

# create sidebar in streamlit application
with st.sidebar:
    # user model and parameter (temperature) selection
    model = st.selectbox(
        'Select Model',
        modelOptions)

    # none means no instruction, default is best_prompt from new_prompts.csv
    prompt_options = ['none'] + ['default'] + \
        prompt_ids['Id'].to_list()

    # listen to user prompt A selection
    first_prompt = st.selectbox(
        'Select First Prompt for Comparison',
        prompt_options)

    # listen to user prompt B selection
    sec_prompt = st.selectbox(
        'Select Second Prompt for Comparison',
        prompt_options)

    # start button shown after model/param selected
    if model != 'None':
        model_name, temp = get_model_dataset_temperature(model)
        # create a start button
        start_button = st.button("Start")


def prompt_outputs(row, prompt_tbl, prompt_id):
    """
    Instruction to test
    Args:
        row(Pandas table row): Particular question/answer pair to analyze
        prompt_tbl(Pandas): map prompt indices to instructions
        prompt_id(str): either default, none, or a prompt index
    Return:
        str: instruction to test
    """
    if prompt_id == 'default':
        return row['best_prompt']
    elif prompt_id == 'none':
        return "No Instruction"
    else:
        return prompt_tbl.iloc[prompt_id]['Prompts']


def response_outputs(prompt_output, row):
    """
    Return answer for particular instruction and question
    Args:
        prompt_id(str): either default, none, or a prompt index
        row(Pandas table row): Particular question/answer pair to analyze
    Return:
        str: generated answer
    """
    if prompt_output != 'none':
        # Run Claude AI bot on the new instruction
        text = 'Human:\n\n' + prompt_output + '\n' + f"Question: {row['question']}" + "\n Assistant: "
        response = claude_llm_response(text, model_name, float(temp))
        return response
    return row['answer']


# update the windows
def update_windows(row, prompt_tbl, first_prompt_idx, sec_prompt_idx):
    """
    Update text boxes in the streamlit app, primarily the questions,
        instructions and the outputs.
    Args:
        row(Pandas table row): Particular question/answer pair to analyze
        prompt_tbl(Pandas): map prompt indices to instructions
        first_prompt_id(str): either default, none, or a prompt index for the first output
        sec_prompt_id(str): either default, none, or a prompt index for the second output
    Returns:
        None
    """
    question_window.info(row['question'], icon="🤖")
    first_prompt_output = prompt_outputs(row, prompt_tbl, first_prompt_idx)
    sec_prompt_output = prompt_outputs(row, prompt_tbl, sec_prompt_idx)

    first_prompt_window.info(first_prompt_output)
    sec_prompt_window.info(sec_prompt_output)
    first_response = response_outputs(first_prompt_output, row)
    sec_response = response_outputs(sec_prompt_output, row)
    first_output_window.info(first_response)
    sec_output_window.info(sec_response)


# update prompt approve scores and comparison scores
def update_scores():
    """
    Update all session state variables
    """
    if vote == 'Output A is better':
        st.session_state.first_score += 1
    elif vote == 'Output B is better':
        st.session_state.sec_score += 1

    if first_prompt_score == "Approve":
        st.session_state.first_prompt_score += 1

    if sec_prompt_score == "Approve":
        st.session_state.sec_prompt_score += 1

    for label in first_options:
        st.session_state.first_label_map[label] += 1

    for label in sec_options:
        st.session_state.sec_label_map[label] += 1


# update result analysis table
def update_results(row, is_output_1):
    """
    Update the session state results dataframe
    Args:
        row(Pandas table row): Particular question/answer pair to analyze
        is_output_1(boolean): True if for first output else second output
    """
    first_output = 'placeholder 1'
    sec_output = 'placeholder 2'
    output = first_output if is_output_1 else sec_output
    category = first_options if is_output_1 else sec_options
    comment = first_comment if is_output_1 else sec_comment

    if category or comment:
        new_row = {
            'question': row['question'],
            'output': output,
            'labels': category,
            'comment': comment,
        }
        st.session_state.result_df = pd.concat([st.session_state.result_df,
                                                pd.DataFrame([new_row])])


# set up title and captions of the webpage
st.title("Prompt Human Evaluation")
st.caption("This tool aims to provide an objective method for evaluating \
             the performance \
             of different prompts with a selected large language model. \
             The questions \
             for evaluation are chosen by considering the similarity of model \
             outputs' embeddings and toxicity scores.")

# define question and context window
question_window = st.empty()
question_window.info('Here is the input question ', icon="🤖")

# Prompt Windows
first_prompt_window = st.empty()
first_prompt_window.info('Here is the prompt')
sec_prompt_window = st.empty()
sec_prompt_window.info('Here is the prompt')

# define model output window
first_col, sec_col = st.columns([0.5, 0.5])
first_subheader = "Model Output (Prompt A):"
sec_subheader = "Model Output (Prompt B):"

with first_col:
    st.subheader(first_subheader)
    first_output_window = st.info('Here is the model output from using prompt \
                template A')

with sec_col:
    st.subheader(sec_subheader)
    sec_output_window = st.info('Here is the model output from using prompt \
        template B')

# define evaluation window
first_form = st.form('First Form', clear_on_submit=True)

with first_form:
    vote = st.radio('Vote :thumbsup:',
                    ('Output A is better', 'Tie', 'Output B is better'))
    first_input_col, sec_input_col = st.columns([0.5, 0.5])

    with first_input_col:
        first_prompt_score = st.radio('Do you approve output A?',
                                 ('Approve', 'Disapprove'))
        first_options = st.multiselect(
            'Select labels for output A',
            LABEL_LIST,
            [])
        first_comment = st.text_input(
            "Feedback A",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            placeholder='Your text input for the prompt A',
        )

    with sec_input_col:
        sec_prompt_score = st.radio('Do you approve output B?',
                                 ('Approve', 'Disapprove'))
        sec_options = st.multiselect(
            'Select labels for output B',
            LABEL_LIST,
            [])
        sec_comment = st.text_input(
            "Feedback B",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
            placeholder='Your text input for the prompt B',
        )

    next_button = st.form_submit_button("Save results and Next :arrow_right:")

# create the end buttons
end_button = st.button("End and show scores")

# define start button clicking operation
if model != 'None' and start_button:
    st.session_state.row_index = 0
    st.session_state.first_score = 0
    st.session_state.sec_score = 0
    update_windows(df_labels.iloc[st.session_state.row_index], prompt_ids,
                   first_prompt, sec_prompt)

# Button click actions
if next_button:
    # Get the next row in the dataframe
    update_results(df_labels.iloc[st.session_state.row_index], True)
    update_results(df_labels.iloc[st.session_state.row_index], False)
    update_scores()
    st.session_state.row_index += 1

    if st.session_state.row_index < len(df_labels):
        update_windows(df_labels.iloc[st.session_state.row_index], prompt_ids,
                       first_prompt, sec_prompt)

# define end button clicking operation
if end_button:
    st.divider()
    st.subheader(":white_check_mark: Result Analysis")
    st.markdown("###### Experiment Settings:")
    model_setting_dict = parse_model_setting_string(model,
                                                    "Prompt One", "Prompt Two")
    st.write(model_setting_dict)
    st.markdown("###### Prompt Comparison Result:")
    first_result_col, sec_result_col = st.columns([0.5, 0.5])
    total_score = st.session_state.row_index

    # display prompt comparison scores
    with first_result_col:
        st.write("Prompt A Score:", st.session_state.first_score)
        if total_score:
            st.write("Prompt A approve rate:",
                     str(round(st.session_state.first_prompt_score * 100 / total_score,
                               2)) + '%')

    with sec_result_col:
        st.write("Prompt B Score:", st.session_state.sec_score)
        if total_score:
            st.write("Prompt B approve rate:",
                     str(round(st.session_state.sec_prompt_score * 100 / total_score,
                               2)) + '%')

    st.markdown("###### Label Distribution:")
    result_col3, result_col4 = st.columns([0.5, 0.5])
    y_label = "Count"

    # display label distribution bar charts and pi charts
    with result_col3:
        if sum(st.session_state.first_label_map.values()) > 0:
            draw_bar_chart(st.session_state.first_label_map,
                           "Output A Label Distribution", y_label)
            draw_pie_chart(st.session_state.first_label_map)
        else:
            st.markdown(":red[No Labels Selected!]")

    with result_col4:
        if sum(st.session_state.sec_label_map.values()) > 0:
            draw_bar_chart(st.session_state.sec_label_map,
                           "Output B Label Distribution", y_label)
            draw_pie_chart(st.session_state.sec_label_map)
        else:
            st.markdown(":red[No Labels Selected!]")

    # save the result dataframe
    st.session_state.result_df.to_csv(OUTPUT_PATH, index=False)

# save human interaction
if len(st.session_state.result_df):
    st.markdown("###### Saved Samples:")
    st.write(st.session_state.result_df)