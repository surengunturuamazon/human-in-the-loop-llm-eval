# constants
MODEL_COL_NAME = 'model_name'
TEMPERATURE_COL_NAME = 'temperature'

LABEL_LIST = ['Good', 'Hallucination', 'Toxic Content', 'Low Fluency',
              'Low Factual Consistency']

INPUT_PATH = 'data/new_prompts.csv'
Prompt_Ids = '/home/sagemaker-user/HITL-Demo/HITL_Demo/data/prompt_ids.csv'
Experiment_Path = 'data/model_parameters.json'
OUTPUT_PATH = 'data/labels_and_feedback.csv'
max_tokens_to_sample_claude = 4000