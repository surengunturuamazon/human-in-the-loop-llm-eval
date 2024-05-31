# Human In the Loop And Prompt Evaluation Demo

## Overview
Streamlit is used to display all prompt evaluations from Ragas as well as the human in the loop demo (to efficiently perform thumbs up and thumbs down). 

This streamlit demo is a comprehensive framework that encompasses a wide variety of processes including running automatic RAG evaluations and testing new instructions using a thumbs up/thumbs down approach. This aims to improve the current RAG solution so that it can produce accurate, contextually relevant, and safe outputs. The demonstration will play a crucial role in deploying improved RAG solutions effectively, addressing challenges related to bias, misinformation, and unintended outputs. 

This repository implements some of these features:

1. **Prompt Evaluation**: this implements a prototype tool to visualize and inspect the evaluated prompts using a steamlit app.

2. **Human in the Loop (HITL_Demo)**: this module implements a tool to enable the evaluation of LLMs by humans using a double-blind evaluation. It displays the different versions of outputs based on different settings of LLMs so that a human can compare quality of the outputs. 


### Environment Settings
In order to run the different modules, we used following environment and settings.

```
RAM: 5GB memory 
SageMaker Instance: ml.t3.large
```
Run on a sagemaker domain and create a JupyterLab instance. 

## Setup and Install  
  
1. Clone the repository:  

2. Install dependencies using the [requirements.txt](./requirements.txt):
```
pip install -r requirements.txt
```

## Getting Started  

### Prompt_Evaluation
This folder contains the code for the visualization tool. This tool is designed to make the inspection of the logs easier.  
1. Navigate to the appropriate folder
```
cd Prompt_Evaluation
```
2. Install dependencies(optional):
```
pip install streamlit streamlit-aggrid
```  

3. Update datasources in  [config.py](./Prompt_Evaluation/config.py). Make sure you have ragas_output.csv in the folder (includes question, context, answer, ground truth, and list of metrics and their values). 

4. Start the web app:  
```streamlit run main.py --server.runOnSave true```

5. In the browser, go to this url: `https://{notebook-url}/proxy/8501/`

### HITL Demo
1. Navigate to the appropriate folder
```
cd HITL_Demo
```

2. Install dependencies using [requirements.txt](./HITL_Demo/requirements.txt) (optional):
```
pip install -r requirements.txt
```
#### Input files
---

model_parameters.json
```
A JSON file containing the following keys along with their corresponding data types:

model_name                   string
temperature                  string
```

new_prompts.csv
```
A csv file consisting of prompts/instructions containing the following columns:

question
contexts
answer
ground_truths
context_recall
context_precision
answer_relevancy
faithfulness
answer_similarity
answer_correctness
best_prompt
second_best_prompt
third_best_prompt

This is the output from running Ragas and AutoPrompting. 

```

prompt_ids.csv
```
A csv file consisting of prompts/instructions containing the following columns:

Id
Prompts

```
----

3. Please open [config.py](./HITL_Demo/config.py) and make appropriate changes to the data paths and other settings.

4. For running the prompt human evaluation Streamlit app, please use the command:
```
streamlit run prompt_human_evaluation_app.py --server.runOnSave true
```
Then, you'll see the information in the terminal:
```
  You can now view your Streamlit app in your browser.

  Network URL: http://169.255.254.1:8501
  External URL: http://35.174.73.137:8501
```
Because of the restriction in SageMaker Studio, click the link doesn't open the app automatically. To address this issue, please copy the link of the terminal page and add **/proxy/port_number/** right after **/default/**. Copy and paste the link into a new tap and hit enter. The sample link:
```
https://{notebook-url}/jupyter_lab/default/proxy/8501/
```
In the link, add a model name, and an instruction for prompt one and an instruction for prompt two. When selecting the first and second prompt for instruction, remember that None = no instruction (just the question), Default = autoprompt outputted instruction for the given question, and 0-68 refer to prompts as shown in prompt_ids.csv. 
