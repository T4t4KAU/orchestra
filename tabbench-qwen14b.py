from orchestra_agents import *

import os
import agentscope

from eval_wikitq.sys_prompt import *



dataset = pd.read_csv('dataset/TabBench/tabbench_test.tsv', sep='\t')

gpt_model = "qwen2.5-14b-instruct"
program = 'sql-py'
template = 'two-agent-template'
line_limit = float('inf')

####
# Setup LLM params
# Use same models by default;
# may use different LLM models, e.g., coding agent may use qwen coder
model_configs = [
    {
        "config_name": "reasoning_agent",
        "model_type": "dashscope_chat",
        "model_name": gpt_model,
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),  # Load from env
        "generate_args": {
            "temperature": 0.7,
        },
    },
    {
        "config_name": "coding_agent",
        "model_type": "dashscope_chat",
        "model_name": gpt_model,
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),  # Load from env
        "generate_args": {
            "temperature": 0.7,
        },
    }
]

agentscope.init(
    model_configs=model_configs,  # model_config,
    project="TQA 3 agent",
    logger_level="ERROR"
)

with open('dataset/WikiTableQuestions/few-shot-demo/reasoning-agent-prompt.txt') as file:
    reasoner_pt = file.read()

with open('dataset/WikiTableQuestions/few-shot-demo/coding-agent-prompt.txt') as file:
    coder_pt = file.read()

few_shot_pt_bank = {
    "reasoner_pt": reasoner_pt,
    "coder_pt": coder_pt
}

def parallel_func(i):
    max_retry = 3
    while max_retry > 0:
        try:
            tqa_solver = ThreeAgent(
                f'prompt_template/{template}.json',
                dataset.iloc[i]['id'],
                dataset.iloc[i]['utterance'],
                dataset.iloc[i]['context'],
                dataset.iloc[i]['targetValue'],
                base_path='dataset/TabBench/',
                demo_file=f'few-shot-demo/WikiTQ-{program}.json',
                sep="\t",   # \t for TabBench
                line_limit=line_limit,
                sys_pt_bank=system_pt_bank,
                few_shot_pt_bank=few_shot_pt_bank
            )
            tqa_solver.get_gpt_prediction_majority_vote(repeat_times=5)
            log_3agents, log_2agents = tqa_solver.get_log_dict()
            break
        except Exception as e:
            log_3agents = {
                'id': dataset.iloc[i]['id'],
                'uncaught_err': str(e)
            }
            log_2agents = {
                'id': str(dataset.iloc[i]['id']),
                'uncaught_err': str(e)
            }
            if "model's maximum context length" in str(e):
                return log_3agents, log_2agents
            if "DataInspectionFailed" in str(e):
                return log_3agents, log_2agents
            max_retry -= 1
    return log_3agents, log_2agents


n_threads = 1
maxLimit = float('inf')

output_result_file_3 = f'dataset/TabBench/results/3_agents/model_{gpt_model.split("/")[-1]}_limit{maxLimit}.json'
output_result_file_2 = f'dataset/TabBench/results/2_agents/model_{gpt_model.split("/")[-1]}_limit{maxLimit}.json'

log_3agent_list = []
log_2agent_list = []

from eval_tabbench import *
metric_eval_engine = QAMetric()

for i in tqdm(range(min(maxLimit, dataset.shape[0]))):
    log_3agents, log_2agents = parallel_func(i)
    if "uncaught_err" in log_3agents:
        print(f"uncaught error qid-{i+1}")
        continue
    log_3agent_list.append(log_3agents)
    log_2agent_list.append(log_2agents)

    if (i+1) % 10 == 0:
        tmp_result_file3 = f'dataset/TabFact/results/3_agents/tmp/model_{gpt_model}_{i+1}.json'
        json.dump(log_3agent_list, open(tmp_result_file3, 'w'), indent=4)

        ref_2agent = []
        pred_2agent = []
        for log_2agents in log_2agent_list:
            if 'predicted_value' in log_2agents:
                ref_2agent.append(log_2agents['target_value'])
                pred_2agent.append(log_2agents['predicted_value'])
            else:
                continue
        metric_scores_2 = metric_eval_engine.compute(references=ref_2agent, predictions=pred_2agent)

        ref_3agent = []
        pred_3agent = []
        for log_3agents in log_3agent_list:
            if 'predicted_value' in log_3agents:
                ref_3agent.append(log_3agents['target_value'])
                pred_3agent.append(log_3agents['predicted_value'])
            else:
                continue
        metric_scores_3 = metric_eval_engine.compute(references=ref_3agent, predictions=pred_3agent)

        print("------=")
        print(f"Time: ", datetime.datetime.now())
        print(f"Output file {tmp_result_file3}")
        print(f"Accuracy 2 Agent: {metric_scores_2}")
        print(f"Accuracy 3 Agent: {metric_scores_3}")
        print("------")

json.dump(log_3agent_list, open(output_result_file_3, 'w'), indent=4)


# evaluate:
ref_2agent = []
pred_2agent = []
for log_2agents in log_2agent_list:
    if 'predicted_value' in log_2agents:
        ref_2agent.append(log_2agents['target_value'])
        pred_2agent.append(log_2agents['predicted_value'])
    else:
        continue
metric_scores_2 = metric_eval_engine.compute(references=ref_2agent, predictions=pred_2agent)

ref_3agent = []
pred_3agent = []
for log_3agents in log_3agent_list:
    if 'predicted_value' in log_3agents:
        ref_3agent.append(log_3agents['target_value'])
        pred_3agent.append(log_3agents['predicted_value'])
    else:
        continue
metric_scores_3 = metric_eval_engine.compute(references=ref_3agent, predictions=pred_3agent)

print("======")
print(f"Time: ", datetime.datetime.now())
print(f"Output file {output_result_file_3}")
print(f"Accuracy 2 Agent: {metric_scores_2}")
print(f"Accuracy 3 Agent: {metric_scores_3}")
print("======")
