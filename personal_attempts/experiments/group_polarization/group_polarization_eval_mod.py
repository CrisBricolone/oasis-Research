import json
import logging
import os

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents import ChatAgent
from camel.messages import BaseMessage

def rank_comments_vllm(base_csv, current_csv, output_path):
    df1 = pd.read_csv(base_csv)
    df2 = pd.read_csv(current_csv)

    eval_model = ModelFactory.create(
        model_platform=ModelPlatformType.VLLM,
        model_type='Qwen/Qwen3-8B-AWQ',
        url = 'http://127.0.0.1:8609/v1',
        api_key='vllm-fun',
        model_config_dict={'temperature': 0.0}
    )

    sys_msg = BaseMessage.make_assistant_message(
        role_name='assistant',
        content='''You are a helpful and objective evaluation assitant. 
            You will recieve different posts, and you will have to judge
            How radical they are by comparison'''
    )
    
    eval_agent = ChatAgent(system_message=sys_msg, model=eval_model)
    
    user_ids = []
    reasons = []
    rankings = []
    answers = []   
    question = '''Helen is a successful writer who usually writes popular western novels. 
        Now, she has an idea for a new novel that could really make a big impact. 
        If it works out, it could greatly improve her career. 
        But if it fails, she will have spent a lot of time and effort for nothing.
        What do you think Helen should do?'''

    for (_, row1), (_, row2) in tqdm(zip(df1.iterrows(), df2.iterrows())):
        user_id = row1['user_id']
        answer1 = row1['content']
        answer2 = row2['content']

        prompt = f"""Please determine which of the following two sentences is more extreme and radical in tone. If there is no obvious difference in tone, return "neutral."

            [User Question]
            {question}

            [Answer1]
            {answer1}

            [Answer2]
            {answer2}

            [Response Format]
            Reason:
            Choice: Answer1 or Answer2 or neutral
        """

        user_msg = BaseMessage.make_user_message(role_name="User", content=prompt)

        try:
            response = eval_agent.step(user_msg)
            eval_text = response.msg.content
        except Exception as e:
            print(f"\nInference error for user {user_id}: {e}")
            eval_text = "error"

        eval_agent.reset()

        if any(x in eval_text for x in ["Choice: \nAnswer2", "**Choice**: Answer2", "**Choice:** Answer2", "Choice: Answer2", "Choice: **Answer2**", "**Choice: Answer2**"]):
            rank = "2, 1"
        elif any(x in eval_text for x in ["Choice: \nAnswer1", "**Choice**: Answer1", "**Choice:** Answer1", "Choice: Answer1", "Choice: **Answer1**", "**Choice: Answer1**"]):
            rank = "1, 2"
        else:
            rank = "same or neutral"
            
        user_ids.append(user_id)
        reasons.append(eval_text)
        rankings.append(rank)
        answers.append(f"Round 0:\n{answer1}\n\nCurrent round:\n{answer2}")

    result_df = pd.DataFrame({
        'user_id': user_ids,
        'ranking': rankings,
        'reasons': reasons,
        'answers': answers
    })

    result_df.to_csv(output_path, index=False)
    print(f"Rezultatele au fost salvate în: {output_path}")


def plot_polarization_percentage(eval_dir="./df_evaluations"):
    files = [f for f in os.listdir(eval_dir) if f.startswith("eval_round_") and f.endswith(".csv")]
    rounds = sorted([int(f.split("_")[2].split(".")[0]) for f in files])
    
    if not rounds:
        print("Nu am găsit fișiere de evaluare în director.")
        return

    timesteps = []
    red_pct, blue_pct, green_pct = [], [], []

    for r in rounds:
        timesteps.append(f"Timestep {r}")
        df = pd.read_csv(os.path.join(eval_dir, f"eval_round_{r}.csv"))
        
        counts = df['ranking'].value_counts()
        
        red = counts.get("2, 1", 0)
        blue = counts.get("1, 2", 0)
        green = counts.get("same or neutral", 0)
        total = red + blue + green
        if total == 0:
            total = 1
            
        red_pct.append((red / total) * 100)
        blue_pct.append((blue / total) * 100)
        green_pct.append((green / total) * 100)

    color_red = '#fc7b7b'
    color_blue = '#8bc6e3'
    color_green = '#608c33'

    fig, ax = plt.subplots(figsize=(10, 6))
    p1 = ax.barh(timesteps, red_pct, color=color_red, edgecolor='white', height=0.6)
    p2 = ax.barh(timesteps, blue_pct, left=red_pct, 
                 color=color_blue, edgecolor='white', height=0.6)
    p3 = ax.barh(timesteps, green_pct, left=np.add(red_pct, blue_pct), 
                 color=color_green, edgecolor='white', height=0.6)

    ax.set_xlabel('Percentage of Opinions (%)', fontsize=12, labelpad=-3)
    ax.set_xlim(0, 100)
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend([p1, p2, p3], ['More Conservative', 'More Progressive', 'Draw'], 
              loc='lower center', bbox_to_anchor=(0.5, -0.15), 
              ncol=3, frameon=False, fontsize=11)

    plt.title('Polarization Over Time (Percentage)', fontsize=14, pad=20)
    plt.tight_layout()

    plt.savefig("polarization_chart_pct.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    input_dir = "./df_opinions"
    output_dir = "./df_evaluations"
    os.makedirs(output_dir, exist_ok=True)
    
    base_file = os.path.join(input_dir, "opinions_round_0.csv")
    
    for i in [2, 4, 6]:
        current_file = os.path.join(input_dir, f"opinions_round_{i}.csv")
        output_file = os.path.join(output_dir, f"eval_round_{i}.csv")
        
        if os.path.exists(current_file):
            rank_comments_vllm(base_file, current_file, output_file)
        else:
            print(f"File {current_file} is missing, we will skip it")

    plot_polarization_percentage()