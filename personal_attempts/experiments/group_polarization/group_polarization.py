
import asyncio
import os
import sqlite3
import glob
import json

from camel.models import ModelFactory, ModelManager
from camel.types import ModelPlatformType
from camel.messages import BaseMessage

import oasis
from oasis import (ActionType, LLMAction, ManualAction, generate_twitter_agent_graph)

import pandas as pd

def extract_interviews_from_db(db_path: str, output_dir = './df_opinions'):
    print('Extragem interview-urile din baza de date')

    os.makedirs(output_dir, exist_ok=True)
    old_files = glob.glob(os.path.join(output_dir, "opinions_round_*.csv"))
    for file_path in old_files:
        os.remove(file_path)
    if old_files:
        print(f"Am sters {len(old_files)} fisiere vechi din rularile trecute.")

    conn = sqlite3.connect(db_path)
    query = "SELECT user_id, created_at, info FROM trace WHERE action = 'INTERVIEW' COLLATE NOCASE"
    df_traces = pd.read_sql(query, conn)
    conn.close()

    if df_traces.empty:
        print('NO interviews in db')
        return

    time_steps = sorted(df_traces['created_at'].unique())

    for round_num, t_step in enumerate(time_steps):
        df_step = df_traces[df_traces['created_at'] == t_step]
        user_ids, opinions= [], []

        for _, row in df_step.iterrows():
            user_ids.append(row['user_id'])
            try:
                info_dict = json.loads(row['info'])
                response_text = info_dict.get('response', '')
            except Exception as e:
                response_text = f'Failed to get information: {e}'

            opinions.append(response_text)

            file_name = f'opinions_round_{round_num * 2}.csv'
            output_csv = os.path.join(output_dir, file_name)
            df_out = pd.DataFrame({'user_id': user_ids, 'content': opinions})
            df_out.to_csv(output_csv, index=False)


async def main():
    vllm_model_1 = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type='Qwen3.8-27B',
        url = 'http://127.0.0.1:8609/v1',
        api_key='vllm-fun',
        model_config_dict={'temperature': 0.3}
    )

    # vllm_model_2 = ModelFactory.create(
    #     model_platform=ModelPlatformType.VLLM,
    #     model_type='Qwen/Qwen3-8B-AWQ',
    #     url = 'http://127.0.0.1:8001/v1',
    #     api_key='vllm-fun',
    #     model_config_dict={'temperature': 0.0}
    # )

    available_actions = [ActionType.LIKE_POST, ActionType.DISLIKE_POST, ActionType.FOLLOW,
                      ActionType.REPOST, ActionType.CREATE_COMMENT, ActionType.LIKE_COMMENT,
                      ActionType.DISLIKE_COMMENT, ActionType.DO_NOTHING]
    agent_graph = await generate_twitter_agent_graph(
        profile_path=("data/twitter_dataset/anonymous_topic_200_1h/"
                      "False_Business_0.csv"),
        model = vllm_model_1,
        available_actions=available_actions
    )

    db_path = "data/tiktok_simulation.db"
    os.environ["OASIS_DB_PATH"] = os.path.abspath(db_path)
    if os.path.exists(db_path):
        os.remove(db_path)

    env = oasis.make(
        agent_graph = agent_graph,
        platform = oasis.DefaultPlatformType.TIKTOK,
        database_path=db_path
    )
    await env.reset()

    source_post = 'Should Halen take the risk to write a great novel, or should he continue writing ordinary novels without taking any risks?'

    question = 'What should Halen do?'

    actions_1 = {}
    actions_1[env.agent_graph.get_agent(0)] = ManualAction(
        action_type=ActionType.CREATE_POST,
        action_args={'content': source_post}
    )
    await env.step(actions_1)

    actions_agents = {
        agent: LLMAction()
        for _, agent in env.agent_graph.get_agents([1, 3, 5, 7, 9])
    }

    await env.step(actions_agents)

    actions_interviews = {}
    for _, agent in env.agent_graph.get_agents([1, 3, 5, 7, 9]):
        agent.interview_record = False
        actions_interviews[agent] = ManualAction(
            action_type=ActionType.INTERVIEW,
            action_args={"prompt": question} 
        )

    #base opinions

    total_steps = 2
    interview_cnt = 1
    for i in range(0, total_steps + 1):
        if not ((i + 1) % interview_cnt):
            print(f'Colectam opiniile dupa runda {i + 1}\n')
            await env.step(actions_interviews)

        await env.step(actions_agents)

    await env.close()

    #trimitem rezultatul la interview-uri in df (asa il prelucreaza ei in group_polarization_eval si o sa urmez oarecum ce fac ei acolo)
    extract_interviews_from_db(db_path)

if __name__ == '__main__':
    asyncio.run(main())