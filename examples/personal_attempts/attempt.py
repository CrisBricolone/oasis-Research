import asyncio
import os

from camel.models import ModelFactory, ModelManager
from camel.types import ModelPlatformType
from camel.messages import BaseMessage

import oasis
from oasis import (ActionType, LLMAction, ManualAction, generate_twitter_agent_graph, Platform)
from PIL import Image


async def main():
    vllm_model_1 = ModelFactory.create(
        model_platform=ModelPlatformType.VLLM,
        model_type='Qwen/Qwen2.5-7B-Instruct-AWQ',
        url = 'http://127.0.0.1:8000/v1',
        api_key='vllm-fun',
        model_config_dict={'temperature': 0.8}
    )

    shared_model_manager = ModelManager(
        models=[vllm_model_1],
        scheduling_strategy='round_robin'
    )

    available_actions = ActionType.get_default_tiktok_actions()
    agent_graph = await generate_twitter_agent_graph(
        profile_path=("../../data/twitter_dataset/anonymous_topic_200_1h/"
                      "False_Business_0.csv"),
        model = shared_model_manager,
        available_actions= available_actions
    )

    #chestii de database pe care nu le inteleg 100%
    db_path = "../../data/twitter_simulation.db"
    os.environ["OASIS_DB_PATH"] = os.path.abspath(db_path)
    if os.path.exists(db_path):
        os.remove(db_path)

    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.TIKTOK,
        database_path=db_path
    )
    await env.reset()

    actions_1 = {}
    actions_1[env.agent_graph.get_agent(0)] = ManualAction(
        action_type=ActionType.CREATE_POST,
        action_args={'content': 'Earth is Flat'}
    )
    await env.step(actions_1)

    actions_2 = {
        agent: LLMAction()
        for _, agent in env.agent_graph.get_agents([2, 3, 5, 7, 9, 11, 13, 15])
    }
    await env.step(actions_2)

    actions_3 = {}
    actions_3[env.agent_graph.get_agent(1)] = ManualAction(
        action_type=ActionType.CREATE_POST,
        action_args={'content': 'Earth is not flat!'}
    )
    await env.step(actions_3)

    actions_4 = {
        agent: LLMAction()
        for _, agent in env.agent_graph.get_agents()
    }
    await env.step(actions_4)
    
    await env.close()

if __name__ == '__main__':
    asyncio.run(main())