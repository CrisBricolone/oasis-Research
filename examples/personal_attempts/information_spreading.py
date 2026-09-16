import asyncio
import os

from camel.models import ModelFactory, ModelManager
from camel.types import ModelPlatformType
from camel.messages import BaseMessage

import oasis
from oasis import (ActionType, LLMAction, ManualAction, generate_twitter_agent_graph)

import sys
sys.path.append('../../visualization/twitter_simulation/align_with_real_world/code')
from graph import prop_graph

async def main():
    vllm_model_1 = ModelFactory.create(
        model_platform=ModelPlatformType.VLLM,
        model_type='Qwen/Qwen3.5-2B',
        url = 'http://127.0.0.1:8000/v1',
        api_key='vllm-fun',
        model_config_dict={'temperature': 0.1}
    )

    available_actions = [ActionType.LIKE_POST, ActionType.FOLLOW, ActionType.REPOST, ActionType.DO_NOTHING]
    agent_graph = await generate_twitter_agent_graph(
        profile_path=("../../data/twitter_dataset/anonymous_topic_200_1h/"
                      "False_Business_0.csv"),
        model = vllm_model_1,
        available_actions=available_actions
    )

    source_post = 'The Earth is flat as shit, snowflakes!!! 💀💀'

    db_path = "../../data/twitter_simulation.db"
    os.environ["OASIS_DB_PATH"] = os.path.abspath(db_path)
    if os.path.exists(db_path):
        os.remove(db_path)

    env = oasis.make(
        agent_graph = agent_graph,
        platform = oasis.DefaultPlatformType.TIKTOK,
        database_path=db_path
    )
    await env.reset()

    actions_1 = {}
    actions_1[env.agent_graph.get_agent(0)] = ManualAction(
        action_type=ActionType.CREATE_POST,
        action_args={'content': source_post}
    )
    await env.step(actions_1)

    actions_2 = {}
    actions_2[env.agent_graph.get_agent(1)] = ManualAction(
        action_type=ActionType.CREATE_POST,
        action_args={'content': 'Are you dumb? The Earth is flat???'}
    )
    await env.step(actions_2)

    actions_3 = {
        agent: LLMAction()
        # Activate 5 agents with id 1, 3, 5, 7, 9
        for _, agent in env.agent_graph.get_agents([1, 3, 5, 7, 9])
    }

    await env.step(actions_3)

    actions_4 = {}
    actions_4[env.agent_graph.get_agent(2)] = ManualAction(
        action_type=ActionType.REPOST,
        action_args={'post_id': 1}
    )

    await env.step(actions_4)

    await env.close()

    pg = prop_graph(source_post, db_path, viz=True)
    pg.build_graph()
    pg.plot_scale_time()
    pg.plot_depth_time()
    pg.plot_max_breadth_time()

if __name__ == '__main__':
    asyncio.run(main())