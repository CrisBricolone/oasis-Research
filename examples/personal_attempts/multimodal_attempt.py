import asyncio
import os

from camel.models import ModelFactory, ModelManager
from camel.types import ModelPlatformType
from camel.messages import BaseMessage

import oasis
from oasis import (ActionType, LLMAction, ManualAction, generate_twitter_agent_graph)

async def main():
    vllm_model_1 = ModelFactory.create(
        model_platform=ModelPlatformType.VLLM,
        model_type='granite3.2-vision',
        url = 'http://localhost:11434/v1',
        api_key='ollama',
        model_config_dict={
            'temperature': 0.0,
            'max_tokens': 150, 
            'timeout': 3600.0 
        }
    )

    shared_model_manager = ModelManager(
        models=[vllm_model_1],
        scheduling_strategy='round_robin'
    )

    available_actions = ActionType.get_default_twitter_actions() + [ActionType.POST_PHOTO]
    agent_graph = await generate_twitter_agent_graph(
        profile_path=("../../data/twitter_dataset/anonymous_topic_200_1h/"
                    "False_Business_0.csv"),
        model = shared_model_manager,
        available_actions= available_actions
    )

    db_path = "../../data/twitter_simulation.db"
    os.environ["OASIS_DB_PATH"] = os.path.abspath(db_path)
    if os.path.exists(db_path):
        os.remove(db_path)

    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path = db_path
    )

    await env.reset()

    image_path = './image.png'
    actions_1 = {}
    actions_1[env.agent_graph.get_agent(0)] = ManualAction(
        action_type=ActionType.POST_PHOTO,
        action_args={'image_path': image_path}
    )

    await env.step(actions_1)

    agent_1 = env.agent_graph.get_agent(1)
    actions_2 = {
        agent_1: LLMAction()
    }

    await env.step(actions_2)

    await env.close()

if __name__ == '__main__':
    asyncio.run(main())