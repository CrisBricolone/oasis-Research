import asyncio
import os

from camel.models import ModelFactory, ModelManager
from camel.types import ModelPlatformType
from camel.messages import BaseMessage

import oasis
from oasis import (ActionType, LLMAction, ManualAction, generate_twitter_agent_graph, Platform)
from PIL import Image


async def manual_image_pass(image_path: str, prompt: str = "Please look at the following image and tell me what you think about it"):
    '''
    Takes a local image and transforms it to Base64 pentru a putea fi interpretata de catre model
    '''

    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f'Error: Image was not found at: {image_path}')

    llm_message = BaseMessage.make_user_message(
        role_name='User',
        content=prompt,
        image_list=[img]
    )

    return llm_message



async def main():
    vllm_model_1 = ModelFactory.create(
        model_platform=ModelPlatformType.VLLM,
        model_type='Qwen/Qwen2-VL-2B-Instruct',
        url='http://localhost:8000/v1'
    )

    shared_model_manager = ModelManager(
        models=[vllm_model_1],
        scheduling_strategy='round_robin'
    )

    available_actions = ActionType.get_default_twitter_actions()
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
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=db_path
    )
    await env.reset()

    print('Distribuim o imagine catre toti agentii')
    image_path_to_send = './image.png'
    image_msg = await manual_image_pass(
        image_path=image_path_to_send,
        prompt='Please look at the following image and tell me what you think about it',

    )

    reactii = {}
    if image_msg:
        all_agents = env.agent_graph.get_agents()
        for agent_id, agent_obj in all_agents:
            print(f'\nAgent {agent_id} analizeaza')
            response = await agent_obj.astep(image_msg)
            reactii[agent_id] = response.msg.content

    print("\n--- REACTIILE CENTRALIZATE ALE AGENTILOR ---")
    for aid, text in reactii.items():
        print(f"Agent {aid} a spus: {text}\n")
    
    await env.close()

if __name__ == '__main__':
    asyncio.run(main())