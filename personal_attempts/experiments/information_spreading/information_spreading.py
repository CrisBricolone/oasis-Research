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
        model_type='Qwen/Qwen3-8B-AWQ',
        url = 'http://127.0.0.1:8609/v1',
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
        for _, agent in env.agent_graph.get_agents()
    }

    await env.step(actions_3)

    actions_4 = {}
    actions_4[env.agent_graph.get_agent(2)] = ManualAction(
        action_type=ActionType.REPOST,
        action_args={'post_id': 1}
    )

    await env.step(actions_4)

    actions_5 = {
        agent: LLMAction()
        for _, agent in env.agent_graph.get_agents()
    }

    await env.step(actions_5)
    
    actions_6 = {
        agent: LLMAction()
        for _, agent in env.agent_graph.get_agents()
    }

    await env.step(actions_6)

    actions_7 = {
        agent: LLMAction()
        for _, agent in env.agent_graph.get_agents()
    }

    await env.step(actions_7)

    await env.close()

    pg = prop_graph(source_post, db_path, viz=False)
    import matplotlib.pyplot as plt

    try:
        pg.build_graph()

        t_scale, y_scale = pg.plot_scale_time()
        t_depth, y_depth = pg.plot_depth_time()
        t_breadth, y_breadth = pg.plot_max_breadth_time()

        fig, axes = plt.subplots(3, 1, figsize=(5, 9))

        #Grafic 1: Scale
        axes[0].plot(t_scale, y_scale, color='blue', linewidth=2)
        axes[0].set_title("Propagation Scale-Time")
        axes[0].set_xlabel("Time/minute")
        axes[0].set_ylabel("Scale (Users)")
        axes[0].grid(True, linestyle='--', alpha=0.7)

        #Grafic 2: Depth
        axes[1].plot(t_depth, y_depth, color='red', linewidth=2)
        axes[1].set_title("Propagation Depth-Time")
        axes[1].set_xlabel("Time/minute")
        axes[1].set_ylabel("Depth")
        axes[1].grid(True, linestyle='--', alpha=0.7)

        #Grafic 3: Max Breadth
        axes[2].plot(t_breadth, y_breadth, color='green', linewidth=2)
        axes[2].set_title("Propagation Max Breadth-Time")
        axes[2].set_xlabel("Time/minute")
        axes[2].set_ylabel("Max Breadth")
        axes[2].grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

        pg.viz = True
        pg.viz_graph(time_threshold=999)

    except Exception as e:
        print(f'Could not show graph: {e}')

if __name__ == '__main__':
    asyncio.run(main())