import asyncio
import os

from camel.models import ModelFactory, ModelManager
from camel.types import ModelPlatformType
from camel.messages import BaseMessage

import oasis
from oasis import (ActionType, LLMAction, ManualAction, generate_tiktok_agent_graph)

import sys
sys.path.append('../../visualization/twitter_simulation/align_with_real_world/code')
from graph import prop_graph
import random
from datetime import datetime, timedelta

async def main():
    vllm_model_1 = ModelFactory.create(
        model_platform=ModelPlatformType.VLLM,
        model_type='Qwen/Qwen3-8B-AWQ',
        url = 'http://127.0.0.1:8609/v1',
        api_key='vllm-fun',
        model_config_dict={'temperature': 0.1}
    )

    available_actions = [ActionType.CREATE_POST,
                        ActionType.LIKE_POST, 
                        ActionType.FOLLOW,
                        ActionType.REPOST,
                        ActionType.DO_NOTHING]
    #generate tiktok agent graph e cam acelasi lucru cu cel de twitter, nu face nimic in plus in mod special
    agent_graph = await generate_tiktok_agent_graph(
        profile_path=("../../../data/tiktok/processed_tiktok_dataset_700.csv/"),
        model = vllm_model_1,
        available_actions=available_actions
    )

    source_post = 'The Earth is flat as shit, snowflakes!!! 💀💀'

    db_path = "../../../data/tiktok_simulation.db"
    os.environ["OASIS_DB_PATH"] = os.path.abspath(db_path)
    if os.path.exists(db_path):
        os.remove(db_path)

    env = oasis.make(
        agent_graph = agent_graph,
        platform = oasis.DefaultPlatformType.TIKTOK,
        database_path=db_path
    )
    await env.reset()

    #TODO -> pe viitor ar fi bine sa introducem in simulare si cateva dintre postarile anterioare ale userilor
    #-> also ne trb mai multe topic uri pe baza carora sa actioneze !!!
    actions_spread = {}
    actions_spread[env.agent_graph.get_agent(0)] = ManualAction(
        action_type=ActionType.CREATE_POST,
        action_args={'content': source_post}
    )
    await env.step(actions_spread)

    #TODO
    #We need to create our own time engine cu randomised chance of interaction in functie de time step
    #in paper fiecare step = 3 minute reale -> fiecare 20 de pasi schimbam elementul din vector la care ne uitam
    #cam doar asta ar fi pe partea de time step

    #O sa vrem sa facem mai multe simulari cu timpi random, si cu diferite subiecte
    virtual_time = datetime(2026, 9, 24, 8, 0)
    minutes_per_step = 3

    for step in range(0, 60):
        current_time = virtual_time + timedelta(step * minutes_per_step)
        current_hour = current_time.hour

        print(f"\n--- Step {step} | Virtual Time: {current_time.strftime('%H:%M')} ---")
        active_actions = {}

        for agent_id, agent in env.agent_graph.get_agents():
            #TODO -> implementare cu probabilitate pe bune 
            prob = 0.15
            rand_chance = random.random()

            if rand_chance <= prob:
                active_actions[agent] = LLMAction()

        if active_actions:
            env.step(active_actions)


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