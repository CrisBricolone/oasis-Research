import sys
import matplotlib
matplotlib.use('Agg') 
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]   # .../oasis-Research
DATA_DIR = PROJECT_ROOT / "data"

sys.path.append(str(PROJECT_ROOT / "visualization" / "twitter_simulation" / "align_with_real_world" / "code"))
from graph import prop_graph

def main():
    post_index = [1, 2, 4, 5, 6]

    posts_propagation = str(DATA_DIR / "twitter_dataset" / "multimodal" / "fakeedit_6.csv")
    df_posts = pd.read_csv(posts_propagation)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 14))
    colors = ['blue', 'red', 'green', 'purple', 'orange']

    for i, post_idx in enumerate(post_index):
        if post_idx == 1:
            db_path = str(DATA_DIR / "twitter_simulation.db")
            if not Path(db_path).exists():
                db_path = str(DATA_DIR / "twitter_simulation_post_1.db")
        else:
            db_path = str(DATA_DIR / f"twitter_simulation_post_{post_idx}.db")
        
        if not Path(db_path).exists():
            print(f"Baza de date lipseste pentru postarea {post_idx}: {db_path}. Se ignora.")
            continue

        row_idx = post_idx - 1
        source_post = str(df_posts.iloc[row_idx].get('title', df_posts.iloc[row_idx].get('clean_title', '')))

        pg = prop_graph(source_post, db_path, viz=False)

        try:
            pg.build_graph()

            t_scale, y_scale = pg.plot_scale_time()
            t_depth, y_depth = pg.plot_depth_time()
            t_breadth, y_breadth = pg.plot_max_breadth_time()

            color = colors[i]
            label_name = f"Post {i + 1}"

            axes[0].plot(t_scale, y_scale, color=color, linewidth=2, label=label_name)
            axes[1].plot(t_depth, y_depth, color=color, linewidth=2, label=label_name)
            axes[2].plot(t_breadth, y_breadth, color=color, linewidth=2, label=label_name)

            plt.figure(figsize=(30, 20)) 
            pg.viz = True
            pg.viz_graph(time_threshold=999) 
            viz_fig = plt.gcf() 
            
            output_dir = SCRIPT_DIR / "graphs"
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            viz_save_path = output_dir / f"propagation_tree_post_{post_idx}_{timestamp}.png"
            viz_fig.savefig(str(viz_save_path), dpi=400, bbox_inches='tight')
            print(f"Arborele detaliat pentru postarea {post_idx} salvat la: {viz_save_path}")
            
            plt.close(viz_fig) 
  

        except Exception as e:
            print(f"ERROR {post_idx}: {e}")

    # Finalizam si salvam graficul combinat cu cele 3 metrici
    titles = ["Propagation Scale", "Propagation Depth", "Propagation Max Breadth"]
    ylabels = ["Scale (Users)", "Depth", "Max Breadth"]

    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_title(title)
        ax.set_xlabel("Steps")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend() 

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = output_dir / f"combined_metrics_stats_{timestamp}.png"
    
    # fig este figura definita la inceput de tot
    fig.savefig(str(save_path), dpi=300, bbox_inches='tight')
    print(f"\nGraficul combinat pe metrici a fost salvat cu succes la: {save_path}")

if __name__ == '__main__':
    main()