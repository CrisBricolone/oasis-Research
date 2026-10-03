import pandas as pd
import numpy as np
 
INPUT_CSV = "../../../data/tiktok/processed_tiktok_dataset_700_with_vectors.csv"
OUTPUT_CSV = "../../../data/tiktok/polarization_dataset_700_conservative.csv"
 
NUM_CORE_USERS = 267  # matches the paper's core-user count for this experiment
 
conservative_prompts = [
    " You hold politically conservative views: you value tradition, caution, and established norms, and tend to be skeptical of risky or unconventional choices.",
    " You are politically conservative in outlook — you prefer stability and proven approaches over risky change, and you value personal responsibility and careful decision-making.",
    " You lean conservative in your political and social views, favoring low-risk choices and respect for established institutions and traditions.",
    " Politically, you are conservative: you're cautious about change, prioritize personal responsibility, and prefer tried-and-tested approaches over experimentation.",
]
 
def add_conservative_framing(text, seed):
    rng = np.random.RandomState(seed)
    return str(text) + conservative_prompts[rng.randint(len(conservative_prompts))]
 
def main():
    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
 
    if 'followers_count' not in df.columns:
        raise ValueError(
            "Expected a 'followers_count' column to rank core users by "
            "follower count — check the input CSV's schema."
        )
 
    n_core = min(NUM_CORE_USERS, len(df))
    if n_core < NUM_CORE_USERS:
        print(f"Warning: only {len(df)} agents available, using all of them as core users.")
 
    core_idx = df.sort_values('followers_count', ascending=False).index[:n_core]
 
    df.loc[core_idx, 'description'] = [
        add_conservative_framing(df.loc[i, 'description'], i) for i in core_idx
    ]
 
    if 'user_char' in df.columns:
        df.loc[core_idx, 'user_char'] = [
            add_conservative_framing(df.loc[i, 'user_char'], i) for i in core_idx
        ]
 
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done. Wrote {len(df)} agents ({n_core} with conservative framing) to '{OUTPUT_CSV}'.")
 
    sample_idx = core_idx[0]
    print("\nSample check (first core agent):")
    print(f"  description: {df.loc[sample_idx, 'description']}")
 
if __name__ == "__main__":
    main()