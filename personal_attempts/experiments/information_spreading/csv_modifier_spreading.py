import pandas as pd
import numpy as np

def prepare_tiktok_dataset_for_oasis(input_csv_path, output_csv_path, max_agents=None):
    print(f"Citim datele din {input_csv_path}...")
    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"Eroare la citirea fișierului: {e}")
        return


    processed_df = pd.DataFrame()
    processed_df['username'] = df['account_id'].fillna('unknown_user')
    processed_df['name'] = df['nickname'].fillna('TikTok User')
    processed_df['description'] = df['biography'].fillna('No bio available.')
    processed_df['user_char'] = "TikTok user. Bio: " + processed_df['description']
    processed_df['followers_count'] = pd.to_numeric(df['followers'], errors='coerce').fillna(0).astype(int)
    processed_df['following_agentid_list'] = '[]'
    processed_df['previous_tweets'] = '[]'
    processed_df = processed_df[processed_df['username'] != 'unknown_user'].reset_index(drop=True)

    if max_agents is not None and max_agents < len(processed_df):
        processed_df = processed_df.sort_values(by='followers_count', ascending=False)
        processed_df = processed_df.head(max_agents).reset_index(drop=True)
        processed_df = processed_df.sample(frac=1).reset_index(drop=True)

    processed_df.to_csv(output_csv_path, index=True, index_label='Unnamed: 0')
    print(f"Gata! Dataset-ul procesat a fost salvat în '{output_csv_path}' cu {len(processed_df)} agenți.")


if __name__ == "__main__":
    INPUT_FILE = "../../../data/tiktok/TikTok profiles dataset (Public web data).csv"  
    OUTPUT_FILE = "../../../data/tiktok/processed_tiktok_dataset.csv"
    
    prepare_tiktok_dataset_for_oasis(INPUT_FILE, OUTPUT_FILE, max_agents=None)