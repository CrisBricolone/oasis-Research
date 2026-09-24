import pandas as pd
import json
import numpy as np

df = pd.read_csv("../../../data/tiktok/processed_tiktok_dataset_700.csv")

base_profiles = {
    'student': [0.05, 0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.3, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 0.8, 0.8, 0.7, 0.5, 0.2, 0.1],
    'job':     [0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.2, 0.8, 0.4, 0.2, 0.2, 0.2, 0.7, 0.3, 0.2, 0.2, 0.3, 0.8, 0.9, 0.8, 0.6, 0.4, 0.1, 0.05],
    'influencer': [0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.3, 0.5, 0.6, 0.7, 0.7, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.8, 0.7, 0.6, 0.4, 0.2],
    'default': [0.05, 0.02, 0.01, 0.01, 0.01, 0.05, 0.2, 0.5, 0.4, 0.3, 0.4, 0.5, 0.6, 0.5, 0.5, 0.6, 0.7, 0.8, 0.8, 0.7, 0.6, 0.4, 0.2, 0.1]
}

def randomize(profile, seed):
    np.random.seed(seed)
    return [round(min(1.0, max(0.0, p + np.random.uniform(-0.05, 0.05))), 2) for p in profile]

tz_offsets = {
    'usa': -5, 'ny': -5, 'california': -8, 'texas': -6,
    'uk': 0, 'london': 0, '🇵🇱': 1, 'poland': 1, 'warszawa': 1,
    '🇪🇸': 1, 'spain': 1, 'madrid': 1, '🇩🇪': 1, 'germany': 1,
    '🇫🇷': 1, 'france': 1, '🇮🇹': 1, 'italy': 1, '🇷🇺': 3, 'russia': 3, 'moscow': 3,
    '🇧🇷': -3, 'brazil': -3, '🇲🇽': -6, 'mexico': -6, '🇯🇵': 9, 'japan': 9,
    '🇰🇷': 9, 'korea': 9, '🇨🇳': 8, 'china': 8, '🇮🇳': 5, 'india': 5,
    '🇦🇺': 10, 'australia': 10, '🇦🇷': -3, 'argentina': -3, '🇨🇦': -5, 'canada': -5,
    'philippines': 8, '🇵🇭': 8, 'indonesia': 7, '🇮🇩': 7,
    'turkey': 3, '🇹🇷': 3, 'egypt': 2, '🇪🇬': 2, 'saudi': 3, '🇸🇦': 3, 'uae': 4, '🇦🇪': 4,
}

vectors_list = []

for idx, row in df.iterrows():
    bio = str(row['description']).lower()
    followers = row['followers_count']
    
    profile_type = 'default'
    if followers > 10000:
        profile_type = 'influencer'
    elif any(word in bio for word in ['student', 'school', 'uni', 'college', '15y', '16y', '17y', '18y', '19y', '20y', '21y', 'highschool', 'grad']):
        profile_type = 'student'
    elif any(word in bio for word in ['work', 'job', 'manager', 'ceo', 'founder', 'engineer', 'nurse', 'teacher', 'mom', 'dad']):
        profile_type = 'job'
        
    local_vector = randomize(base_profiles[profile_type], idx)

    offset = 0
    for key, val in tz_offsets.items():
        if key in bio:
            offset = val
            break

    utc_vector = [0.0] * 24
    for utc_hour in range(24):
        local_hour = (utc_hour + offset) % 24
        utc_vector[utc_hour] = local_vector[local_hour]
    vectors_list.append(json.dumps(utc_vector))

df['activity_vector'] = vectors_list
df.to_csv("../../../data/tiktok/processed_tiktok_dataset_700_with_vectors.csv", index=False)

