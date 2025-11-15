import os
import pandas as pd
from utils.load_json import load_jsonl
from sklearn.model_selection import train_test_split

from utils.extra import (
    pokemon_base_stats_nested,
    pokemon_embeddings,
    META_THREATS_GEN1,
    STATUS_MOVES,
    SETUP_MOVES,
    TYPE_CHART_GEN1,
    POKEMON_RANKING,
    POKEMON_LIST,
    STATUS_WEIGHTS
)

from utils.functions import (
    extract_base_stats,
    get_type_effectiveness,
    encode_pokemon_set,
    get_pokemons_seen_in_battle,
    get_base_stats,
    get_all_pokemons_used,
    safe_get,
    extract_levels,
    check_missing,
    get_best_stab_advantage
)

from Features.features_olya import create_advanced_features_gen2
from Models.pipeline import get_pipeline
from Submission import submit

from paramethers.cat_grid import param_grid as catboost_param_grid
from paramethers.gb_grid import param_grid as gradientboost_param_grid
from paramethers.lgb_grid import param_grid as lightgbm_param_grid
from paramethers.log_grid import param_grid as logistic_param_grid
from paramethers.rf_grid import param_grid as randomforest_param_grid
from paramethers.xgb_grid import param_grid as xgboost_param_grid


from optimisers.gridsearch_optimizer import run_grid_search
from optimisers.optuna_optimizer import optimize_optuna
from optimisers.randomsearch_optimizer import run_random_search


numerical_features = [
    'lead_speed_diff',
    'hp_advantage_seen','mons_revealed_diff','team_status_diff','end_boost_diff',
    'total_damage_dealt','total_healing_done','status_turns',
    'first_faint_turn','total_stats_diff','damage_diff_turn10',
    'damage_diff_turn20','damage_diff_turn25','damage_diff_turn30',
    'hp_trend_diff','feat_switch_diff','feat_aggression_diff','hp_diff_std',
    'hp_diff_range','momentum_shift_turn','comeback_score','early_sustain',
    'status_balance','boost_volatility','boost_trend','move_power_diff',
    'move_diversity_diff','stall_ratio','aggression_index',
    'stats_speed_interaction',
    'hp_vs_stats_ratio','damage_ratio_turn25_30','damage_ratio_turn20_25',
    'damage_ratio_turn10_20','damage_ratio_turn10_30',
    'atk_def_ratio_p1','atk_def_ratio_p2','hp_speed_interaction_lead','hp_def_ratio_p1',
    'hp_def_ratio_p2','p1_hp_mean','p2_hp_mean','hp_diff_mean','hp_diff_last',
    'p1_boost_mean','p2_boost_mean','boost_diff_mean','p1_status_total',
    'p2_status_total','momentum_flips','p1_aggression','p2_aggression',
    'aggression_diff','feat_team_emb_sim',
    'lead_type_adv','meta_diff','feat_status_diff_inflicted','status_setup_diff',
    
]


categorical_features = [
    'feat_team_emb_sim',  # similarity metric between seen Pokémon sets (technically numeric but derived from categorical sets)
    # Other categorical info like moves/types is encoded elsewhere, not direct in the df
]


embedding_features = [
    # Lead Pokémon embeddings (6 stats each)
    'p1_lead_hp', 'p1_lead_atk', 'p1_lead_def', 'p1_lead_spa', 'p1_lead_spd', 'p1_lead_spe',
    'p2_lead_hp', 'p2_lead_atk', 'p2_lead_def', 'p2_lead_spa', 'p2_lead_spd', 'p2_lead_spe',
    
    # Team embeddings: each team gets 3*embedding_dim features (mean/max/min concatenated)
    # Assuming embedding_dim=6, so 18 per team
    'p1_team_emb_0','p1_team_emb_1','p1_team_emb_2','p1_team_emb_3','p1_team_emb_4','p1_team_emb_5',
    'p1_team_emb_6','p1_team_emb_7','p1_team_emb_8','p1_team_emb_9','p1_team_emb_10','p1_team_emb_11',
    'p1_team_emb_12','p1_team_emb_13','p1_team_emb_14','p1_team_emb_15','p1_team_emb_16','p1_team_emb_17',
    'p2_team_emb_0','p2_team_emb_1','p2_team_emb_2','p2_team_emb_3','p2_team_emb_4','p2_team_emb_5',
    'p2_team_emb_6','p2_team_emb_7','p2_team_emb_8','p2_team_emb_9','p2_team_emb_10','p2_team_emb_11',
    'p2_team_emb_12','p2_team_emb_13','p2_team_emb_14','p2_team_emb_15','p2_team_emb_16','p2_team_emb_17',
]


import os
import pandas as pd
from utils.load_json import load_jsonl
from sklearn.model_selection import train_test_split
from Features.features_olya import create_advanced_features_gen2

# def load_data(data_dir=None):
#     """
#     Loads train/test JSONL, applies feature engineering, and splits train/val.

#     Args:
#         data_dir: path to the Data folder (if None, uses current script folder + 'Data')
#         test_size: fraction for validation split
#         random_state: seed for reproducibility

#     Returns:
#         X_train_split, X_val_split, y_train_split, y_val_split, X_test_features
#     """
#     if data_dir is None:
#         data_dir = os.path.join(os.path.dirname(__file__), "Data")

#     train_path = os.path.join(data_dir, "train.jsonl")
#     test_path = os.path.join(data_dir, "test.jsonl")

#     train_df = load_jsonl(train_path)
#     test_df = load_jsonl(test_path)
    
#     #####
#     riga_da_rimuovere = 4877

#     # Usiamo un controllo per sicurezza, nel caso la riga non esista
#     if riga_da_rimuovere in train_df.index:
#         train_df = train_df.drop(riga_da_rimuovere)
#         print(f"Riga {riga_da_rimuovere} rimossa con successo.")
#     else:
#         print(f"Riga {riga_da_rimuovere} non trovata (forse già rimossa o non presente).")

#     filtro_livello_100 = train_df['p1_team_details'].apply(
#         lambda team_list: all(pokemon.get('level') == 100 for pokemon in team_list)
#     )

#     train_df = train_df[filtro_livello_100]

#     print("✓ train.jsonl loaded successfully. Shape:", train_df.shape)
#     print("✓ test.jsonl loaded successfully. Shape:", test_df.shape)


#     return train_df, test_df

def load_data(data_dir=None):
    """
    Loads train/test JSONL data.
    
    This function is environment-aware. It will automatically detect
    if it's running in Kaggle and use the standard Kaggle input path.
    Otherwise, it will fall back to a local 'Data' folder.
    """
    
    KAGGLE_INPUT_PATH = "/kaggle/input/fds-pokemon-battle-data"
    LOCAL_INPUT_PATH = "Data" # The local folder your project uses
    
    if data_dir is None:
        if os.path.exists(KAGGLE_INPUT_PATH):
            print(f"✓ Kaggle environment detected. Loading data from: {KAGGLE_INPUT_PATH}")
            data_dir = KAGGLE_INPUT_PATH
        else:
            print(f"✓ Local environment detected. Loading data from: {LOCAL_INPUT_PATH}")
            data_dir = LOCAL_INPUT_PATH

    train_path = os.path.join(data_dir, "train.jsonl")
    test_path = os.path.join(data_dir, "test.jsonl")

    if not os.path.exists(train_path):
        print(f"Error: train.jsonl not found at {train_path}")
        return pd.DataFrame(), pd.DataFrame()
    if not os.path.exists(test_path):
        print(f"Error: test.jsonl not found at {test_path}")
        return pd.DataFrame(), pd.DataFrame()

    train_df = load_jsonl(train_path)
    test_df = load_jsonl(test_path)
    
    #####
    riga_da_rimuovere = 4877

    # Usiamo un controllo per sicurezza, nel caso la riga non esista
    if riga_da_rimuovere in train_df.index:
        train_df = train_df.drop(riga_da_rimuovere)
        print(f"Riga {riga_da_rimuovere} rimossa con successo.")
    else:
        print(f"Riga {riga_da_rimuovere} non trovata (forse già rimossa o non presente).")

    filtro_livello_100 = train_df['p1_team_details'].apply(
        lambda team_list: all(pokemon.get('level') == 100 for pokemon in team_list)
    )

    train_df = train_df[filtro_livello_100]
    
    print("✓ train.jsonl loaded successfully. Shape:", train_df.shape)
    print("✓ test.jsonl loaded successfully. Shape:", test_df.shape)

    return train_df, test_df


    '''
    
    pipeline_rf = get_pipeline('random_forest', rf_features)
    pipeline_log = get_pipeline('logistic', log_features)
    pipeline_xgb = get_pipeline('xgboost', xgb_features)
    pipeline_lgb = get_pipeline('lightgbm', lgb_features)
    pipeline_cat = get_pipeline('catboost', cat_features)
    pipeline_gb = get_pipeline('gradient_boost', gb_features)

    '''
