from tqdm.auto import tqdm
import pandas as pd
import numpy as np 
import sklearn as skl
from utils.extra import (
    pokemon_base_stats_nested,
    pokemon_embeddings,
    META_THREATS_GEN1,
    STATUS_MOVES,
    SETUP_MOVES,
    TYPE_CHART_GEN1,
    POKEMON_RANKING,
    POKEMON_LIST
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

def create_advanced_features(df):
    processed_data = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Creazione features"):
        p1_team = row['p1_team_details']
        p2_lead = row['p2_lead_details']
        timeline = row['battle_timeline']
        p1_lead = p1_team[0]
        
        # --- Static Features ---
        feat_lead_speed_diff = p1_lead['base_spe'] - p2_lead['base_spe']
        feat_lead_hp_diff = p1_lead['base_hp'] - p2_lead['base_hp']
        feat_lead_atk_diff = p1_lead['base_atk'] - p2_lead['base_atk']
        feat_lead_def_diff = p1_lead['base_def'] - p2_lead['base_def']
        feat_lead_spa_diff = p1_lead['base_spa'] - p2_lead['base_spa']
        feat_lead_spd_diff = p1_lead['base_spd'] - p2_lead['base_spd']
        
        # --- Lead Type Advantage ---
        p1_lead_types = [t for t in p1_lead['types'] if t != 'notype']
        p2_lead_types = [t for t in p2_lead['types'] if t != 'notype']
        
        p1_best_adv = get_best_stab_advantage(p1_lead_types, p2_lead_types)
        p2_best_adv = get_best_stab_advantage(p2_lead_types, p1_lead_types)
        
        # A positive number means P1's STABs are more effective
        feat_lead_type_adv_diff = p1_best_adv - p2_best_adv

        # Statistiche aggregate P1
        feat_p1_team_avg_atk = np.mean([p['base_atk'] for p in p1_team])
        feat_p1_team_avg_spe = np.mean([p['base_spe'] for p in p1_team])
        feat_p1_team_max_hp = np.max([p['base_hp'] for p in p1_team])
        
        # Using 'base_spa' as it's identical to 'base_spd' in Gen 1 data
        feat_p1_team_avg_special = np.mean([p['base_spa'] for p in p1_team])
        
        # Meta threats
        feat_p1_team_meta_count = sum(1 for p in p1_team if p['name'].title() in META_THREATS_GEN1)
        feat_p2_lead_is_meta = 1 if p2_lead['name'].title() in META_THREATS_GEN1 else 0
        
        p2_lead_speed = p2_lead['base_spe']
        feat_team_speed_adv_vs_lead = sum(1 for p in p1_team if p['base_spe'] > p2_lead_speed)

        # --- Dynamic Features (Timeline) ---
        p1_seen_status = {p['name']: {'hp_pct': 100, 'status': None} for p in p1_team}
        p2_seen_status = {p2_lead['name']: {'hp_pct': 100, 'status': None}}
        
        feat_end_boost_diff = 0
        p1_status_moves = 0
        p1_setup_moves = 0
        p2_status_moves = 0
        p2_setup_moves = 0
        
        p1_total_bp = 0
        p2_total_bp = 0
        p1_confused_turns = 0
        p2_confused_turns = 0
        p1_active_hp_end = 100 
        p2_active_hp_end = 100 
        
        last_turn_num = 0 

        if timeline:
            last_turn_num = timeline[-1].get('turn', 0)
            for turn in timeline:
                p1_state = turn.get('p1_pokemon_state')
                if p1_state and p1_state.get('name'):
                    p1_name = p1_state['name']
                    p1_seen_status.setdefault(p1_name, {'hp_pct': 100, 'status': None})
                    p1_seen_status[p1_name]['hp_pct'] = p1_state.get('hp_pct', p1_seen_status[p1_name]['hp_pct'])
                    p1_seen_status[p1_name]['status'] = p1_state.get('status', p1_seen_status[p1_name]['status'])
                    
                    if 'confusion' in p1_state.get('volatile_effects', []):
                        p1_confused_turns += 1
                    
                p2_state = turn.get('p2_pokemon_state')
                if p2_state and p2_state.get('name'):
                    p2_name = p2_state['name']
                    p2_seen_status.setdefault(p2_name, {'hp_pct': 100, 'status': None})
                    p2_seen_status[p2_name]['hp_pct'] = p2_state.get('hp_pct', p2_seen_status[p2_name]['hp_pct'])
                    p2_seen_status[p2_name]['status'] = p2_state.get('status', p2_seen_status[p2_name]['status'])
                    
                    if 'confusion' in p2_state.get('volatile_effects', []):
                        p2_confused_turns += 1

                p1_move = turn.get('p1_move_details')
                if p1_move:
                    move_name_p1 = p1_move.get('name', '').title()
                    if move_name_p1 in STATUS_MOVES: p1_status_moves += 1
                    if move_name_p1 in SETUP_MOVES: p1_setup_moves += 1
                    if p1_move.get('base_power'):
                        p1_total_bp += p1_move['base_power']
                    
                p2_move = turn.get('p2_move_details')
                if p2_move:
                    move_name_p2 = p2_move.get('name', '').title()
                    if move_name_p2 in STATUS_MOVES: p2_status_moves += 1
                    if move_name_p2 in SETUP_MOVES: p2_setup_moves += 1
                    if p2_move.get('base_power'):
                        p2_total_bp += p2_move['base_power']

                if turn.get('turn') == last_turn_num:
                    p1_boosts = sum(p1_state.get('boosts', {}).values()) if p1_state else 0
                    p2_boosts = sum(p2_state.get('boosts', {}).values()) if p2_state else 0
                    feat_end_boost_diff = p1_boosts - p2_boosts
                    
                    p1_active_hp_end = p1_state.get('hp_pct', 0) if p1_state else 0
                    p2_active_hp_end = p2_state.get('hp_pct', 0) if p2_state else 0

        # Calcoli finali
        p1_total_hp_seen = sum(p['hp_pct'] for p in p1_seen_status.values())
        p2_total_hp_seen = sum(p['hp_pct'] for p in p2_seen_status.values())
        feat_hp_advantage_seen = p1_total_hp_seen - p2_total_hp_seen
        
        feat_mons_revealed_diff = len(p2_seen_status) - len(p1_seen_status)
        
        p1_team_status_count = sum(1 for p in p1_seen_status.values() if p['status'] is not None)
        p2_team_status_count = sum(1 for p in p2_seen_status.values() if p['status'] is not None)
        feat_team_status_diff = p1_team_status_count - p2_team_status_count

        feat_status_move_diff = p1_status_moves - p2_status_moves
        feat_setup_move_diff = p1_setup_moves - p2_setup_moves
        
        p1_fainted_count = sum(1 for p in p1_seen_status.values() if p['hp_pct'] == 0)
        p2_fainted_count = sum(1 for p in p2_seen_status.values() if p['hp_pct'] == 0)
        feat_fainted_mons_diff = p2_fainted_count - p1_fainted_count 

        feat_hp_advantage_active = p1_active_hp_end - p2_active_hp_end
        
        feat_total_base_power_diff = p1_total_bp - p2_total_bp
        feat_volatile_status_diff = p2_confused_turns - p1_confused_turns 

        processed_data.append({
            'battle_id': row['battle_id'],
            # Categoriche
            'p1_lead_name': p1_lead['name'], 
            'p2_lead_name': p2_lead['name'],
            # Numeriche (Core)
            'lead_speed_diff': feat_lead_speed_diff,
            'hp_advantage_seen': feat_hp_advantage_seen,
            'mons_revealed_diff': feat_mons_revealed_diff,
            'team_status_diff': feat_team_status_diff,
            'end_boost_diff': feat_end_boost_diff,
            # Numeriche (Aggregati Team e Meta)
            'p1_team_avg_atk': feat_p1_team_avg_atk,
            'p1_team_avg_spe': feat_p1_team_avg_spe,
            'p1_team_max_hp': feat_p1_team_max_hp,
            'p1_team_meta_count': feat_p1_team_meta_count,
            'p2_lead_is_meta': feat_p2_lead_is_meta,
            # Numeriche (Aggregati Mosse)
            'status_move_diff': feat_status_move_diff,
            'setup_move_diff': feat_setup_move_diff,
            'total_base_power_diff': feat_total_base_power_diff,
            'volatile_status_diff': feat_volatile_status_diff,
            # Numeriche (Lead Diffs)
            'lead_hp_diff': feat_lead_hp_diff,
            'lead_atk_diff': feat_lead_atk_diff,
            'lead_def_diff': feat_lead_def_diff,
            'lead_spa_diff': feat_lead_spa_diff,
            'lead_spd_diff': feat_lead_spd_diff,
            # Numeriche (Momentum)
            'team_speed_adv_vs_lead': feat_team_speed_adv_vs_lead,
            'fainted_mons_diff': feat_fainted_mons_diff,
            'hp_advantage_active': feat_hp_advantage_active,
            
            # --- ADD NEW FEATURES HERE ---
            'lead_type_adv_diff': feat_lead_type_adv_diff,
            'p1_team_avg_special': feat_p1_team_avg_special
            
        })
    return pd.DataFrame(processed_data).set_index('battle_id')