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
    POKEMON_LIST,
    STATUS_WEIGHTS,
    KEY_ATTACKS
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
def create_specialist_features(df):
    
    useless_leads = {
        'Articuno', 'Golem', 'Rhydon', 'Lapras', 'Cloyster', 
        'Charizard', 'Victreebel', 'Dragonite', 'Gengar', 'Persian' 
    }
    
    processed_data = []
    
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Analisi 'Specialist'"):
        
        p1_team = row['p1_team_details']
        p2_lead = row['p2_lead_details']
        timeline = row['battle_timeline']
        p1_lead = p1_team[0]
        
        p1_lead_name = p1_lead['name'] if p1_lead['name'] not in useless_leads else 'Other'
        p2_lead_name = p2_lead['name'] if p2_lead['name'] not in useless_leads else 'Other'

        p1_types = p1_lead.get('types', [])
        p2_types = p2_lead.get('types', [])
        p1_max_eff = max([get_type_effectiveness(t, p2_types) for t in p1_types] + [1.0])
        p2_max_eff = max([get_type_effectiveness(t, p1_types) for t in p2_types] + [1.0])
        feat_lead_type_adv = p1_max_eff - p2_max_eff
        
        feat_lead_speed_diff = p1_lead['base_spe'] - p2_lead['base_spe']
        feat_lead_atk_diff = p1_lead['base_atk'] - p2_lead['base_atk']
        p1_bulk = p1_lead['base_hp'] + p1_lead['base_def'] + p1_lead['base_spa']
        p2_bulk = p2_lead['base_hp'] + p2_lead['base_def'] + p2_lead['base_spa']
        feat_lead_bulk_diff = p1_bulk - p2_bulk
        
        feat_p1_team_avg_speed = np.mean([p.get('base_spe', 70) for p in p1_team])
        feat_p1_team_avg_bulk = np.mean([(p.get('base_hp', 80) + p.get('base_def', 80) + p.get('base_spa', 80)) for p in p1_team])
        feat_p1_meta_threat_count = sum(1 for p in p1_team if p.get('name') in META_THREATS_GEN1)
        feat_p2_lead_is_meta_threat = 1 if p2_lead['name'] in META_THREATS_GEN1 else 0
        
        p1_seen_status = {p['name']: {'hp_pct': 100, 'status': None} for p in p1_team}
        p2_seen_status = {p2_lead['name']: {'hp_pct': 100, 'status': None}}
        feat_end_boost_diff = 0
        feat_p1_lead_stay_duration = 0
        feat_p2_lead_forced_out = 0
        feat_first_ko_turn = 0
        feat_p1_setup_moves = 0
        feat_p2_setup_moves = 0
        feat_p1_key_attacks = 0
        feat_p2_key_attacks = 0
        feat_num_turns = 0 
        
        if timeline:
            feat_num_turns = timeline[-1].get('turn', 0) 
            first_ko_achieved = False
            last_p2_mon_name = p2_lead['name']
            
            for i, turn in enumerate(timeline):
                p1_state = turn.get('p1_pokemon_state', {})
                p2_state = turn.get('p2_pokemon_state', {})
                p1_move = turn.get('p1_move_details')
                p2_move = turn.get('p2_move_details')
                
                if p1_state and p1_state.get('name'):
                    p1_name = p1_state['name']
                    p1_seen_status.setdefault(p1_name, {'hp_pct': 100, 'status': None})
                    if p1_state.get('hp_pct') is not None: p1_seen_status[p1_name]['hp_pct'] = p1_state.get('hp_pct')
                    p1_seen_status[p1_name]['status'] = p1_state.get('status')
                if p2_state and p2_state.get('name'):
                    p2_name = p2_state['name']
                    p2_seen_status.setdefault(p2_name, {'hp_pct': 100, 'status': None})
                    if p2_state.get('hp_pct') is not None: p2_seen_status[p2_name]['hp_pct'] = p2_state.get('hp_pct')
                    p2_seen_status[p2_name]['status'] = p2_state.get('status')
                
                if turn.get('turn') == feat_num_turns:
                    feat_end_boost_diff = sum(p1_state.get('boosts', {}).values()) - sum(p2_state.get('boosts', {}).values())
                
                if p1_state.get('name') == p1_lead['name']: feat_p1_lead_stay_duration += 1
                if i > 0 and p2_state.get('name') != last_p2_mon_name and last_p2_mon_name == p2_lead['name']: feat_p2_lead_forced_out = 1
                if p2_state.get('name') != last_p2_mon_name: last_p2_mon_name = p2_state.get('name')
                if not first_ko_achieved and (p1_state.get('hp_pct') == 0 or p2_state.get('hp_pct') == 0):
                    feat_first_ko_turn = turn.get('turn', 0)
                    first_ko_achieved = True
                
                if p1_move:
                    if p1_move.get('name') in SETUP_MOVES: feat_p1_setup_moves += 1
                    if p1_move.get('name') in KEY_ATTACKS: feat_p1_key_attacks += 1
                if p2_move:
                    if p2_move.get('name') in SETUP_MOVES: feat_p2_setup_moves += 1
                    if p2_move.get('name') in KEY_ATTACKS: feat_p2_key_attacks += 1

        feat_setup_advantage = feat_p1_setup_moves - feat_p2_setup_moves
        feat_key_attack_adv = feat_p1_key_attacks - feat_p2_key_attacks

        p1_total_hp_seen = sum(p['hp_pct'] for p in p1_seen_status.values())
        p2_total_hp_seen = sum(p['hp_pct'] for p in p2_seen_status.values())
        feat_hp_advantage_seen = p1_total_hp_seen - p2_total_hp_seen 
        feat_mons_revealed_diff = len(p2_seen_status) - len(p1_seen_status) 

        def calculate_status_score(status_dict, weights):
            score = 0
            for p in status_dict.values():
                status = p.get('status')
                if status in weights: 
                    score += weights[status]
            return score
        feat_weighted_status_diff = calculate_status_score(p1_seen_status, STATUS_WEIGHTS) - calculate_status_score(p2_seen_status, STATUS_WEIGHTS)
        
        p1_team_status_count = sum(1 for p in p1_seen_status.values() if p['status'] is not None)
        p2_team_status_count = sum(1 for p in p2_seen_status.values() if p['status'] is not None)
        feat_team_status_diff = p1_team_status_count - p2_team_status_count

        processed_data.append({
            'battle_id': row['battle_id'],
            'p1_lead_name': p1_lead_name, 
            'p2_lead_name': p2_lead_name,
            
            'lead_speed_diff': feat_lead_speed_diff,
            'hp_advantage_seen': feat_hp_advantage_seen,
            'mons_revealed_diff': feat_mons_revealed_diff,
            'team_status_diff': feat_team_status_diff, 
            'end_boost_diff': feat_end_boost_diff,
            'num_turns': feat_num_turns,
            
            'lead_type_adv': feat_lead_type_adv,
            'lead_atk_diff': feat_lead_atk_diff,
            'lead_bulk_diff': feat_lead_bulk_diff,
            'p1_team_avg_speed': feat_p1_team_avg_speed,
            'p1_team_avg_bulk': feat_p1_team_avg_bulk,
            'p1_meta_threat_count': feat_p1_meta_threat_count,
            'p2_lead_is_meta_threat': feat_p2_lead_is_meta_threat,
            'p1_lead_stay_duration': feat_p1_lead_stay_duration,
            'p2_lead_forced_out': feat_p2_lead_forced_out,
            'first_ko_turn': feat_first_ko_turn,
            'setup_advantage': feat_setup_advantage,
            'key_attack_adv': feat_key_attack_adv,
            'weighted_status_diff': feat_weighted_status_diff,
        })
    return pd.DataFrame(processed_data).set_index('battle_id')