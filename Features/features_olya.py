from tqdm.auto import tqdm
import pandas as pd
import numpy as np 
import sklearn as skl


from utils.functions import (
    extract_base_stats,
    get_type_effectiveness,
    encode_pokemon_set,
    get_pokemons_seen_in_battle,
    get_base_stats,
    get_all_pokemons_used,
    safe_get,
    extract_levels,
    check_missing
)

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

'''  
### **Lead Pokémon Features**

* `lead_speed_diff`: Speed difference between player 1 and player 2 leads.
* `lead_type_adv`: Product of type effectiveness of player 1 lead against player 2 lead.
* `total_stats_diff`: Sum of differences in base stats (hp, atk, def, spa, spd, spe) between leads.
* `stats_speed_interaction`: Interaction term of total stats difference × speed difference.
* `hp_vs_stats_ratio`: Ratio of HP advantage to total stats difference.
* `atk_def_ratio_p1` / `atk_def_ratio_p2`: Attack-to-defense ratio for each lead.
* `hp_speed_interaction_lead`: Product of lead’s HP × speed.
* `hp_def_ratio_p1` / `hp_def_ratio_p2`: HP-to-defense ratio for each lead.

---

### **HP & Healing Features**

* `hp_advantage_seen`: Difference in total HP of Pokémon seen for each player.
* `hp_diff_std`: Standard deviation of HP difference over the battle.
* `hp_diff_range`: Range of HP difference over the battle.
* `hp_diff_mean`: Mean HP difference over the battle.
* `hp_diff_last`: HP difference at the last turn.
* `feat_hp_trend_diff`: Average turn-to-turn change in HP difference.
* `feat_total_healing_done`: Total healing done by player 1.
* `feat_total_damage_dealt`: Total damage dealt by player 1.

---

### **Damage Snapshot Features**

* `damage_diff_turn10` / `damage_diff_turn20` / `damage_diff_turn25` / `damage_diff_turn30`: Damage differences at specific turns.
* `damage_ratio_turn10_20` / `damage_ratio_turn10_30` / `damage_ratio_turn20_25` / `damage_ratio_turn25_30`: Ratios of damage differences between different turns.

---

### **Battle Timeline & Momentum**

* `feat_num_turns`: Total number of turns in the battle.
* `feat_first_faint_turn`: Turn when first faint occurred.
* `momentum_shift_turn`: Turn when control of battle changed (HP difference sign flipped).
* `comeback_score`: Absolute difference between mid-battle HP difference and final HP difference.
* `early_sustain`: Difference between mid-battle HP difference and start.
* `momentum_flips`: Number of sign flips in HP difference over turns.

---

### **Boost Features**

* `end_boost_diff`: Difference in boosts at the end of battle.
* `boost_volatility`: Standard deviation of boost difference over battle.
* `boost_trend`: Average trend in boost difference over turns.
* `p1_boost_mean` / `p2_boost_mean`: Average boosts per player.
* `boost_diff_mean`: Mean difference in boosts per turn.

---

### **Status & Setup Features**

* `status_turns`: Number of turns Pokémon had status effects.
* `team_status_diff`: Difference in number of Pokémon with status effects between players.
* `status_balance`: Difference in total status inflicted vs. suffered.
* `feat_status_diff_inflicted`: Difference between inflicted and suffered status counts.
* `status_setup_diff`: Difference in number of moves that are setup or status moves between leads.

---

### **Meta & Threat Features**

* `meta_diff`: Difference in number of meta-threat Pokémon between player 1 team and player 2 lead.

---

### **Switch & Aggression Features**

* `feat_switch_diff`: Difference in number of switches between players.
* `feat_aggression_diff`: Difference in proportion of attack actions between players.
* `p1_aggression` / `p2_aggression`: Proportion of turns player performed attacks.
* `stall_ratio`: Fraction of turns with minimal HP change (stalling).
* `aggression_index`: Weighted aggression score based on HP changes.

---

### **Per-Turn Aggregates**

* `p1_hp_mean` / `p2_hp_mean`: Average HP per player over turns.
* `hp_diff_mean` / `hp_diff_std`: Mean & std of HP difference per turn.
* `p1_status_total` / `p2_status_total`: Total number of Pokémon with status per player.
* `p1_hp_per_turn` / `p2_hp_per_turn`: (implicit, used for aggregation) HP per turn.

---

### **Team Embedding Features**

* `p1_seen_pokemons` / `p2_seen_pokemons`: Multi-hot vector encoding Pokémon seen.
* `feat_team_emb_sim`: Cosine similarity between seen Pokémon embeddings.
* `p1_team_emb_0..17` / `p2_team_emb_0..17`: Aggregated team embeddings (mean, max, min concatenated).

'''


def create_advanced_features_gen2(df):
    processed_data = []
    embedding_dim = 6  # hp, atk, def, spa, spd, spe

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating advanced features"):
        p1_team = row['p1_team_details']
        p2_lead = row['p2_lead_details']
        timeline = row['battle_timeline']
        p1_lead = p1_team[0]

        # --- Ensure Pokémon stats exist ---
        def fill_missing_stats(pokemon):
            name = pokemon.get('name', '').lower()
            if not name:
                return pokemon
            if any(k not in pokemon for k in ['base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe']):
                base_info = pokemon_base_stats_nested.get(name)
                if base_info:
                    for stat in ['base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe']:
                        pokemon[stat] = pokemon.get(stat, base_info[stat]['value'])
                else:
                    pokemon.update({s: 80 for s in ['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']})
            return pokemon

        p1_lead = fill_missing_stats(p1_lead)
        p2_lead = fill_missing_stats(p2_lead)
        for mon in p1_team:
            fill_missing_stats(mon)

        # --- Basic numeric features ---
        feat_lead_speed_diff = p1_lead['base_spe'] - p2_lead['base_spe']
        feat_end_boost_diff = 0
        feat_total_damage_dealt = 0
        feat_total_healing_done = 0
        feat_status_turns = 0
        feat_first_faint_turn = 0
        feat_damage_diff_turn25 = 0
        feat_damage_diff_turn30 = 0

        # --- Track seen Pokémon ---
        p1_seen_status = {p['name']: {'hp_pct': 100, 'status': None} for p in p1_team}
        p2_seen_status = {p2_lead['name']: {'hp_pct': 100, 'status': None}}

        hp_diff_series = []  # new: track per-turn total HP difference

        # --- Timeline analysis ---
        if timeline:
            feat_num_turns = timeline[-1].get('turn', 0)
            for turn in timeline:
                p1_state = turn.get('p1_pokemon_state', {})
                p2_state = turn.get('p2_pokemon_state', {})
                turn_num = turn.get('turn', 0)

                # Update p1 and p2 states
                if p1_state and p1_state.get('name'):
                    name = p1_state['name']
                    p1_seen_status.setdefault(name, {'hp_pct': 100, 'status': None})
                    prev_hp = p1_seen_status[name]['hp_pct']
                    p1_seen_status[name]['hp_pct'] = p1_state.get('hp_pct', prev_hp)
                    p1_seen_status[name]['status'] = p1_state.get('status', p1_seen_status[name]['status'])

                if p2_state and p2_state.get('name'):
                    name = p2_state['name']
                    p2_seen_status.setdefault(name, {'hp_pct': 100, 'status': None})
                    prev_hp = p2_seen_status[name]['hp_pct']
                    p2_seen_status[name]['hp_pct'] = p2_state.get('hp_pct', prev_hp)
                    p2_seen_status[name]['status'] = p2_state.get('status', p2_seen_status[name]['status'])

                # Track total damage / healing
                if p2_state.get('hp_pct') is not None:
                    feat_total_damage_dealt += max(0, 100 - p2_state['hp_pct'])
                if p1_state.get('hp_pct') is not None:
                    feat_total_healing_done += max(0, p1_state['hp_pct'] - 100)

                # Count status turns
                feat_status_turns += sum(s['status'] is not None for s in p1_seen_status.values())
                feat_status_turns += sum(s['status'] is not None for s in p2_seen_status.values())

                # First faint turn
                if not feat_first_faint_turn:
                    if any(s['hp_pct'] <= 0 for s in p1_seen_status.values()) or any(s['hp_pct'] <= 0 for s in p2_seen_status.values()):
                        feat_first_faint_turn = turn_num

                # --- Record total HP diff this turn ---
                total_hp_p1 = sum(p['hp_pct'] for p in p1_seen_status.values())
                total_hp_p2 = sum(p['hp_pct'] for p in p2_seen_status.values())
                hp_diff_series.append(total_hp_p1 - total_hp_p2)

                
                # Snapshots
                
                if turn_num == 10 or (turn_num == feat_num_turns and turn_num < 10):
                    feat_damage_diff_turn10 = (600 - total_hp_p1) - (600 - total_hp_p2)
                if turn_num == 20 or (turn_num == feat_num_turns and turn_num < 20):
                    feat_damage_diff_turn20 = (600 - total_hp_p1) - (600 - total_hp_p2)
                if turn_num == 25 or (turn_num == feat_num_turns and turn_num < 25):
                    feat_damage_diff_turn25 = (600 - total_hp_p1) - (600 - total_hp_p2)
                if turn_num == 30 or (turn_num == feat_num_turns and turn_num < 30):
                    feat_damage_diff_turn30 = (600 - total_hp_p1) - (600 - total_hp_p2)

            # End-of-battle boosts
            last_turn = timeline[-1]
            p1_boosts = sum(last_turn.get('p1_pokemon_state', {}).get('boosts', {}).values())
            p2_boosts = sum(last_turn.get('p2_pokemon_state', {}).get('boosts', {}).values())
            feat_end_boost_diff = p1_boosts - p2_boosts

        # --- New: HP trend difference ---
        if len(hp_diff_series) > 1:
            feat_hp_trend_diff = np.mean(np.diff(hp_diff_series))
        else:
            feat_hp_trend_diff = 0

        # --- Derived battle features ---
        p1_total_hp_seen = sum(p['hp_pct'] for p in p1_seen_status.values())
        p2_total_hp_seen = sum(p['hp_pct'] for p in p2_seen_status.values())
        feat_hp_advantage_seen = p1_total_hp_seen - p2_total_hp_seen
        feat_mons_revealed_diff = len(p2_seen_status) - len(p1_seen_status)
        feat_team_status_diff = (
            sum(1 for p in p1_seen_status.values() if p['status'] is not None)
            - sum(1 for p in p2_seen_status.values() if p['status'] is not None)
        )

        # --- Type and setup stuff ---
        p1_lead_types = p1_lead.get('types', [])
        p2_lead_types = p2_lead.get('types', [])
        feat_lead_type_adv = np.prod([get_type_effectiveness(t1, p2_lead_types) for t1 in p1_lead_types]) if p1_lead_types else 1.0
        feat_meta_diff = sum(1 for p in p1_team if p['name'] in META_THREATS_GEN1) - (1 if p2_lead['name'] in META_THREATS_GEN1 else 0)
        feat_status_setup_diff = (
            sum(1 for m in p1_lead.get('moves', []) if m in STATUS_MOVES or m in SETUP_MOVES)
            - sum(1 for m in p2_lead.get('moves', []) if m in STATUS_MOVES or m in SETUP_MOVES)
        )

        p1_seen_set, p2_seen_set = get_pokemons_seen_in_battle(timeline)
        p1_seen_encoded = encode_pokemon_set(p1_seen_set)
        p2_seen_encoded = encode_pokemon_set(p2_seen_set)

        # --- Lead embeddings ---
        p1_lead_embedding = pokemon_embeddings.get(p1_lead['name'], np.zeros(embedding_dim))
        p2_lead_embedding = pokemon_embeddings.get(p2_lead['name'], np.zeros(embedding_dim))

        feat_total_stats_diff = sum(
            p1_lead.get(stat, 0) - p2_lead.get(stat, 0)
            for stat in ['base_hp', 'base_atk', 'base_def', 'base_spe', 'base_spa', 'base_spd']
        )
        
        feat_switches_p1 = sum(1 for t in timeline if t.get('p1_action') == 'switch')
        feat_switches_p2 = sum(1 for t in timeline if t.get('p2_action') == 'switch')
        feat_switch_diff = feat_switches_p1 - feat_switches_p2
        
        feat_aggression_p1 = sum(1 for t in timeline if t.get('p1_action') == 'attack') / len(timeline)
        feat_aggression_p2 = sum(1 for t in timeline if t.get('p2_action') == 'attack') / len(timeline)
        feat_aggression_diff = feat_aggression_p1 - feat_aggression_p2
        
                # --- Additional high-value features (new) ---

        # 1. Momentum & volatility
        feat_hp_diff_std = np.std(hp_diff_series) if hp_diff_series else 0
        feat_hp_diff_range = (max(hp_diff_series) - min(hp_diff_series)) if hp_diff_series else 0
        feat_momentum_shift_turn = next(
            (i for i in range(1, len(hp_diff_series))
             if np.sign(hp_diff_series[i]) != np.sign(hp_diff_series[i-1])),
            0
        )  # when control of the match flipped

        # 2. Sustain & comeback potential
        feat_hp_final = hp_diff_series[-1] if hp_diff_series else 0
        feat_hp_mid = hp_diff_series[len(hp_diff_series)//2] if len(hp_diff_series) > 2 else feat_hp_final
        feat_comeback_score = abs(feat_hp_final - feat_hp_mid)
        feat_early_sustain = feat_hp_mid - hp_diff_series[0] if hp_diff_series else 0

        # 3. Status dynamics
        feat_status_inflicted = sum(1 for s in p2_seen_status.values() if s['status'])
        feat_status_suffered = sum(1 for s in p1_seen_status.values() if s['status'])
        feat_status_balance = feat_status_inflicted - feat_status_suffered

        # 4. Boost trends
        boost_diffs = []
        for turn in timeline:
            p1_boost_sum = sum(turn.get('p1_pokemon_state', {}).get('boosts', {}).values()) if 'boosts' in turn.get('p1_pokemon_state', {}) else 0
            p2_boost_sum = sum(turn.get('p2_pokemon_state', {}).get('boosts', {}).values()) if 'boosts' in turn.get('p2_pokemon_state', {}) else 0
            boost_diffs.append(p1_boost_sum - p2_boost_sum)
        feat_boost_volatility = np.std(boost_diffs) if boost_diffs else 0
        feat_boost_trend = np.mean(np.diff(boost_diffs)) if len(boost_diffs) > 1 else 0

        # 5. Move dynamics (if move info available)
        p1_move_power = [t.get('p1_move_details', {}).get('base_power', 0) for t in timeline if t.get('p1_move_details')]
        p2_move_power = [t.get('p2_move_details', {}).get('base_power', 0) for t in timeline if t.get('p2_move_details')]
        feat_move_power_diff = np.mean(p1_move_power or [0]) - np.mean(p2_move_power or [0])

        feat_move_variety_p1 = len(set(t.get('p1_move_details', {}).get('type', None)
                                       for t in timeline if t.get('p1_move_details')))
        feat_move_variety_p2 = len(set(t.get('p2_move_details', {}).get('type', None)
                                       for t in timeline if t.get('p2_move_details')))
        feat_move_diversity_diff = feat_move_variety_p1 - feat_move_variety_p2

        # 6. Aggression and stalling behavior
        hp_changes = [abs(hp_diff_series[i] - hp_diff_series[i-1]) for i in range(1, len(hp_diff_series))]
        feat_stall_ratio = sum(1 for c in hp_changes if c < 5) / len(hp_changes) if hp_changes else 0
        feat_aggression_index = (1 - feat_stall_ratio) * (abs(np.mean(np.diff(hp_diff_series))) if len(hp_diff_series) > 1 else 0)


        # --- Combine all features ---
        # --- Combine all features ---
        feature_dict = {
            'battle_id': row['battle_id'],
            'lead_speed_diff': feat_lead_speed_diff,                 # ✅ top 10
            'hp_advantage_seen': feat_hp_advantage_seen,             # ✅ top 10
            'mons_revealed_diff': feat_mons_revealed_diff,           # ✅ top 10
            'team_status_diff': feat_team_status_diff,             # ❌ not top 10
            'end_boost_diff': feat_end_boost_diff,                 # ❌ not top 10
            'total_damage_dealt': feat_total_damage_dealt,           # ✅ top 10
            'total_healing_done': feat_total_healing_done,         # ❌ not top 10
            'status_turns': feat_status_turns,                       # ✅ top 10
            'first_faint_turn': feat_first_faint_turn,               # ✅ top 10
            'lead_type_adv': feat_lead_type_adv,                   # ❌ not top 10
            'meta_diff': feat_meta_diff,                           # ❌ not top 10
            'status_setup_diff': feat_status_setup_diff,           # ❌ not top 10
            'p1_seen_pokemons': p1_seen_encoded,                   # ❌ not top 10
            'p2_seen_pokemons': p2_seen_encoded,                   # ❌ not top 10
            'total_stats_diff': feat_total_stats_diff,              # ✅ top 10
            'damage_diff_turn10': feat_damage_diff_turn25,           # ✅ top 10
            'damage_diff_turn20': feat_damage_diff_turn30,
            'damage_diff_turn25': feat_damage_diff_turn25,           # ✅ top 10
            'damage_diff_turn30': feat_damage_diff_turn30,           # ✅ top 10
            'hp_trend_diff': feat_hp_trend_diff,                     # ✅ top 10
            'feat_switch_diff': feat_switch_diff,                  # ❌ not top 10
            'feat_aggression_diff': feat_aggression_diff,          # ❌ not top 10
            'hp_diff_std': feat_hp_diff_std,
            'hp_diff_range': feat_hp_diff_range,
            'momentum_shift_turn': feat_momentum_shift_turn,
            'comeback_score': feat_comeback_score,
            'early_sustain': feat_early_sustain,
            'status_balance': feat_status_balance,
            'boost_volatility': feat_boost_volatility,
            'boost_trend': feat_boost_trend,
            'move_power_diff': feat_move_power_diff,
            'move_diversity_diff': feat_move_diversity_diff,
            'stall_ratio': feat_stall_ratio,
            'aggression_index': feat_aggression_index,
        }

        feature_dict['stats_speed_interaction'] = feat_total_stats_diff * feat_lead_speed_diff
        feature_dict['hp_vs_stats_ratio'] = feat_hp_advantage_seen / feat_total_stats_diff if feat_total_stats_diff != 0 else 0
        feature_dict['damage_ratio_turn25_30'] = feat_damage_diff_turn25 / feat_damage_diff_turn30 if feat_damage_diff_turn30 != 0 else 0
        feature_dict['damage_ratio_turn20_25'] = feat_damage_diff_turn20 / feat_damage_diff_turn25 if feat_damage_diff_turn25 != 0 else 0
        feature_dict['damage_ratio_turn10_20'] = feat_damage_diff_turn10 / feat_damage_diff_turn20 if feat_damage_diff_turn20 != 0 else 0
        feature_dict['damage_ratio_turn10_30'] = feat_damage_diff_turn10 / feat_damage_diff_turn30 if feat_damage_diff_turn30 != 0 else 0
        
        feature_dict['p1_lead_special_total'] = p1_lead['base_spa'] + p1_lead['base_spd']
        feature_dict['p2_lead_special_total'] = p2_lead['base_spa'] + p2_lead['base_spd']
        feature_dict['special_total_diff'] = feature_dict['p1_lead_special_total'] - feature_dict['p2_lead_special_total']

        
        feature_dict['p1_lead_physical_total'] = p1_lead['base_atk'] + p1_lead['base_def']
        feature_dict['p2_lead_physical_total'] = p2_lead['base_atk'] + p2_lead['base_def']
        feature_dict['physical_total_diff'] = feature_dict['p1_lead_physical_total'] - feature_dict['p2_lead_physical_total']
        feature_dict['atk_def_ratio_p1'] = p1_lead['base_atk'] / (p1_lead['base_def'] + 1)
        feature_dict['atk_def_ratio_p2'] = p2_lead['base_atk'] / (p2_lead['base_def'] + 1)
        
        feature_dict['hp_speed_interaction_lead'] = p1_lead['base_hp'] * p1_lead['base_spe']
        feature_dict['hp_def_ratio_p1'] = p1_lead['base_hp'] / (p1_lead['base_def'] + 1)
        feature_dict['hp_def_ratio_p2'] = p2_lead['base_hp'] / (p2_lead['base_def'] + 1)
        feature_dict['hp_vs_total_stats_p1'] = p1_lead['base_hp'] / (sum([p1_lead[stat] for stat in ['base_atk','base_def','base_spa','base_spd','base_spe']]) + 1)
        feature_dict['hp_vs_total_stats_p2'] = p2_lead['base_hp'] / (sum([p2_lead[stat] for stat in ['base_atk','base_def','base_spa','base_spd','base_spe']]) + 1)

        
        feature_dict['lead_total_stats_p1'] = sum([p1_lead[stat] for stat in ['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']])
        feature_dict['lead_total_stats_p2'] = sum([p2_lead[stat] for stat in ['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']])

        feature_dict['atk_hp_ratio_p1'] = p1_lead['base_atk'] / (p1_lead['base_hp'] + 1)
        feature_dict['atk_hp_ratio_p2'] = p2_lead['base_atk'] / (p2_lead['base_hp'] + 1)
        feature_dict['def_hp_ratio_p1'] = p1_lead['base_def'] / (p1_lead['base_hp'] + 1)
        feature_dict['def_hp_ratio_p2'] = p2_lead['base_def'] / (p2_lead['base_hp'] + 1)

        # HP trend difference (you already compute this as feat_hp_trend_diff)
        feature_dict['feat_hp_trend_diff'] = feat_hp_trend_diff

        # Status inflicted difference
        feat_status_inflicted = sum(1 for s in p2_seen_status.values() if s['status'])
        feat_status_suffered = sum(1 for s in p1_seen_status.values() if s['status'])
        feature_dict['feat_status_diff_inflicted'] = feat_status_inflicted - feat_status_suffered
        
        
        # --- Inside your for _, row in df.iterrows() loop, after timeline processing ---

        # --- 1. Per-turn team statistics ---
        p1_hp_per_turn = []
        p2_hp_per_turn = []
        p1_boost_per_turn = []
        p2_boost_per_turn = []
        p1_status_per_turn = []
        p2_status_per_turn = []

        for turn in timeline:
            # Sum HP for all Pokémon alive
            p1_total_hp = sum(p['hp_pct'] for p in p1_seen_status.values())
            p2_total_hp = sum(p['hp_pct'] for p in p2_seen_status.values())
            p1_hp_per_turn.append(p1_total_hp)
            p2_hp_per_turn.append(p2_total_hp)

            # Sum boosts
            p1_boost_sum = sum(turn.get('p1_pokemon_state', {}).get('boosts', {}).values()) if 'boosts' in turn.get('p1_pokemon_state', {}) else 0
            p2_boost_sum = sum(turn.get('p2_pokemon_state', {}).get('boosts', {}).values()) if 'boosts' in turn.get('p2_pokemon_state', {}) else 0
            p1_boost_per_turn.append(p1_boost_sum)
            p2_boost_per_turn.append(p2_boost_sum)

            # Status count
            p1_status_count = sum(1 for s in p1_seen_status.values() if s['status'])
            p2_status_count = sum(1 for s in p2_seen_status.values() if s['status'])
            p1_status_per_turn.append(p1_status_count)
            p2_status_per_turn.append(p2_status_count)

        # --- 2. Aggregate stats ---
        feature_dict['p1_hp_mean'] = np.mean(p1_hp_per_turn)
        feature_dict['p2_hp_mean'] = np.mean(p2_hp_per_turn)
        feature_dict['hp_diff_mean'] = np.mean(np.array(p1_hp_per_turn) - np.array(p2_hp_per_turn))
        feature_dict['hp_diff_std'] = np.std(np.array(p1_hp_per_turn) - np.array(p2_hp_per_turn))
        feature_dict['hp_diff_last'] = p1_hp_per_turn[-1] - p2_hp_per_turn[-1]

        feature_dict['p1_boost_mean'] = np.mean(p1_boost_per_turn)
        feature_dict['p2_boost_mean'] = np.mean(p2_boost_per_turn)
        feature_dict['boost_diff_mean'] = np.mean(np.array(p1_boost_per_turn) - np.array(p2_boost_per_turn))
        feature_dict['boost_trend'] = np.mean(np.diff(p1_boost_per_turn) - np.diff(p2_boost_per_turn))

        feature_dict['p1_status_total'] = sum(p1_status_per_turn)
        feature_dict['p2_status_total'] = sum(p2_status_per_turn)
        feature_dict['status_balance'] = sum(p1_status_per_turn) - sum(p2_status_per_turn)

        # --- 3. Momentum & flips ---
        hp_diff_series = np.array(p1_hp_per_turn) - np.array(p2_hp_per_turn)
        sign_changes = np.sum(np.diff(np.sign(hp_diff_series)) != 0)
        feature_dict['momentum_flips'] = sign_changes
        feature_dict['comeback_score'] = abs(hp_diff_series[-1] - hp_diff_series[len(hp_diff_series)//2])
        feature_dict['early_sustain'] = hp_diff_series[len(hp_diff_series)//2] - hp_diff_series[0]

        # --- 4. Aggression / stalling ---
        p1_attack_count = sum(1 for t in timeline if t.get('p1_action') == 'attack')
        p2_attack_count = sum(1 for t in timeline if t.get('p2_action') == 'attack')
        feature_dict['p1_aggression'] = p1_attack_count / len(timeline) if timeline else 0
        feature_dict['p2_aggression'] = p2_attack_count / len(timeline) if timeline else 0
        feature_dict['aggression_diff'] = feature_dict['p1_aggression'] - feature_dict['p2_aggression']

        # Stall ratio
        hp_changes = np.abs(np.diff(hp_diff_series)) if len(hp_diff_series) > 1 else [0]
        feature_dict['stall_ratio'] = sum(1 for c in hp_changes if c < 5) / len(hp_changes) if len(hp_changes) > 0 else 0
        feature_dict['aggression_index'] = (1 - feature_dict['stall_ratio']) * np.mean(hp_changes) if len(hp_changes) > 0 else 0

        # --- 5. Team embedding aggregation ---
        embedding_dim = 6
        def aggregate_team_embedding(team):
            arr = np.array([pokemon_embeddings.get(p['name'], np.zeros(embedding_dim)) for p in team])
            if len(arr) == 0:
                return np.zeros(embedding_dim)
            return np.concatenate([arr.mean(axis=0), arr.max(axis=0), arr.min(axis=0)])  # 3 x embedding_dim

        p1_team_emb = aggregate_team_embedding(p1_team)
        p2_team_emb = aggregate_team_embedding(row.get('p2_team_details', []))
        for i, val in enumerate(p1_team_emb):
            feature_dict[f'p1_team_emb_{i}'] = val
        for i, val in enumerate(p2_team_emb):
            feature_dict[f'p2_team_emb_{i}'] = val

        
        
        
        # Embeddings
        for stat_name, val in zip(['hp','atk','def','spa','spd','spe'], p1_lead_embedding):
            feature_dict[f'p1_lead_{stat_name}'] = val
        for stat_name, val in zip(['hp','atk','def','spa','spd','spe'], p2_lead_embedding):
            feature_dict[f'p2_lead_{stat_name}'] = val

        processed_data.append(feature_dict)

    return pd.DataFrame(processed_data).set_index('battle_id')
