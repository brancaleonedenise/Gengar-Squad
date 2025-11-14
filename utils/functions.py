import numpy as np
import pandas as pd

from collections.abc import Mapping, Sequence
from tqdm import tqdm

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


"""
Utility functions for extracting structured competitive Pokémon data
from raw battle logs (JSONL → DataFrame rows).

This module provides:
1. Base stat extraction:
   - `extract_base_stats` flattens the first 1000 Pokémon team entries into numeric base stat rows.
   - `get_base_stats` retrieves canonical or fallback stats for a Pokémon.

2. Battle timeline parsing:
   - `get_pokemons_seen_in_battle` returns which Pokémon appeared for each player.
   - `get_all_pokemons_used` scans the full dataset for every Pokémon name across teams, leads, and timeline states.

3. Encoding utilities:
   - `encode_pokemon_set` converts a set of Pokémon names into a multi-hot strength-ranking vector.
   - `pokemon_embeddings` maps each Pokémon to a fixed 6-dimensional stat embedding.

4. Gameplay logic helpers:
   - `get_type_effectiveness` evaluates Gen 1 damage multipliers based on the move's type and the target's types.

5. General helpers:
   - `safe_get` safely traverses deeply nested dictionaries.
   - `extract_levels` extracts Pokémon levels from the team structure.
   - `check_missing` recursively detects missing, null, or malformed values anywhere in a dict/list tree.

Everything assumes Showdown-style Gen 1 battle logs with fields like:
`p1_team_details`, `p2_lead_details`, and `battle_timeline`.
"""



def extract_base_stats(df):
    stats = []
    for _, row in df.head(1000).iterrows():
        team = row.get("p1_team_details")
        if isinstance(team, list):
            for poke in team:
                def grab_stat(field):
                    val = safe_get(poke, field, "value")
                    if val is None and isinstance(poke.get(field), (int, float)):
                        val = poke[field]
                    return val
                stat_entry = {
                    'hp': grab_stat('base_hp'),
                    'atk': grab_stat('base_atk'),
                    'def': grab_stat('base_def'),
                    'spa': grab_stat('base_spa'),
                    'spd': grab_stat('base_spd'),
                    'spe': grab_stat('base_spe')
                }
                if all(isinstance(v, (int, float)) for v in stat_entry.values()):
                    stats.append(stat_entry)
    return pd.DataFrame(stats)



# Funzione helper per calcolare l'efficacia
def get_type_effectiveness(move_type, target_types):
    if move_type not in TYPE_CHART_GEN1:
        return 1.0
    
    multiplier = 1.0
    chart_for_move = TYPE_CHART_GEN1[move_type]
    
    for target_type in target_types:
        if target_type in chart_for_move:
            multiplier *= chart_for_move[target_type]
            
    return multiplier


def encode_pokemon_set(pokemon_set):
    """Convert a set of Pokémon names into a multi-hot vector based on strength ranking."""
    vector = np.zeros(len(POKEMON_LIST), dtype=int)
    if not pokemon_set:
        return vector
    for p in pokemon_set:
        rank = POKEMON_RANKING.get(p)
        if rank:  # skip unknown Pokémon
            vector[rank - 1] = 1  # ranks start at 1, indices start at 0
    return vector


def get_pokemons_seen_in_battle(battle_timeline):
    """
    Extracts the set of Pokémon seen for each player during the battle.

    Args:
        battle_timeline (list): list of turn dictionaries. Each contains
            'p1_pokemon_state' and 'p2_pokemon_state' dicts with a 'name' key.

    Returns:
        (p1_seen, p2_seen): sets of Pokémon names for player 1 and player 2.
    """
    p1_seen = set()
    p2_seen = set()

    for turn in battle_timeline:
        # Player 1 Pokémon
        if 'p1_pokemon_state' in turn and isinstance(turn['p1_pokemon_state'], dict):
            name = turn['p1_pokemon_state'].get('name')
            if name:
                p1_seen.add(name)

        # Player 2 Pokémon
        if 'p2_pokemon_state' in turn and isinstance(turn['p2_pokemon_state'], dict):
            name = turn['p2_pokemon_state'].get('name')
            if name:
                p2_seen.add(name)

    return p1_seen, p2_seen

# --- helper function ---
def get_base_stats(name):
    """Return base stats from the nested dictionary or default neutral values."""
    base = pokemon_base_stats_nested.get(name.lower())
    if base:
        return {
            'base_hp': base['base_hp']['value'],
            'base_atk': base['base_atk']['value'],
            'base_def': base['base_def']['value'],
            'base_spa': base['base_spa']['value'],
            'base_spd': base['base_spd']['value'],
            'base_spe': base['base_spe']['value']
        }
    # fallback if not in the dictionary
    return {
        'base_hp': 80,
        'base_atk': 80,
        'base_def': 80,
        'base_spa': 80,
        'base_spd': 80,
        'base_spe': 80
    }


def get_all_pokemons_used(df):
    all_pokemons = set()
    
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Collecting Pokémon names"):
        # Player 1 team
        for p in row.get('p1_team_details', []):
            if 'name' in p:
                all_pokemons.add(p['name'])
        
        # Player 2 lead
        p2_lead = row.get('p2_lead_details', {})
        if 'name' in p2_lead:
            all_pokemons.add(p2_lead['name'])
        
        # Pokémon seen in battle timeline
        timeline = row.get('battle_timeline', [])
        for turn in timeline:
            # Player 1
            p1_state = turn.get('p1_pokemon_state')
            if p1_state and 'name' in p1_state:
                all_pokemons.add(p1_state['name'])
            
            # Player 2
            p2_state = turn.get('p2_pokemon_state')
            if p2_state and 'name' in p2_state:
                all_pokemons.add(p2_state['name'])
    
    return sorted(all_pokemons)  # sorted list for easier reference

def safe_get(d, *keys):
    """Traverse nested dicts or return None if something’s missing."""
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return None
    return d

def extract_levels(df):
    levels = []
    for _, row in df.head(1000).iterrows():
        team = row.get("p1_team_details")
        if isinstance(team, list):
            for poke in team:
                # Handle both nested and flat structure
                lvl = safe_get(poke, "level", "value")
                if lvl is None and isinstance(poke.get("level"), (int, float)):
                    lvl = poke["level"]
                if isinstance(lvl, (int, float)):
                    levels.append(lvl)
    return levels


# === Function to recursively detect missing / malformed values ===
def check_missing(data, path="root"):
    results = []
    if isinstance(data, Mapping):
        for k, v in data.items():
            results.extend(check_missing(v, f"{path}.{k}"))
    elif isinstance(data, Sequence) and not isinstance(data, str):
        for i, v in enumerate(data):
            results.extend(check_missing(v, f"{path}[{i}]"))
    else:
        if pd.isna(data) or data is None:
            results.append(path)
    return results


def get_best_stab_advantage(attacker_types, defender_types):
    """
    Calculates the best possible STAB multiplier an attacker has against a defender.
    """
    # Clean 'notype' from lists
    attacker_types = [t for t in attacker_types if t.upper() != 'NOTYPE']
    defender_types = [t for t in defender_types if t.upper() != 'NOTYPE']

    if not attacker_types:
        return 1.0 # No types, no STAB advantage

    best_multiplier = 0.0
    
    for move_type in attacker_types:
        # Get the multiplier for this STAB type against the defender's types
        multiplier = get_type_effectiveness(move_type.upper(), [t.upper() for t in defender_types])
        
        # We're looking for the *best* STAB move
        if multiplier > best_multiplier:
            best_multiplier = multiplier
            
    # If no STAB move is effective (e.g., Normal vs. Ghost), multiplier is 0.
    # Otherwise, it's the best one we found.
    # If best_multiplier is 0, we should return 0, not 1.
    if best_multiplier == 0.0:
        # Check if any type was effective at all, even if not > 0
        # This handles neutral hits (1.0)
        is_neutral = any(get_type_effectiveness(t.upper(), [d.upper() for d in defender_types]) >= 1.0 for t in attacker_types)
        if is_neutral:
             return 1.0
        
    return best_multiplier