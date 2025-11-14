import pandas as pd
import numpy as np 

pokemon_base_stats_nested = {
    'alakazam': {
        'name': {'value': 'alakazam'},
        'level': {'value': None},  # placeholder if unknown
        'types': {'value': []},    # placeholder if unknown
        'base_hp': {'value': 55},
        'base_atk': {'value': 50},
        'base_def': {'value': 45},
        'base_spa': {'value': 135},
        'base_spd': {'value': 81},
        'base_spe': {'value': 120}
    },
    'articuno': {
        'name': {'value': 'articuno'},
        'level': {'value': None},
        'types': {'value': []},
        'base_hp': {'value': 90},
        'base_atk': {'value': 85},
        'base_def': {'value': 100},
        'base_spa': {'value': 125},
        'base_spd': {'value': 97},
        'base_spe': {'value': 85}
    },
    'chansey': {
        'name': {'value': 'chansey'},
        'level': {'value': None},
        'types': {'value': []},
        'base_hp': {'value': 250},
        'base_atk': {'value': 5},
        'base_def': {'value': 5},
        'base_spa': {'value': 105},
        'base_spd': {'value': 83},
        'base_spe': {'value': 50}
    },
    'charizard': {
        'name': {'value': 'charizard'},
        'level': {'value': None},
        'types': {'value': []},
        'base_hp': {'value': 78},
        'base_atk': {'value': 84},
        'base_def': {'value': 78},
        'base_spa': {'value': 85},
        'base_spd': {'value': 85},
        'base_spe': {'value': 100}
    },
    'cloyster': {
        'name': {'value': 'cloyster'},
        'level': {'value': None},
        'types': {'value': []},
        'base_hp': {'value': 50},
        'base_atk': {'value': 95},
        'base_def': {'value': 180},
        'base_spa': {'value': 85},
        'base_spd': {'value': 96},
        'base_spe': {'value': 70}
    },
    'dragonite': {
        'name': {'value': 'dragonite'},
        'level': {'value': None},
        'types': {'value': []},
        'base_hp': {'value': 91},
        'base_atk': {'value': 134},
        'base_def': {'value': 95},
        'base_spa': {'value': 100},
        'base_spd': {'value': 100},
        'base_spe': {'value': 80}
    },
    'exeggutor': {
        'name': {'value': 'exeggutor'},
        'level': {'value': None},
        'types': {'value': []},
        'base_hp': {'value': 95},
        'base_atk': {'value': 95},
        'base_def': {'value': 85},
        'base_spa': {'value': 125},
        'base_spd': {'value': 91},
        'base_spe': {'value': 55}
    },
    'gengar': {
        'name': {'value': 'gengar'},
        'level': {'value': None},
        'types': {'value': []},
        'base_hp': {'value': 60},
        'base_atk': {'value': 65},
        'base_def': {'value': 60},
        'base_spa': {'value': 130},
        'base_spd': {'value': 85},
        'base_spe': {'value': 110}
    },
    'golem': {
        'name': {'value': 'golem'},
        'level': {'value': None},
        'types': {'value': []},
        'base_hp': {'value': 80},
        'base_atk': {'value': 110},
        'base_def': {'value': 130},
        'base_spa': {'value': 55},
        'base_spd': {'value': 84},
        'base_spe': {'value': 45}
    },
    'jolteon': {
        'name': {'value': 'jolteon'},
        'level': {'value': None},
        'types': {'value': []},
        'base_hp': {'value': 65},
        'base_atk': {'value': 65},
        'base_def': {'value': 60},
        'base_spa': {'value': 110},
        'base_spd': {'value': 86},
        'base_spe': {'value': 130}
    },
    'jynx': {
        'name': {'value': 'jynx'},
        'level': {'value': None},
        'types': {'value': []},
        'base_hp': {'value': 65},
        'base_atk': {'value': 50},
        'base_def': {'value': 35},
        'base_spa': {'value': 95},
        'base_spd': {'value': 68},
        'base_spe': {'value': 95}
    },
    'lapras': {
        'name': {'value': 'lapras'},
        'level': {'value': None},
        'types': {'value': []},
        'base_hp': {'value': 130},
        'base_atk': {'value': 85},
        'base_def': {'value': 80},
        'base_spa': {'value': 95},
        'base_spd': {'value': 90},
        'base_spe': {'value': 60}
    },
    'persian': {
        'name': {'value': 'persian'},
        'level': {'value': None},
        'types': {'value': []},
        'base_hp': {'value': 65},
        'base_atk': {'value': 70},
        'base_def': {'value': 60},
        'base_spa': {'value': 65},
        'base_spd': {'value': 75},
        'base_spe': {'value': 115}
    },
    'rhydon': {
        'name': {'value': 'rhydon'},
        'level': {'value': None},
        'types': {'value': []},
        'base_hp': {'value': 105},
        'base_atk': {'value': 130},
        'base_def': {'value': 120},
        'base_spa': {'value': 45},
        'base_spd': {'value': 88},
        'base_spe': {'value': 40}
    },
    'slowbro': {
        'name': {'value': 'slowbro'},
        'level': {'value': None},
        'types': {'value': []},
        'base_hp': {'value': 95},
        'base_atk': {'value': 75},
        'base_def': {'value': 110},
        'base_spa': {'value': 80},
        'base_spd': {'value': 78},
        'base_spe': {'value': 30}
    },
    'snorlax': {
        'name': {'value': 'snorlax'},
        'level': {'value': None},
        'types': {'value': []},
        'base_hp': {'value': 160},
        'base_atk': {'value': 110},
        'base_def': {'value': 65},
        'base_spa': {'value': 65},
        'base_spd': {'value': 86},
        'base_spe': {'value': 30}
    },
    'starmie': {
        'name': {'value': 'starmie'},
        'level': {'value': None},
        'types': {'value': []},
        'base_hp': {'value': 60},
        'base_atk': {'value': 75},
        'base_def': {'value': 85},
        'base_spa': {'value': 100},
        'base_spd': {'value': 87},
        'base_spe': {'value': 115}
    },
    'tauros': {
        'name': {'value': 'tauros'},
        'level': {'value': None},
        'types': {'value': []},
        'base_hp': {'value': 75},
        'base_atk': {'value': 100},
        'base_def': {'value': 95},
        'base_spa': {'value': 70},
        'base_spd': {'value': 90},
        'base_spe': {'value': 110}
    },
    'victreebel': {
        'name': {'value': 'victreebel'},
        'level': {'value': None},
        'types': {'value': []},
        'base_hp': {'value': 80},
        'base_atk': {'value': 105},
        'base_def': {'value': 65},
        'base_spa': {'value': 100},
        'base_spd': {'value': 84},
        'base_spe': {'value': 70}
    },
    'zapdos': {
        'name': {'value': 'zapdos'},
        'level': {'value': None},
        'types': {'value': []},
        'base_hp': {'value': 90},
        'base_atk': {'value': 90},
        'base_def': {'value': 85},
        'base_spa': {'value': 125},
        'base_spd': {'value': 98},
        'base_spe': {'value': 100}
    }
}


# Flatten base stats into arrays
pokemon_embeddings = {}
for name, data in pokemon_base_stats_nested.items():
    pokemon_embeddings[name] = np.array([
        data['base_hp']['value'],
        data['base_atk']['value'],
        data['base_def']['value'],
        data['base_spa']['value'],
        data['base_spd']['value'],
        data['base_spe']['value']
    ])


# Pokémon dominanti nel metagame Gen 1 OU (S-Tier e A-Tier)
# La loro presenza è un segnale fortissimo.
META_THREATS_GEN1 = {
    'Snorlax', 'Tauros', 'Chansey', 'Alakazam', 'Starmie', 'Exeggutor', 
    'Zapdos', 'Jolteon', 'Rhydon', 'Golem', 'Lapras'
}

# Mosse di setup o status chiave
STATUS_MOVES = {'Thunder Wave', 'Sleep Powder', 'Sing', 'Toxic', 'Lovely Kiss', 'Spore', 'Stun Spore', 'Glare'}
SETUP_MOVES = {'Amnesia', 'Swords Dance', 'Agility', 'Growth'}

TYPE_CHART_GEN1 = {
    'NORMAL': {'ROCK': 0.5, 'GHOST': 0.0},
    'FIRE': {'FIRE': 0.5, 'WATER': 0.5, 'GRASS': 2.0, 'ICE': 2.0, 'BUG': 2.0, 'ROCK': 0.5},
    'WATER': {'FIRE': 2.0, 'WATER': 0.5, 'GRASS': 0.5, 'GROUND': 2.0, 'ROCK': 2.0, 'DRAGON': 0.5},
    'ELECTRIC': {'WATER': 2.0, 'ELECTRIC': 0.5, 'GRASS': 0.5, 'GROUND': 0.0, 'FLYING': 2.0, 'DRAGON': 0.5},
    'GRASS': {'FIRE': 0.5, 'WATER': 2.0, 'ELECTRIC': 1.0, 'GRASS': 0.5, 'POISON': 0.5, 'GROUND': 2.0, 'FLYING': 0.5, 'BUG': 0.5, 'ROCK': 2.0, 'DRAGON': 0.5},
    'ICE': {'WATER': 0.5, 'GRASS': 2.0, 'ICE': 0.5, 'GROUND': 2.0, 'FLYING': 2.0, 'DRAGON': 2.0},
    'FIGHTING': {'NORMAL': 2.0, 'POISON': 0.5, 'FLYING': 0.5, 'PSYCHIC': 0.5, 'BUG': 0.5, 'ROCK': 2.0, 'GHOST': 0.0},
    'POISON': {'GRASS': 2.0, 'POISON': 0.5, 'GROUND': 0.5, 'BUG': 2.0, 'ROCK': 0.5, 'GHOST': 0.5},
    'GROUND': {'FIRE': 2.0, 'ELECTRIC': 2.0, 'GRASS': 0.5, 'POISON': 2.0, 'FLYING': 0.0, 'BUG': 0.5, 'ROCK': 2.0},
    'FLYING': {'ELECTRIC': 0.5, 'GRASS': 2.0, 'FIGHTING': 2.0, 'BUG': 2.0, 'ROCK': 0.5},
    'PSYCHIC': {'FIGHTING': 2.0, 'POISON': 2.0, 'PSYCHIC': 0.5, 'GHOST': 1.0}, # In Gen 1, Psychic era immune a Ghost per un bug, ma i dati Showdown potrebbero averlo corretto. Assumiamo 1.0 per sicurezza, o 0.0 se il bug è emulato. Qui usiamo 1.0.
    'BUG': {'FIRE': 0.5, 'GRASS': 2.0, 'FIGHTING': 0.5, 'POISON': 2.0, 'FLYING': 0.5, 'PSYCHIC': 2.0},
    'ROCK': {'FIRE': 2.0, 'ICE': 2.0, 'FIGHTING': 0.5, 'GROUND': 0.5, 'FLYING': 2.0, 'BUG': 2.0},
    'GHOST': {'NORMAL': 0.0, 'PSYCHIC': 0.0, 'GHOST': 2.0}, # Famoso bug: Lick (Ghost) non colpisce Psychic.
    'DRAGON': {'DRAGON': 2.0},
}

# Define strength-based encoding
POKEMON_RANKING = {
    'alakazam': 1, 'dragonite': 2, 'zapdos': 3, 'articuno': 4, 'gengar': 5,
    'snorlax': 6, 'lapras': 7, 'jolteon': 8, 'exeggutor': 9, 'rhydon': 10,
    'charizard': 11, 'starmie': 12, 'cloyster': 13, 'chansey': 14, 'victreebel': 15,
    'slowbro': 16, 'persian': 17, 'jynx': 18, 'golem': 19, 'tauros': 20
}

POKEMON_LIST = sorted(POKEMON_RANKING, key=lambda k: POKEMON_RANKING[k])
STATUS_WEIGHTS = {'slp': 5, 'frz': 5, 'par': 3, 'tox': 2, 'psn': 1}

KEY_ATTACKS = {
    'Body Slam', 'Hyper Beam', 'Earthquake', 'Blizzard', 
    'Ice Beam', 'Thunderbolt', 'Rock Slide', 'Surf', 
    'Self-Destruct', 'Explosion'
}