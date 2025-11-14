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

def get_type_effectiveness(move_type, target_types):
    if move_type not in TYPE_CHART_GEN1:
        return 1.0
    
    multiplier = 1.0
    chart_for_move = TYPE_CHART_GEN1[move_type]
    
    for target_type in target_types:
        if target_type in chart_for_move:
            multiplier *= chart_for_move[target_type]
            
    return multiplier

META_THREATS_GEN1 = {
    'Snorlax', 'Tauros', 'Chansey', 'Alakazam', 'Starmie', 'Exeggutor', 
    'Zapdos', 'Jolteon', 'Rhydon', 'Golem', 'Lapras'
}

STATUS_MOVES = {'Thunder Wave', 'Sleep Powder', 'Sing', 'Toxic', 'Lovely Kiss', 'Spore', 'Stun Spore', 'Glare'}
SETUP_MOVES = {'Amnesia', 'Swords Dance', 'Agility', 'Growth'}