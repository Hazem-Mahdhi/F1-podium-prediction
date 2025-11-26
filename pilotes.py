import requests
import json 

BASE_URL = "https://api.openf1.org/v1"

def get_driver_acronym(driver_number, session_key=None):
    """
    Récupère l'acronyme (ex: VER) d'un pilote via l'API.
    """
    url = f"{BASE_URL}/drivers"
    params = {'driver_number': driver_number}
    if session_key:
        params['session_key'] = session_key
        
    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        if data:
            # On retourne l'acronyme (ex: "VER")
            return data[0]['name_acronym']
    except:
        pass
    
    return "???" # Si non trouvé

def get_pilotes_number(pilote_full_name):
    """
    Récupère le numéro de pilote pour une saison et un nom de pilote donnés.

    :param pilote_full_name: Nom complet du pilote (ex: "Lewis HAMILTON").
    :return: Le numéro du pilote si trouvé, sinon None.
    """
    
    pilotes_url = f"{BASE_URL}/drivers"
    params_pilotes = {
        'full_name': pilote_full_name
    }
    
    try:
        response_pilotes = requests.get(pilotes_url, params=params_pilotes)
        response_pilotes.raise_for_status() 
        pilotes = response_pilotes.json()
    except requests.exceptions.RequestException as e:
        print(f"Erreur de connexion lors de la recherche du pilote : {e}")
        return None

    if not pilotes:
        print(f"Aucun pilote trouvé pour le nom '{pilote_full_name}'.")
        return None

    pilote_number = pilotes[0]['driver_number']
    print(f"Pilote trouvé : {pilote_full_name} avec le numéro {pilote_number}")

    return pilote_number

def get_pilote_name(pilote_number):
    """
    Récupère le nom de pilote pour un numéro de pilote donné.

    :param pilote_number: Numéro du pilote (ex: 44).
    :return: Le nom du pilote si trouvé, sinon None.
    """
    
    pilotes_url = f"{BASE_URL}/drivers"
    params_pilotes = {
        'driver_number': pilote_number
    }
    
    try:
        response_pilotes = requests.get(pilotes_url, params=params_pilotes)
        response_pilotes.raise_for_status() 
        pilotes = response_pilotes.json()
    except requests.exceptions.RequestException as e:
        print(f"Erreur de connexion lors de la recherche du pilote : {e}")
        return None

    if not pilotes:
        print(f"Aucun pilote trouvé pour le numéro '{pilote_number}'.")
        return None

    pilote_name = pilotes[0]['full_name']
    print(f"Pilote trouvé : Numéro {pilote_number} correspond à {pilote_name}")
    
    return pilote_name

# Exemple d'utilisation
if __name__ == "__main__":
    pilote_full_name = "Max VERSTAPPEN"
    pilote_number = get_pilotes_number(pilote_full_name)
    
    if pilote_number:
        print(f"\nLe numéro de {pilote_full_name} est : {pilote_number}")
    
    pilote_number_to_lookup = 1
    pilote_name = get_pilote_name(pilote_number_to_lookup)
    
    if pilote_name:
        print(f"\nLe pilote avec le numéro {pilote_number_to_lookup} est : {pilote_name}")

def get_drivers_in_session(session_key):
    """Récupère la liste des numéros de pilotes présents dans une session"""
    url = f"{BASE_URL}/drivers"
    params = {'session_key': session_key}
    try:
        response = requests.get(url, params=params, timeout=10)
        drivers = response.json()
        # On retourne la liste des numéros (ex: [1, 16, 44, ...])
        return [d['driver_number'] for d in drivers]
    except:
        return []