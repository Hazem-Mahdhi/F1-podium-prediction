import requests
import json 

BASE_URL = "https://api.openf1.org/v1"

def get_session_list(**filters):
    """
    Récupère la liste des sessions correspondant aux filtres dynamiques.
    Exemple: get_session_list(year=2023, country_name="Singapore")
    """
    url = f"{BASE_URL}/sessions"
    
    try:
        # On passe directement le dictionnaire d'arguments
        response = requests.get(url, params=filters) 
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"Erreur API : {e}")
        return []