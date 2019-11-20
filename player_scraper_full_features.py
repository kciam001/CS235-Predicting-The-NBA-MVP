import time
import re
from collections import defaultdict
import numpy as np
import requests
from bs4 import BeautifulSoup
import csv
import unicodedata

#gets CURRENT stats for given player
def Get_Player_Stats(player, season):

    #find url of player profile
    last_name = player[player.find(' ')+1:]
    letter = last_name[0].lower()
    url = "https://www.basketball-reference.com/players/" + letter
    res = requests.get(url)
    soup = BeautifulSoup(res.text, features="html5lib")
    #find table
    player_table = soup.find(attrs={'id': 'players'})
    #find rows
    rows = player_table.findAll('tr')
    profile = " "
    for row in rows:
        #find name
        head = row.find('th')
        name = head.find('a')
        if(name is not None):
            #removes accents (looking at you Nikola Jokić)
            website_name = Strip_Accents(name.contents[0].lower())
            if(website_name == player.lower()):
                profile = name['href']
    if(profile == " "):
        print("Player not found.")
        return None
    else:
        url = "https://www.basketball-reference.com" + profile

    res = requests.get(url)
    soup = BeautifulSoup(res.text, features="html5lib")
    stats = {}
    stats['Player'] = player
    
    #find per game stats
    per_game = soup.find(attrs={'id': 'all_per_game'})
    for row in per_game.findAll("tr"):
        if 'id' in row.attrs and row.attrs['id'] == "per_game." + season:
            stats['fga'] = float(row.find('td', attrs={'data-stat': 'fga_per_g'}).text)
            stats['fg3a'] = float(row.find('td', attrs={'data-stat': 'fg3a_per_g'}).text)
            stats['fta'] = float(row.find('td', attrs={'data-stat': 'fta_per_g'}).text)
            stats['g'] = float(row.find('td', attrs={'data-stat': 'g'}).text)
            stats['mp_per_g'] = float(row.find('td', attrs={'data-stat': 'mp_per_g'}).text)
            stats['pts_per_g'] = float(row.find('td', attrs={'data-stat': 'pts_per_g'}).text)
            stats['trb_per_g'] = float(row.find('td', attrs={'data-stat': 'trb_per_g'}).text)
            stats['ast_per_g'] = float(row.find('td', attrs={'data-stat': 'ast_per_g'}).text)
            stats['stl_per_g'] = float(row.find('td', attrs={'data-stat': 'stl_per_g'}).text)
            stats['blk_per_g'] = float(row.find('td', attrs={'data-stat': 'blk_per_g'}).text)
            stats['fg_pct'] = float(row.find('td', attrs={'data-stat': 'fg_pct'}).text)
            stats['fg3_pct'] = float(row.find('td', attrs={'data-stat': 'fg3_pct'}).text)
            stats['ft_pct'] = float(row.find('td', attrs={'data-stat': 'ft_pct'}).text)
            break
    #find advanced stats
    advanced_stats = soup.find(attrs={'id': 'all_advanced'})
    for child in advanced_stats.children:
        if "table_outer_container" in child:
            other_soup = BeautifulSoup(child, features="html5lib")
            rows = other_soup.findAll("tr")
    for row in rows:
        if 'id' in row.attrs and row.attrs['id'] == "advanced." + season:
            stats['ws'] = float(row.find('td', attrs={'data-stat': 'ws'}).text)
            stats['ws_per_48'] = float(row.find('td', attrs={'data-stat': 'ws_per_48'}).text)
            stats['per'] = float(row.find('td', attrs={'data-stat': 'per'}).text)
            stats['ts_pct'] = float(row.find('td', attrs={'data-stat': 'ts_pct'}).text)
            stats['usg_pct'] = float(row.find('td', attrs={'data-stat': 'usg_pct'}).text)
            stats['bpm'] = float(row.find('td', attrs={'data-stat': 'bpm'}).text)     
    return stats

def Strip_Accents(text):
    try:
        text = unicode(text, 'utf-8')
    except NameError:
        pass

    text = unicodedata.normalize('NFD', text)\
            .encode('ascii', 'ignore')\
            .decode("utf-8")
    return str(text)



print("Enter player full name (e.g. James Harden): ")
player = input()
new_data = defaultdict(list)
season = "2020"

player_stats = Get_Player_Stats(player, season)
if player_stats is not None:
    print(player_stats)
    filename = player.replace(" ", "_")
    with open(filename + '_stats.csv', 'w') as f: 
        w = csv.DictWriter(f, player_stats.keys())
        w.writeheader()
        w.writerow(player_stats)
