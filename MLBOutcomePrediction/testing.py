import pymongo
import pandas
import numpy as np
import datahelp
# MongoDB Connections
client = pymongo.MongoClient('mongodb://localhost:27017')
db = client['MLB_DB']
gameSummaryDB = db['GameSummary']
gamePlayerDataDB = db['GamePlayerData']
playerProjectionsDB = db['PlayerProjectedStats']

def getStatAvg(stats_list, projections_str, player_projected_stats_doc):
    stat_averages = [0] * len(stats_list)
    count = 0
    for player in player_projected_stats_doc[projections_str].values():
        stat_index = 0
        for stat in stats_list:
            try:
                stat_val = float(player[stat])
            except:
                stat_val = 0
            stat_averages[stat_index] = stat_averages[stat_index] + stat_val
            stat_index = stat_index + 1
        count = count + 1
    stat_averages = [i/count for i in stat_averages]
    return stat_averages

columns = datahelp.columns
data = []
for document in gameSummaryDB.find():
    game_id = document['game_id_num']
    for player_data in gamePlayerDataDB.find({'game_id_num': str(game_id)}):
        home_umpire_name = [player_data['umpires']['home']['id']]
    for player_projections in playerProjectionsDB.find({'game_id_num': str(game_id)}):
        # Get Stat averages
        home_batting_avg = getStatAvg(datahelp.batting_stats, 'home_batting_projections', player_projections)
        away_batting_avg = getStatAvg(datahelp.batting_stats, 'away_batting_projections', player_projections)
        home_pitching = getStatAvg(datahelp.pitching_stats, 'home_pitcher_projections', player_projections)
        away_pitching = getStatAvg(datahelp.pitching_stats, 'away_pitcher_projections', player_projections)
    home_win = [1 if document['home_score']>document['away_score'] else 0]
    data_row = home_umpire_name + home_batting_avg + away_batting_avg + home_pitching + away_pitching + home_win
    data.append(data_row)
x = 1