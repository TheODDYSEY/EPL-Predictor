import streamlit as st
import pandas as pd
import numpy as np

##########################################################################################################

# Gets cumulative match results arranged by teams and matchweek
def cum_results(ds):
    # Create a dictionary with team names as keys
    unique_teams = list(np.sort(ds['HomeTeam'].unique()))
    matchweeks = int(len(ds)/10)

    wins_dict = {}
    draws_dict = {}
    loss_dict = {}
    points_dict = {}

    for i in unique_teams:
        wins_dict[i] = []
        draws_dict[i] = []
        loss_dict[i] = []
        points_dict[i] = []

    # Create new columns for home wins and away wins for each fixture
    ds['HomeWins'] = np.where(ds['FTR'] == 'H', 1, 0)
    ds['AwayWins'] = np.where(ds['FTR'] == 'A', 1, 0)

    # Create new columns for home draws and away draws for each fixture
    ds['HomeDraws'] = np.where(ds['FTR'] == 'D', 1, 0)
    ds['AwayDraws'] = np.where(ds['FTR'] == 'D', 1, 0)

    # Create new columns for home losses and away losses for each fixture
    ds['HomeLosses'] = np.where(ds['FTR'] == 'A', 1, 0)
    ds['AwayLosses'] = np.where(ds['FTR'] == 'H', 1, 0)

    # Create new columns for homepoints and awaypoints for each fixture
    ds['HomePoints'] = np.where(ds['FTR'] == 'H', 3, np.where(ds['FTR'] == 'A', 0, 1))
    ds['AwayPoints'] = np.where(ds['FTR'] == 'A', 3, np.where(ds['FTR'] == 'H', 0, 1))
    
    # the value corresponding to keys is a list containing the match location.
    for i in range(len(ds)):
        HW = ds.iloc[i]['HomeWins']
        AW = ds.iloc[i]['AwayWins']
        HD = ds.iloc[i]['HomeDraws']
        AD = ds.iloc[i]['AwayDraws']
        HL = ds.iloc[i]['HomeLosses']
        AL = ds.iloc[i]['AwayLosses']
        HP = ds.iloc[i]['HomePoints']
        AP = ds.iloc[i]['AwayPoints']

        wins_dict[ds.iloc[i].HomeTeam].append(HW)
        wins_dict[ds.iloc[i].AwayTeam].append(AW)
        draws_dict[ds.iloc[i].HomeTeam].append(HD)
        draws_dict[ds.iloc[i].AwayTeam].append(AD)
        loss_dict[ds.iloc[i].HomeTeam].append(HL)
        loss_dict[ds.iloc[i].AwayTeam].append(AL)
        points_dict[ds.iloc[i].HomeTeam].append(HP)
        points_dict[ds.iloc[i].AwayTeam].append(AP)
    
    # Create a dataframe for league points where rows are teams and cols are matchweek.
    Wins = pd.DataFrame(data=wins_dict, index = [i for i in range(1, matchweeks+1)]).T
    Draws = pd.DataFrame(data=wins_dict, index = [i for i in range(1, matchweeks+1)]).T
    Loss = pd.DataFrame(data=loss_dict, index = [i for i in range(1, matchweeks+1)]).T
    Points = pd.DataFrame(data=points_dict, index = [i for i in range(1, matchweeks+1)]).T
    PrevResult = pd.DataFrame(data=points_dict, index = [i for i in range(1, matchweeks+1)]).T
    Form5M = pd.DataFrame(data=points_dict, index = [i for i in range(1, matchweeks+1)]).T

    # print(Points.head())

    Wins[0] = 0
    Draws[0] = 0
    Loss[0] = 0
    Points[0] = 0
    PrevResult[0] = 0
    Form5M[0] = 0

    # Calculate previous result and 5-match form
    for i in range(2, matchweeks+1):
        PrevResult[i] = Points[i-1]
        if i<6:
            Form5M[i] = 0
            for j in range(1,i):
                Form5M[i] = Form5M[i] + Points[j]
        else:
            Form5M[i] = Points[i-1] + Points[i-2] + Points[i-3] + Points[i-4] + Points[i-5]

    # Aggregate results upto each matchweek
    for i in range(2, matchweeks+1):
        Wins[i] = Wins[i] + Wins[i-1]
        Draws[i] = Draws[i] + Draws[i-1]
        Loss[i] = Loss[i] + Loss[i-1]
        Points[i] = Points[i] + Points[i-1]

    return Wins, Draws, Loss, Points, PrevResult, Form5M



# Gets the cumulative goals scored, conceded and difference arranged by teams and matchweek
def cum_goalstats(ds):

    matchweeks = int(len(ds)/10)
    unique_teams = list(np.sort(ds['HomeTeam'].unique()))

    # Create dictionaries with team names as keys
    gs_dict = {}
    gc_dict = {}
    gd_dict = {}
    sf_dict = {}
    stf_dict = {}
    sc_dict = {}
    stc_dict = {}

    for i in unique_teams:
        gs_dict[i] = []
        gc_dict[i] = []
        gd_dict[i] = []
        sf_dict[i] = []
        stf_dict[i] = []
        sc_dict[i] = []
        stc_dict[i] = []

    for i in range(len(ds)):
        HTGS = ds.iloc[i]['FTHG']
        ATGS = ds.iloc[i]['FTAG']
        HTGC = ds.iloc[i]['FTAG']
        ATGC = ds.iloc[i]['FTHG']
        HTSF = ds.iloc[i]['HS']
        ATSF = ds.iloc[i]['AS']
        HTSTF = ds.iloc[i]['HST']
        ATSTF = ds.iloc[i]['AST']
        HTSC = ds.iloc[i]['AS']
        ATSC = ds.iloc[i]['HS']
        HTSTC = ds.iloc[i]['AST']
        ATSTC = ds.iloc[i]['HST']

        gs_dict[ds.iloc[i].HomeTeam].append(HTGS)
        gs_dict[ds.iloc[i].AwayTeam].append(ATGS)
        gc_dict[ds.iloc[i].HomeTeam].append(HTGC)
        gc_dict[ds.iloc[i].AwayTeam].append(ATGC)
        gd_dict[ds.iloc[i].HomeTeam].append(HTGS - HTGC)
        gd_dict[ds.iloc[i].AwayTeam].append(ATGS - ATGC)
        sf_dict[ds.iloc[i].HomeTeam].append(HTSF)
        sf_dict[ds.iloc[i].AwayTeam].append(ATSF)
        stf_dict[ds.iloc[i].HomeTeam].append(HTSTF)
        stf_dict[ds.iloc[i].AwayTeam].append(ATSTF)
        sc_dict[ds.iloc[i].HomeTeam].append(HTSC)
        sc_dict[ds.iloc[i].AwayTeam].append(ATSC)
        stc_dict[ds.iloc[i].HomeTeam].append(HTSTC)
        stc_dict[ds.iloc[i].AwayTeam].append(ATSTC)
        
    # Create data frames where rows are teams and cols are matchweek.
    GoalsScored = pd.DataFrame(data=gs_dict, index = [i for i in range(1, matchweeks+1)]).T
    GoalsConceded = pd.DataFrame(data=gc_dict, index = [i for i in range(1, matchweeks+1)]).T
    GoalDifference = pd.DataFrame(data=gd_dict, index = [i for i in range(1, matchweeks+1)]).T
    ShotsFor = pd.DataFrame(data=sf_dict, index = [i for i in range(1, matchweeks+1)]).T
    ShotsTargetFor = pd.DataFrame(data=stf_dict, index = [i for i in range(1, matchweeks+1)]).T
    ShotsConceded = pd.DataFrame(data=sc_dict, index = [i for i in range(1, matchweeks+1)]).T
    ShotsTargetConceded = pd.DataFrame(data=stc_dict, index = [i for i in range(1, matchweeks+1)]).T
    GoalsScored[0] = 0
    GoalsConceded[0] = 0
    GoalDifference[0] = 0
    ShotsFor[0] = 0
    ShotsTargetFor[0] = 0
    ShotsConceded[0] = 0
    ShotsTargetConceded[0] = 0

    # Aggregate to get uptil that point
    for i in range(2, matchweeks + 1):
        GoalsScored[i] = GoalsScored[i] + GoalsScored[i-1]
        GoalsConceded[i] = GoalsConceded[i] + GoalsConceded[i-1]
        GoalDifference[i] = GoalDifference[i] + GoalDifference[i-1]
        ShotsFor[i] = ShotsFor[i] + ShotsFor[i-1]
        ShotsTargetFor[i] = ShotsTargetFor[i] + ShotsTargetFor[i-1]
        ShotsConceded[i] = ShotsConceded[i] + ShotsConceded[i-1]
        ShotsTargetConceded[i] = ShotsTargetConceded[i] + ShotsTargetConceded[i-1]

    return GoalsScored, GoalsConceded, GoalDifference, ShotsFor, ShotsTargetFor, ShotsConceded, ShotsTargetConceded



def get_league_pos(ds, p, gd, gs):
    unique_teams = list(np.sort(ds['HomeTeam'].unique()))
    alph_dict = dict(zip(unique_teams, range(20,0,-1)))
    alph = pd.DataFrame(data=alph_dict, index=[0]).T
    matchweeks = int(len(ds)/10)

    league_table = pd.DataFrame(index=unique_teams, columns=[i for i in range(1, matchweeks + 1)])

    # Rank teams by points, then goal difference, then goals scored, then alphabetically
    # Hack used: using weighted sum of criteria
    for i in range(1, matchweeks + 1):
        league_table[i] = 5000*p[i] + 100*gd[i] + 20*gs[i] + alph[0]

    # print(league_table[1])
    
    # Rank table values in decreasing order from 1 to 20
    league_table[0] = 0
    for i in range(1, matchweeks + 1):
        league_table[i] = league_table[i].rank(method='min', ascending=False).astype(int)
    
    return league_table



def get_stats(ds):
    GS, GC, GD, SF, STF, SC, STC = cum_goalstats(ds)
    W, D, L, P, PR, F5 = cum_results(ds)
    POS = get_league_pos(ds, P, GD, GS)

    j = 0
    MW = []
    
    HW = []
    AW = []
    HD = []
    AD = []
    HL = []
    AL = []
    HP = []
    AP = []

    HPR = []
    APR = []
    HF5 = []
    AF5 = []

    HTGS = []
    ATGS = []
    HTGC = []
    ATGC = []
    HTGD = []
    ATGD = []

    HTSF = []
    ATSF = []
    HTSTF = []
    ATSTF = []
    HTSC = []
    ATSC = []
    HTSTC = []
    ATSTC = []

    HPOS = []
    APOS = []

    HPR = []
    APR = []
    HF5 = []
    AF5 = []

    for i in range(len(ds)):
        ht = ds.iloc[i].HomeTeam
        at = ds.iloc[i].AwayTeam

        MW.append(j+1)

        HW.append(W.loc[ht][j])
        AW.append(W.loc[at][j])
        HD.append(D.loc[ht][j])
        AD.append(D.loc[at][j])
        HL.append(L.loc[ht][j])
        AL.append(L.loc[at][j])
        HP.append(P.loc[ht][j])
        AP.append(P.loc[at][j])

        HPR.append(PR.loc[ht][j])
        APR.append(PR.loc[at][j])
        HF5.append(F5.loc[ht][j])
        AF5.append(F5.loc[at][j])

        HTGS.append(GS.loc[ht][j])
        ATGS.append(GS.loc[at][j])
        HTGC.append(GC.loc[ht][j])
        ATGC.append(GC.loc[at][j])
        HTGD.append(GD.loc[ht][j])
        ATGD.append(GD.loc[at][j])

        HTSF.append(SF.loc[ht][j])
        ATSF.append(SF.loc[at][j])
        HTSTF.append(STF.loc[ht][j])
        ATSTF.append(STF.loc[at][j])
        HTSC.append(SC.loc[ht][j])
        ATSC.append(SC.loc[at][j])
        HTSTC.append(STC.loc[ht][j])
        ATSTC.append(STC.loc[at][j])

        HPOS.append(POS.loc[ht][j])
        APOS.append(POS.loc[at][j]) 
        
        if ((i + 1)% 10) == 0:
            j = j + 1
        
    ds['MW'] = MW

    ds['HP'] = HP
    ds['AP'] = AP
    ds['Pdiff'] = ds['HP'] - ds['AP']

    ds['HPOS'] = HPOS
    ds['APOS'] = APOS
    ds['POSdiff'] = ds['HPOS'] - ds['APOS']

    ds['HW'] = HW
    ds['AW'] = AW
    ds['HD'] = HD
    ds['AD'] = AD
    ds['HL'] = HL
    ds['AL'] = AL

    ds['HTGS'] = HTGS
    ds['ATGS'] = ATGS
    ds['HTGC'] = HTGC
    ds['ATGC'] = ATGC
    ds['HTGD'] = HTGD
    ds['ATGD'] = ATGD

    ds['HTSF'] = HTSF
    ds['ATSF'] = ATSF
    ds['HTSTF'] = HTSTF
    ds['ATSTF'] = ATSTF
    ds['HTSC'] = HTSC
    ds['ATSC'] = ATSC
    ds['HTSTC'] = HTSTC
    ds['ATSTC'] = ATSTC

    ds['HPR'] = HPR
    ds['APR'] = APR
    ds['HF5'] = HF5
    ds['AF5'] = AF5

    ds['HTHGS'] = ds.groupby(['HomeTeam'])['FTHG'].cumsum() - ds['FTHG']
    ds['ATAGS'] = ds.groupby(['AwayTeam'])['FTAG'].cumsum() - ds['FTAG']
    ds['HTHGC'] = ds.groupby(['HomeTeam'])['FTAG'].cumsum() - ds['FTAG']
    ds['ATAGC'] = ds.groupby(['AwayTeam'])['FTHG'].cumsum() - ds['FTHG']
    ds['HTHSF'] = ds.groupby(['HomeTeam'])['HS'].cumsum() - ds['HS']
    ds['ATASF'] = ds.groupby(['AwayTeam'])['AS'].cumsum() - ds['AS']
    ds['HTHSC'] = ds.groupby(['HomeTeam'])['AS'].cumsum() - ds['AS']
    ds['ATASC'] = ds.groupby(['AwayTeam'])['HS'].cumsum() - ds['HS']
    ds['HTHSTF'] = ds.groupby(['HomeTeam'])['HST'].cumsum() - ds['HST']
    ds['ATASTF'] = ds.groupby(['AwayTeam'])['AST'].cumsum() - ds['AST']
    ds['HTHSTC'] = ds.groupby(['HomeTeam'])['AST'].cumsum() - ds['AST']
    ds['ATASTC'] = ds.groupby(['AwayTeam'])['HST'].cumsum() - ds['HST']

    ds['HTHP'] = ds.groupby(['HomeTeam'])['HomePoints'].cumsum() - ds['HomePoints']
    ds['ATAP'] = ds.groupby(['AwayTeam'])['AwayPoints'].cumsum() - ds['AwayPoints']
    ds['HTHW'] = ds.groupby(['HomeTeam'])['HomeWins'].cumsum() - ds['HomeWins']
    ds['ATAW'] = ds.groupby(['AwayTeam'])['AwayWins'].cumsum() - ds['AwayWins']
    ds['HTHD'] = ds.groupby(['HomeTeam'])['HomeDraws'].cumsum() - ds['HomeDraws']
    ds['ATAD'] = ds.groupby(['AwayTeam'])['AwayDraws'].cumsum() - ds['AwayDraws']
    ds['HTHL'] = ds.groupby(['HomeTeam'])['HomeLosses'].cumsum() - ds['HomeLosses']
    ds['ATAL'] = ds.groupby(['AwayTeam'])['AwayLosses'].cumsum() - ds['AwayLosses']

    ds['avg_HGPG'] = (ds['FTHG'].cumsum() - ds['FTHG'])/(ds.index)
    ds['avg_AGPG'] = (ds['FTAG'].cumsum() - ds['FTAG'])/(ds.index)

    ds.drop(['HomeWins','AwayWins','HomeDraws','AwayDraws','HomeLosses','AwayLosses','HomePoints','AwayPoints'], axis=1, inplace=True)
    
    return ds



# Normalize cumulative stats by Matchweek
def norm_mw(ds):
    cols = ['HP', 'AP', 'HW', 'AW', 'HD', 'AD', 'HL', 'AL', 'HTGS', 
            'ATGS', 'HTGC', 'ATGC', 'HTGD', 'ATGD', 'HTSF', 'ATSF', 
            'HTSTF', 'ATSTF', 'HTSC', 'ATSC', 'HTSTC', 'ATSTC']
    
    ha_cols = ['HTHGS', 'ATAGS', 'HTHGC', 'ATAGC', 'HTHSF', 'ATASF', 
               'HTHSC', 'ATASC', 'HTHSTF', 'ATASTF', 'HTHSTC', 'ATASTC', 
               'HTHP', 'ATAP', 'HTHW', 'ATAW', 'HTHD', 'ATAD', 'HTHL', 'ATAL']

    ds['MW'] = ds['MW'].astype(float)
    for col in cols:
        ds[col] /= (ds['MW']-1)

    for col in ha_cols:
        ds[col] /= (0.5*(ds['MW']-1))




def engg(ds):
    req_cols = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR', 'HS', 'AS', 'HST', 'AST']
    ds = ds[req_cols]
    stats_ds = get_stats(ds)
    engg_ds = stats_ds.iloc[50:]
    norm_mw(engg_ds)
    engg_ds.drop(['FTR','HS','AS','HST','AST','MW'], axis=1, inplace=True)
    return engg_ds


## Use this block if the data files are already generated
data = []
data.append(pd.read_csv('engg_data/epl1314.csv'))
data.append(pd.read_csv('engg_data/epl1415.csv'))
data.append(pd.read_csv('engg_data/epl1516.csv'))
data.append(pd.read_csv('engg_data/epl1617.csv'))
data.append(pd.read_csv('engg_data/epl1718.csv'))
data.append(pd.read_csv('engg_data/epl1819.csv'))
data.append(pd.read_csv('engg_data/epl1920.csv'))
data.append(pd.read_csv('engg_data/epl2021.csv'))
data.append(pd.read_csv('engg_data/epl2122.csv'))
data.append(pd.read_csv('engg_data/epl2223.csv'))
data.append(pd.read_csv('engg_data/epl2324.csv'))


# Extract the top n features for each target variable - Home xG and Away xG

fthg_best_n = []
ftag_best_n = []
for i in range(11):
    corr_matrix = data[i].corr(numeric_only=True)
    fthg_corr = corr_matrix['FTHG'].abs().sort_values(ascending=False)
    top_corr_hg = fthg_corr[1:7]
    fthg_best_n.append(top_corr_hg.index.array)
    
    ftag_corr = corr_matrix['FTAG'].abs().sort_values(ascending=False)
    top_corr_ag = ftag_corr[1:7]
    ftag_best_n.append(top_corr_ag.index.array)


    
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())



#########################################################################################################

# RandomForest model
from sklearn.ensemble import RandomForestRegressor
hg_model_rf = RandomForestRegressor(n_estimators=94, max_depth=2, n_jobs=-1, random_state=42)
ag_model_rf = RandomForestRegressor(n_estimators=280, max_depth=2, n_jobs=-1, random_state=42)


#########################################################################################################

# XGBoost model
from xgboost import XGBRegressor
hg_model_xgb = XGBRegressor(n_estimators=7, max_depth=2)
ag_model_xgb = XGBRegressor(n_estimators=13, max_depth=1)


#########################################################################################################

# Poisson model
from scipy.stats import norm

def calc_xg(ds):
    hxg_pred = ds['HTHGS'] * ds['ATAGC'] / ds['avg_HGPG']
    axg_pred = ds['ATAGS'] * ds['HTHGC'] / ds['avg_AGPG']
    return hxg_pred, axg_pred

def erf(x):
    return 2 * norm.cdf(x * np.sqrt(2)) - 1

def calc_prob(ds):
    hxg_pred, axg_pred = calc_xg(ds)
    prob_hwin = 0.5 * (1 + erf((hxg_pred - axg_pred) / (np.sqrt(2) * np.sqrt(ds['HTHGS'] + ds['ATAGC']))))
    prob_awin = 0.5 * (1 + erf((axg_pred - hxg_pred) / (np.sqrt(2) * np.sqrt(ds['ATAGS'] + ds['HTHGC']))))
    #prob_draw = 1 - prob_hwin - prob_awin
    return prob_hwin, prob_awin#, prob_draw


#########################################################################################################

# Web app code

st.set_page_config(page_icon="img/wizard.png", page_title="EPL Predictor", layout="centered")
st.image("img/epl-long.png", use_column_width=True)

st.write("""
         # ⚽ EPL Expected Goals projector! 🧭
         """)
st.write('---')



cols = st.columns(2)
season = cols[0].selectbox('EPL Season', ('2023-24 season','2022-23 season','2021-22 season','2020-21 season','2019-20 season','2018-19 season','2017-18 season','2016-17 season','2015-16 season','2014-15 season','2013-14 season'))
model = cols[1].selectbox('Projection Model', ('Poisson', 'RandomForest','XGBoost','Ensemble'))

seasons = ['2013-14 season', '2014-15 season', '2015-16 season', '2016-17 season', '2017-18 season', '2018-19 season', '2019-20 season', '2020-21 season', '2021-22 season', '2022-23 season', '2023-24 season'] 
train_data = data[seasons.index(season)]
unique_teams = list(np.sort(train_data['HomeTeam'].unique()))

cols = st.columns(2)
hometeam = cols[0].selectbox('Home Team', tuple(unique_teams))
awayteam = cols[1].selectbox('Away Team', tuple(unique_teams[::-1]))
st.write('---')

def parse_input(model, season, hometeam, awayteam):

    # Prepare training data
    X_train_hg = train_data.loc[:, fthg_best_n[seasons.index(season)]]
    X_train_ag = train_data.loc[:, ftag_best_n[seasons.index(season)]]
    y_train_hg = train_data['FTHG']
    y_train_ag = train_data['FTAG']

    # Prepare sample data. If the match has been played, save the score for comparison purposes
    match_tuple = train_data[(train_data['HomeTeam'] == hometeam) & (train_data['AwayTeam'] == awayteam)]

    if match_tuple.empty:
        match_completed=False
        true_hg = 0
        true_ag = 0

        ht_match_tuple = train_data[train_data['HomeTeam'] == hometeam]
        ht_match_tuple = ht_match_tuple.iloc[-1, :]
        hg_sample_data = ht_match_tuple.loc[fthg_best_n[seasons.index(season)]]
        hg_sample_data = hg_sample_data.values.reshape(1, -1)

        at_match_tuple = train_data[train_data['AwayTeam'] == awayteam]
        at_match_tuple = at_match_tuple.iloc[-1, :]
        ag_sample_data = at_match_tuple.loc[ftag_best_n[seasons.index(season)]]
        ag_sample_data = ag_sample_data.values.reshape(1, -1) 

        pois_match_tuple = pd.DataFrame(columns=['HTHGS','ATAGS','HTHGC','ATAGC','avg_HGPG','avg_AGPG'])
        pois_match_tuple.loc[0] = [ht_match_tuple['HTHGS'], at_match_tuple['ATAGS'], ht_match_tuple['HTHGC'], at_match_tuple['ATAGC'], ht_match_tuple['avg_HGPG'], at_match_tuple['avg_AGPG']]

    else:
        match_completed=True
        true_hg = match_tuple['FTHG']
        true_hg = true_hg.values[0]
        true_ag = match_tuple['FTAG']
        true_ag = true_ag.values[0]

        hg_sample_data = match_tuple.loc[:, fthg_best_n[seasons.index(season)]]
        hg_sample_data = hg_sample_data.values.reshape(1, -1)
        ag_sample_data = match_tuple.loc[:, ftag_best_n[seasons.index(season)]]
        ag_sample_data = ag_sample_data.values.reshape(1, -1)

        pois_match_tuple = match_tuple.loc[:, ['HTHGS','ATAGS','HTHGC','ATAGC','avg_HGPG','avg_AGPG']]


    # Choose and run the model
    if(model == 'Poisson'):

        pred_hg, pred_ag = calc_xg(pois_match_tuple)
        pred_hg = pred_hg.values[0]
        pred_ag = pred_ag.values[0]
        pred_hg_rmse = rmse(pred_hg, true_hg)
        pred_ag_rmse = rmse(pred_ag, true_ag)

        prob_hgwin, prob_agwin = calc_prob(pois_match_tuple)
        prob_hgwin = prob_hgwin[0]
        prob_agwin = prob_agwin[0]

        return pred_hg, pred_ag, true_hg, true_ag, pred_hg_rmse, pred_ag_rmse, match_completed, prob_hgwin, prob_agwin


    elif(model == 'XGBoost'):
            
        hg_model_xgb.fit(X_train_hg, y_train_hg)
        ag_model_xgb.fit(X_train_ag, y_train_ag)

        # Make predictions
        pred_hg = hg_model_xgb.predict(hg_sample_data)
        pred_hg = pred_hg[0].astype(float)
        pred_ag = ag_model_xgb.predict(ag_sample_data)
        pred_ag = pred_ag[0].astype(float)
        pred_hg_rmse = rmse(pred_hg, true_hg)
        pred_ag_rmse = rmse(pred_ag, true_ag)

        return pred_hg, pred_ag, true_hg, true_ag, pred_hg_rmse, pred_ag_rmse, match_completed


    elif(model == 'RandomForest'):

        hg_model_rf.fit(X_train_hg, y_train_hg)
        ag_model_rf.fit(X_train_ag, y_train_ag)

        # Make predictions
        pred_hg = hg_model_rf.predict(hg_sample_data)
        pred_hg = pred_hg[0].astype(float)
        pred_ag = ag_model_rf.predict(ag_sample_data)
        pred_ag = pred_ag[0].astype(float)
        pred_hg_rmse = rmse(pred_hg, true_hg)
        pred_ag_rmse = rmse(pred_ag, true_ag)

        return pred_hg, pred_ag, true_hg, true_ag, pred_hg_rmse, pred_ag_rmse, match_completed
    
    else:
        # Ensemble the predictions
        hg_model_rf.fit(X_train_hg, y_train_hg)
        ag_model_rf.fit(X_train_ag, y_train_ag)
        hg_model_xgb.fit(X_train_hg, y_train_hg)
        ag_model_xgb.fit(X_train_ag, y_train_ag)

        # Make predictions
        pred_hg_rf = hg_model_rf.predict(hg_sample_data)
        pred_hg_rf = pred_hg_rf[0].astype(float)
        pred_ag_rf = ag_model_rf.predict(ag_sample_data)
        pred_ag_rf = pred_ag_rf[0].astype(float)
        pred_hg_xgb = hg_model_xgb.predict(hg_sample_data)
        pred_hg_xgb = pred_hg_xgb[0].astype(float)
        pred_ag_xgb = ag_model_xgb.predict(ag_sample_data)
        pred_ag_xgb = pred_ag_xgb[0].astype(float)

        pred_hg = (pred_hg_rf + pred_hg_xgb)/2
        pred_ag = (pred_ag_rf + pred_ag_xgb)/2
        pred_hg_rmse = rmse(pred_hg, true_hg)
        pred_ag_rmse = rmse(pred_ag, true_ag)

        return pred_hg, pred_ag, true_hg, true_ag, pred_hg_rmse, pred_ag_rmse, match_completed



def runapp(model, season, hometeam, awayteam):

    if(model == 'Poisson'):
        pred_hg, pred_ag, true_hg, true_ag, pred_hg_rmse, pred_ag_rmse, match_completed, prob_hgwin, prob_agwin = parse_input(model, season, hometeam, awayteam)
    else:
        pred_hg, pred_ag, true_hg, true_ag, pred_hg_rmse, pred_ag_rmse, match_completed = parse_input(model, season, hometeam, awayteam)

    st.write('#### Projected Score: (Goals)')
    cols = st.columns(2)
    cols[0].write(f'{hometeam}: `{round(pred_hg, 2)}`')
    cols[1].write(f'{awayteam}: `{round(pred_ag, 2)}`')
    st.write('---')

    if model == 'Poisson':
        st.write('#### Win Probability', use_column_width=True)
        cols = st.columns(2)
        cols[0].write(f'{hometeam}: `{round(prob_hgwin * 100, 2)} %`')
        cols[1].write(f'{awayteam}: `{round(prob_agwin * 100, 2)} %`')
        st.write('---')


    if match_completed:
        st.write('#### Actual Score: (Goals) ')
        cols = st.columns(2)
        cols[0].write(f'{hometeam}: `{true_hg}`')
        cols[1].write(f'{awayteam}: `{true_ag}`')
        st.write('---')

        st.write('#### Root Mean Square Error')
        cols = st.columns(2)
        cols[0].write(f'{hometeam}: `{round(pred_hg_rmse, 2)}`')
        cols[1].write(f'{awayteam}: `{round(pred_ag_rmse, 2)}`')
        
runapp(model, season, hometeam, awayteam)
