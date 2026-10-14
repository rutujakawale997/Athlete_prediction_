import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("/kaggle/input/nfl-draft/train.csv")
test  = pd.read_csv("/kaggle/input/nfl-draft/test.csv")
sub   = pd.read_csv("/kaggle/input/nfl-draft/sample_submission.csv")

numeric_cols = ['Age','Height','Weight','Sprint_40yd','Vertical_Jump',
                'Bench_Press_Reps','Broad_Jump','Agility_3cone','Shuttle']
drill_cols   = ['Sprint_40yd','Vertical_Jump','Bench_Press_Reps',
                'Broad_Jump','Agility_3cone','Shuttle']

X_train = train.copy()
X_test  = test.copy()

for col in ['Age'] + drill_cols:
    X_train[f'{col}_missing'] = X_train[col].isna().astype(int)
    X_test[f'{col}_missing']  = X_test[col].isna().astype(int)
X_train['n_drills_missing'] = X_train[drill_cols].isna().sum(axis=1)
X_test['n_drills_missing']  = X_test[drill_cols].isna().sum(axis=1)

pos_median = X_train.groupby("Position")[numeric_cols].median()
for col in numeric_cols:
    gm = X_train[col].median()
    mask = X_train[col].isna()
    X_train.loc[mask, col] = X_train.loc[mask,"Position"].map(pos_median[col]).fillna(gm)
    mask = X_test[col].isna()
    X_test.loc[mask, col]  = X_test.loc[mask,"Position"].map(pos_median[col]).fillna(gm)

global_mean = train['Drafted'].mean()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

school_oof = np.zeros(len(train))
for tr_idx, val_idx in skf.split(train, train['Drafted']):
    stats = train.iloc[tr_idx].groupby('School')['Drafted'].agg(['sum','count'])
    stats['rate'] = (stats['sum'] + global_mean*10) / (stats['count'] + 10)
    school_oof[val_idx] = train.iloc[val_idx]['School'].map(stats['rate']).fillna(global_mean)
X_train['School_Encoded'] = school_oof
stats_full = train.groupby('School')['Drafted'].agg(['sum','count'])
stats_full['rate'] = (stats_full['sum'] + global_mean*10) / (stats_full['count'] + 10)
X_test['School_Encoded'] = X_test['School'].map(stats_full['rate']).fillna(global_mean)

pos_oof = np.zeros(len(train))
for tr_idx, val_idx in skf.split(train, train['Drafted']):
    stats = train.iloc[tr_idx].groupby('Position')['Drafted'].agg(['sum','count'])
    stats['rate'] = (stats['sum'] + global_mean*5) / (stats['count'] + 5)
    pos_oof[val_idx] = train.iloc[val_idx]['Position'].map(stats['rate']).fillna(global_mean)
X_train['Position_Encoded'] = pos_oof
stats_full = train.groupby('Position')['Drafted'].agg(['sum','count'])
stats_full['rate'] = (stats_full['sum'] + global_mean*5) / (stats_full['count'] + 5)
X_test['Position_Encoded'] = X_test['Position'].map(stats_full['rate']).fillna(global_mean)

le = LabelEncoder()
X_train['Position_Type'] = le.fit_transform(train['Position_Type'])
X_test['Position_Type']  = le.transform(test['Position_Type'])
X_train['Player_Type'] = le.fit_transform(train['Player_Type'])
X_test['Player_Type']  = le.transform(test['Player_Type'])

year_oof = np.zeros(len(train))
for tr_idx, val_idx in skf.split(train, train['Drafted']):
    yr = train.iloc[tr_idx].groupby('Year')['Drafted'].mean()
    year_oof[val_idx] = train.iloc[val_idx]['Year'].map(yr).fillna(global_mean)
X_train['year_draft_rate'] = year_oof
X_test['year_draft_rate']  = test['Year'].map(train.groupby('Year')['Drafted'].mean()).fillna(global_mean)

pos_year_oof = np.zeros(len(train))
for tr_idx, val_idx in skf.split(train, train['Drafted']):
    stats = train.iloc[tr_idx].groupby(['Position','Year'])['Drafted'].mean()
    pos_year_oof[val_idx] = (train.iloc[val_idx]
                             .set_index(['Position','Year']).index.map(stats).fillna(global_mean))
X_train['pos_year_draft_rate'] = pos_year_oof
stats_full = train.groupby(['Position','Year'])['Drafted'].mean()
X_test['pos_year_draft_rate'] = (test.set_index(['Position','Year'])
                                  .index.map(stats_full).fillna(global_mean))

sy_count = train.groupby(['School','Year'])['Drafted'].count()
X_train['school_year_volume'] = train.set_index(['School','Year']).index.map(sy_count).fillna(0).values
X_test['school_year_volume']  = test.set_index(['School','Year']).index.map(sy_count).fillna(0).values

pos_stats = X_train.groupby('Position')[numeric_cols].agg(['mean','std'])
for col in ['Sprint_40yd','Weight','Height','Bench_Press_Reps',
            'Vertical_Jump','Broad_Jump','Agility_3cone','Shuttle']:
    pm = X_train['Position'].map(pos_stats[col]['mean'])
    ps = X_train['Position'].map(pos_stats[col]['std'])
    X_train[f'{col}_vs_pos'] = (X_train[col] - pm) / (ps + 1e-6)
    X_test[f'{col}_vs_pos']  = (X_test[col] - X_test['Position'].map(pos_stats[col]['mean'])) / \
                                (X_test['Position'].map(pos_stats[col]['std']) + 1e-6)

X_train['speed_score'] = (X_train['Weight'] * 200) / (X_train['Sprint_40yd'] ** 4)
X_test['speed_score']  = (X_test['Weight']  * 200) / (X_test['Sprint_40yd']  ** 4)
for df in [X_train, X_test]:
    df['combine_score'] = (
        -df['Sprint_40yd_vs_pos']*2.0 + df['Vertical_Jump_vs_pos']*1.0 +
         df['Broad_Jump_vs_pos']*1.0  - df['Agility_3cone_vs_pos']*1.5 -
         df['Shuttle_vs_pos']*1.0     + df['Bench_Press_Reps_vs_pos']*0.5)

for col in ['Sprint_40yd_vs_pos','Vertical_Jump_vs_pos','Broad_Jump_vs_pos']:
    X_train[f'{col}_elite'] = (X_train[col].abs() > 1.0).astype(int)
    X_test[f'{col}_elite']  = (X_test[col].abs()  > 1.0).astype(int)
X_train['n_elite_drills'] = X_train[['Sprint_40yd_vs_pos_elite','Vertical_Jump_vs_pos_elite','Broad_Jump_vs_pos_elite']].sum(axis=1)
X_test['n_elite_drills']  = X_test[['Sprint_40yd_vs_pos_elite','Vertical_Jump_vs_pos_elite','Broad_Jump_vs_pos_elite']].sum(axis=1)
X_train['bmi'] = X_train['Weight'] / (X_train['Height'] ** 2)
X_test['bmi']  = X_test['Weight']  / (X_test['Height']  ** 2)

for df in [X_train, X_test]:
    df['hard_missing']     = ((df['Age_missing']==1) & (df['n_drills_missing']>=3)).astype(int)
    df['complete_profile'] = ((df['Age_missing']==0) & (df['n_drills_missing']==0)).astype(int)
age_miss_pos = X_train.groupby('Position')['Age_missing'].mean()
X_train['pos_age_missing_rate'] = X_train['Position'].map(age_miss_pos)
X_test['pos_age_missing_rate']  = X_test['Position'].map(age_miss_pos).fillna(0)
school_vol = train.groupby('School').size()
X_train['school_total_volume'] = X_train['School'].map(school_vol)
X_test['school_total_volume']  = X_test['School'].map(school_vol).fillna(1)

pt_year_oof = np.zeros(len(train))
for tr_idx, val_idx in skf.split(train, train['Drafted']):
    ft = train.iloc[tr_idx]
    pt_yr = ft.groupby(['Player_Type','Year'])['Drafted'].mean()
    pt_mn = ft.groupby('Player_Type')['Drafted'].mean()
    val   = train.iloc[val_idx]
    mapped   = val.set_index(['Player_Type','Year']).index.map(pt_yr)
    fallback = val['Player_Type'].map(pt_mn).values
    pt_year_oof[val_idx] = np.where(pd.isna(mapped), fallback, mapped)
X_train['pt_year_rate'] = pt_year_oof
pt_yr_full    = train.groupby(['Player_Type','Year'])['Drafted'].mean()
pt_mn_full    = train.groupby('Player_Type')['Drafted'].mean()
mapped_test   = test.set_index(['Player_Type','Year']).index.map(pt_yr_full)
fallback_test = test['Player_Type'].map(pt_mn_full).values
X_test['pt_year_rate'] = np.where(pd.isna(mapped_test), fallback_test, mapped_test)

print("Features ready:", X_train.shape)

features = [
    'Year','Age','Height','Weight','Sprint_40yd','Vertical_Jump','Bench_Press_Reps',
    'Broad_Jump','Agility_3cone','Shuttle','Position_Type','Player_Type',
    'combine_score','bmi','speed_score','n_elite_drills',
    'hard_missing','pos_age_missing_rate','complete_profile',
    'Age_missing','Sprint_40yd_missing','Vertical_Jump_missing',
    'Bench_Press_Reps_missing','Broad_Jump_missing',
    'Agility_3cone_missing','Shuttle_missing','n_drills_missing',
    'School_Encoded','Position_Encoded',
    'year_draft_rate','pos_year_draft_rate','school_year_volume',
    'school_total_volume','pt_year_rate',
    'Sprint_40yd_vs_pos','Weight_vs_pos','Height_vs_pos',
    'Bench_Press_Reps_vs_pos','Vertical_Jump_vs_pos',
    'Broad_Jump_vs_pos','Agility_3cone_vs_pos','Shuttle_vs_pos',
]

target = X_train['Drafted']
SEEDS  = [42, 123, 777, 2024, 999]

xgb_oof  = np.zeros(len(X_train))
lgb_oof  = np.zeros(len(X_train))
xgb_pred = np.zeros(len(X_test))
lgb_pred = np.zeros(len(X_test))

for seed in SEEDS:
    cv_s = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    for ti, vi in cv_s.split(X_train[features], target):
        Xtr = X_train[features].iloc[ti]; Xval = X_train[features].iloc[vi]
        ytr = target.iloc[ti]

        xgb = XGBClassifier(n_estimators=1000, max_depth=5, learning_rate=0.01,
                            subsample=0.75, colsample_bytree=0.6, min_child_weight=5,
                            reg_alpha=0.1, reg_lambda=2.0, gamma=1.0,
                            random_state=seed, eval_metric='auc', verbosity=0)
        xgb.fit(Xtr, ytr)
        xgb_oof[vi]  += xgb.predict_proba(Xval)[:,1] / len(SEEDS)
        xgb_pred     += xgb.predict_proba(X_test[features])[:,1] / (10*len(SEEDS))

        lgb = LGBMClassifier(n_estimators=1000, learning_rate=0.01, max_depth=6,
                             num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                             min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
                             random_state=seed, verbose=-1)
        lgb.fit(Xtr, ytr)
        lgb_oof[vi]  += lgb.predict_proba(Xval)[:,1] / len(SEEDS)
        lgb_pred     += lgb.predict_proba(X_test[features])[:,1] / (10*len(SEEDS))

    print(f"Seed {seed} done")

blend_pred = 0.5*xgb_pred + 0.5*lgb_pred
blend_oof  = 0.5*xgb_oof  + 0.5*lgb_oof
print("XGB  OOF AUC:", round(roc_auc_score(target, xgb_oof),  5))
print("LGB  OOF AUC:", round(roc_auc_score(target, lgb_oof),  5))
print("Blend OOF AUC:", round(roc_auc_score(target, blend_oof), 5))

conf_mask = (blend_pred > 0.85) | (blend_pred < 0.15)
print(f"\nPseudo labels: {conf_mask.sum()} rows")

X_pseudo = X_test[conf_mask].copy()
X_pseudo['Drafted'] = (blend_pred[conf_mask] > 0.5).astype(int)
X_aug  = pd.concat([X_train, X_pseudo], ignore_index=True)
y_aug  = X_aug['Drafted']
y_strat = (y_aug > 0.5).astype(int)

xgb_oof2  = np.zeros(len(X_aug))
lgb_oof2  = np.zeros(len(X_aug))
xgb_pred2 = np.zeros(len(X_test))
lgb_pred2 = np.zeros(len(X_test))

for seed in SEEDS:
    cv_s = StratifiedKFold(5, shuffle=True, random_state=seed)
    for ti, vi in cv_s.split(X_aug[features], y_strat):
        Xtr = X_aug[features].iloc[ti]; Xval = X_aug[features].iloc[vi]
        ytr = y_aug.iloc[ti]

        xgb = XGBClassifier(n_estimators=1000, max_depth=5, learning_rate=0.01,
                            subsample=0.75, colsample_bytree=0.6, min_child_weight=5,
                            reg_alpha=0.1, reg_lambda=2.0, gamma=1.0,
                            random_state=seed, eval_metric='auc', verbosity=0)
        xgb.fit(Xtr, ytr)
        xgb_oof2[vi]  += xgb.predict_proba(Xval)[:,1] / len(SEEDS)
        xgb_pred2     += xgb.predict_proba(X_test[features])[:,1] / (5*len(SEEDS))

        lgb = LGBMClassifier(n_estimators=1000, learning_rate=0.01, max_depth=6,
                             num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                             min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
                             random_state=seed, verbose=-1)
        lgb.fit(Xtr, ytr)
        lgb_oof2[vi]  += lgb.predict_proba(Xval)[:,1] / len(SEEDS)
        lgb_pred2     += lgb.predict_proba(X_test[features])[:,1] / (5*len(SEEDS))

orig = np.zeros(len(X_aug), dtype=bool)
orig[:len(X_train)] = True

final_pred = 0.5*xgb_pred2 + 0.5*lgb_pred2
print("XGB  OOF AUC (orig):", round(roc_auc_score(y_aug[orig], xgb_oof2[orig]), 5))
print("LGB  OOF AUC (orig):", round(roc_auc_score(y_aug[orig], lgb_oof2[orig]),  5))
print("Blend OOF AUC:", round(roc_auc_score(y_aug[orig], 0.5*xgb_oof2[orig]+0.5*lgb_oof2[orig]), 5))

sub['Drafted'] = final_pred
sub.to_csv("/kaggle/working/submission_final.csv", index=False)
print("\nDone! shape:", sub.shape)
