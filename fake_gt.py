import pandas as pd

read_df = lambda x: (
    pd.read_csv(x, names=['name','obj','frame','J','F'])
    .assign(JF = lambda r: (r.J + r.F) /2)
    .drop(columns=['J','F'])
    .set_index(['name','frame','obj'])
)
def to_frame(df):
    return df.groupby(['name','frame']).mean()

def score(x):
    scores = x.groupby(['name','obj']).mean()
    return scores.mean().item()
def print_score(s):
    print('{:4.2f}'.format(s*100), end='  ')

def ensem_opt(dfs):
    df = dfs[0].copy()
    for d in dfs:
        idx = d.JF > df.JF
        if any(idx):
            df.loc[idx] = d.loc[idx]
        sco = score(df)
        print_score(sco)
        # print(f'update {idx.sum()} / {idx.shape[0]}')
    return df

def ensem_frame(dfs):
    df = dfs[0].copy()
    for d in dfs:
        idx = d.groupby(['name','frame']).mean().JF > df.groupby(['name','frame']).mean().JF
        if  any(idx):
            df.loc[idx] = d.loc[idx]
        
        sco = score(df)
        print_score(sco)

        # print(f'update {idx.sum()} / {idx.shape[0]}')
    return df

def ensem_video(dfs):
    dfs = [d.reset_index(['frame','obj']) for d in dfs]
    df = dfs[0].copy()
    for d in dfs:
        idx = d.groupby(['name','obj']).mean().groupby('name').mean().JF > \
         df.groupby(['name','obj']).mean().groupby('name').mean().JF
        if any(idx):
            df.loc[idx] = d.loc[idx]
        sco = score(df.reset_index().set_index(['name','frame','obj']))
        print_score(sco)

        # print(f'update {idx.sum()} / {idx.shape[0]}')
        
    return df.reset_index().set_index(['name','frame','obj'])

def to_video(x):
    return (x.groupby(['name','obj'])
    .mean().groupby('name')
    .mean()
)
def to_obj(x):
    return (x.groupby(['name','obj'])
    # .mean().groupby('name')
    .mean()
)
