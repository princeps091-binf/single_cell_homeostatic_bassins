#%%
import pandas as pd
import numpy as np
import hdbscan
import statsmodels.stats.rates as st
from scipy.stats import binom
from scipy.stats import false_discovery_control
import xlmhglite
import itertools
from multiprocessing import Pool
from functools import partial
#%%
#%%

filtered_count_file = './../data/filtered_gene_bc_matrices/hg19/matrix.mtx'
filtered_gene_label = './../data/filtered_gene_bc_matrices/hg19/genes.tsv'
filtered_barcode_label = './../data/filtered_gene_bc_matrices/hg19/barcodes.tsv'

#%%

gene_label_tbl = (pd.read_csv(filtered_gene_label,comment='%',sep='\t',header=None)
 .rename(columns={0:'ID',1:'name'})
)

barcode_label_tbl = (pd.read_csv(filtered_barcode_label,comment='%',sep='\t',header=None)
 .rename(columns={0:'ID'})
)

count_tbl = (pd.read_csv(filtered_count_file,comment='%',sep=' ',header=None,skiprows=3)
 .rename(columns={0:'gene_idx',1:'barcode_idx',2:'read_count'})
)
#%%
cell_cov_tbl = (count_tbl
 .groupby('barcode_idx')
 .agg(tot_count= ('read_count','sum'))
 .reset_index())
#%%
gene_stat_df = (
    count_tbl
    .merge(cell_cov_tbl)
    .assign(rel_count = lambda df: df.read_count / df.tot_count)
    .groupby('gene_idx')
    .agg(ncell = ('barcode_idx','nunique'),
        avg_rel = ('rel_count','mean'),
        std_rel = ('rel_count','std'),
        )
    .reset_index()
    .assign(CV2 = lambda df: (df.std_rel/df.avg_rel)**2)
)
gene_stat_df = (
    count_tbl
    .merge(cell_cov_tbl)
    .assign(rel_count = lambda df: df.read_count / df.tot_count)
    .groupby('gene_idx')
    .agg(ncell = ('barcode_idx','nunique'),
        avg_rel = ('rel_count','mean'),
        std_rel = ('rel_count','std'),
        )
    .reset_index()
    .assign(CV2 = lambda df: (df.std_rel/df.avg_rel)**2)
)
#%%
gene_read_rate_tbl =(
    count_tbl
    .groupby('gene_idx')
    .agg(gene_read_count = ('read_count','sum'),
         ncell = ('barcode_idx','nunique'))
)
gene_read_rate_tbl = (gene_read_rate_tbl
                      .assign(gene_read_rate = lambda df: df.gene_read_count / count_tbl.read_count.sum())
                      .assign(detection_limit_sample = lambda df: 1/df.gene_read_rate)
                      .reset_index()
                      )
#%%
# %%

data_tbl = (    count_tbl
    .merge(cell_cov_tbl)
    .merge(gene_read_rate_tbl)
)
#%%
def get_cell_enrich(cell_id,data_tbl):
    return (data_tbl
        .query('barcode_idx == @cell_id')
        .assign(cell_rate = lambda df: df.read_count/df.tot_count)
        .assign(cell_vs_bulk_OR= lambda df: df.cell_rate/df.gene_read_rate)
        .assign(enrichment = lambda df: df.apply(lambda row: binom.sf(row.read_count, row.tot_count, row.gene_read_rate),axis=1),
                fdr = lambda df: false_discovery_control(df.enrichment))
        .assign(pscore = lambda df: -np.log10(df.enrichment),
                prank = lambda df: df.pscore.rank(pct=True,ascending=False)
                )

        )  
#%%
#%%
cell_tbl = get_cell_enrich(230,data_tbl)
score_quantile = cell_tbl.pscore.quantile([0.25,0.75])
out_up_score_thresh = score_quantile.iloc[1] + 1.5*(score_quantile.iloc[1] - score_quantile.iloc[0])
print(cell_tbl.query('fdr < 1/@cell_tbl.shape[0]').shape[0]/cell_tbl.shape[0])
print(cell_tbl.query('pscore > @out_up_score_thresh').shape[0]/cell_tbl.shape[0])

#%%
with Pool(processes=10) as pool:
        # pool.map applies 'parallel_func' to every item in 'pairwise_combinations'
        df = pool.map(partial(get_cell_enrich,data_tbl=data_tbl), count_tbl.barcode_idx.drop_duplicates().to_list())
#%%
(pd.concat(df)
 .assign(credible=lambda df:df.fdr.lt(0.05))
 .groupby('gene_idx')
 .agg(mfdr = ('fdr','min'),
      m_count=('cell_rate','mean'),
      max_count = ('cell_rate','max'),
      std_count = ('cell_rate','std'),
      ncell=('barcode_idx','nunique'),
      prop_cred=('credible','mean'))
 .reset_index()
 .query('ncell > 5')
 .sort_values('m_count',ascending=False)
 .assign(lmfdr = lambda df: df.mfdr.rank(pct=True),
         lmcount = lambda df: df.m_count.rank(pct=True,ascending=False),
         lstd = lambda df: df.std_count.rank(pct=True,ascending=False))
 .plot.scatter(x='ncell',y='lmfdr',c='lmcount',s=10,logx=True)
)
#%%
tmp_gene_idx = gene_label_tbl.query("name == 'GZMB'").index.to_list()[0] + 1

pd.concat(df).query("gene_idx == @tmp_gene_idx").plot.scatter(x='cell_rate',y='fdr',logy=True,s=0.5)
