#%%
import pandas as pd
import numpy as np
from scipy.stats import binom
from scipy.stats import false_discovery_control
from multiprocessing import Pool
from functools import partial
#%%
#%%
filtered_count_folder = './../data/filtered_gene_bc_matrices/hg19/'
filtered_count_file = './../data/filtered_gene_bc_matrices/hg19/matrix.mtx'
filtered_gene_label = './../data/filtered_gene_bc_matrices/hg19/genes.tsv'
filtered_barcode_label = './../data/filtered_gene_bc_matrices/hg19/barcodes.tsv'
sctransform_file = './../data/R_sctransform/pbmc3k_sctransform.tsv'
#%%
sctransform_tbl = (pd.melt(pd.read_csv(sctransform_file,sep='\t').reset_index(),
        id_vars=['index'],
        var_name='name',
        value_name='sctransform')
 .rename(columns={'index':'name','name':'ID'})
)
# %%
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
data_tbl = (    count_tbl
    .merge(cell_cov_tbl)
    .merge(gene_read_rate_tbl)
)
with Pool(processes=10) as pool:
        # pool.map applies 'parallel_func' to every item in 'pairwise_combinations'
        df = pool.map(partial(get_cell_enrich,data_tbl=data_tbl), count_tbl.barcode_idx.drop_duplicates().to_list())

data_enrich_tbl = pd.concat(df)
#%%

sctransform_tbl = (sctransform_tbl
.merge(barcode_label_tbl
       .assign(alt_ID = lambda df: df.ID.str.slice(stop=-2))
       .assign(barcode_idx = lambda df: df.index +1 )
       .loc[:,['alt_ID','barcode_idx']]
       .rename(columns={'alt_ID':'ID'}),how='left')
.merge(gene_label_tbl.assign(gene_idx = lambda df: df.index+1).loc[:,['name','gene_idx']],how='left')
.merge(count_tbl,how='left')
.fillna(0)
)
# %%
binom_vs_sctransform_tbl = (sctransform_tbl
 .merge(data_enrich_tbl.loc[:,['gene_idx','barcode_idx','enrichment','cell_rate','cell_vs_bulk_OR']],how='left')
 .assign(enrichment = lambda df: np.where(df.read_count.lt(1),1,df.enrichment),
         cell_rate = lambda df: np.where(df.read_count.lt(1),0,df.cell_rate),
         cell_vs_bulk_OR = lambda df: np.where(df.read_count.lt(1),0,df.cell_vs_bulk_OR))
)
# %%
(
    binom_vs_sctransform_tbl
    .query('read_count > 0')
    .assign(rate_col = lambda df: np.log10(df.cell_rate))
    .sort_values('cell_rate')
    .plot.scatter(x='enrichment',y='sctransform',c='rate_col',xlabel='binomial test p-value',s=0.1)
)

# %%
(binom_vs_sctransform_tbl
.query('ID == "GGCACGTGTGAGAA"')
.query('read_count > 0')
.assign(sc_rank = lambda df: df.sctransform.rank(pct=True),
        binom_rank = lambda df: df.enrichment.rank(pct=True, ascending=False))
.plot
.scatter(x='sc_rank',y='binom_rank',s=1,xlabel='sctransform',ylabel='binomial test')
)

# %%
