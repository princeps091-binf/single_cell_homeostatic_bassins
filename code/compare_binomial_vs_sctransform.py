#%%
import pandas as pd
import numpy as np
import hdbscan
import statsmodels.stats.rates as st
from scipy.stats import binom
from scipy.stats import false_discovery_control
import scanpy as sc
import anndata as ad
from pysctransform import SCTransform
#%%
filtered_count_folder = './../data/filtered_gene_bc_matrices/hg19/'
filtered_count_file = './../data/filtered_gene_bc_matrices/hg19/matrix.mtx'
filtered_gene_label = './../data/filtered_gene_bc_matrices/hg19/genes.tsv'
filtered_barcode_label = './../data/filtered_gene_bc_matrices/hg19/barcodes.tsv'
#%%
adata = sc.read_10x_mtx(filtered_count_folder)
# mitochondrial genes, "MT-" for human, "Mt-" for mouse
adata.var["mt"] = adata.var_names.str.startswith("MT-")
# ribosomal genes
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
# hemoglobin genes
adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")
sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
)
#%%
residuals = SCTransform(adata, var_features_n=3000)
#%%
adata.obsm["pearson_residuals"] = residuals
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
expected_observations_tbl = (count_tbl
 .gene_idx.drop_duplicates().reset_index().loc[:,['gene_idx']]
.merge(count_tbl
       .barcode_idx.drop_duplicates().reset_index().loc[:,['barcode_idx']]
       ,how='cross')
.merge(cell_cov_tbl)
.merge(gene_read_rate_tbl.loc[:,['gene_idx','detection_limit_sample']])
.query('tot_count >= detection_limit_sample')
)

#%%
unobserved_but_expected_tbl = (    count_tbl
    .merge(expected_observations_tbl.assign(status='expected').loc[:,['gene_idx','barcode_idx','status']],
           how='outer')
    .query('status == "expected"')
    .fillna(0)
    .query('read_count < 1')
    .drop('status',axis=1)

)

# %%

cell_of_interest = 'TTTCGAACTCTCAT-1'

barcode_idx = barcode_label_tbl.query('ID == @cell_of_interest').index.to_list()[0] + 1

cell_tbl = (
    count_tbl
    .merge(cell_cov_tbl)
    .merge(gene_read_rate_tbl)
    .assign(cell_rate = lambda df: df.read_count/df.tot_count)
    .assign(cell_vs_bulk_OR= lambda df: df.cell_rate/df.gene_read_rate)
    .query('barcode_idx == @barcode_idx')
    .assign(enrichment = lambda df: df.apply(lambda row: binom.sf(row.read_count, row.tot_count, row.gene_read_rate),axis=1),
            fdr = lambda df: false_discovery_control(df.enrichment))
    .assign(pscore = lambda df: -np.log10(df.enrichment),
            prank = lambda df: df.pscore.rank(pct=True,ascending=False)
            )

)

#%%
cell_tbl = (
    cell_tbl
    .merge(gene_label_tbl
           .assign(gene_idx = lambda df: df.ID.index.to_numpy() + 1)
           .loc[:,['gene_idx','name']])
)
#%%
(cell_tbl.merge(
    adata.obsm['pearson_residuals']
    .loc[cell_of_interest]
    .reset_index()
    .rename(columns={'index':'name',cell_of_interest:'sctransform'})
    ,how='left'
)
.dropna()
.assign(scrank = lambda df: df.sctransform.rank(pct=True,ascending=False))
.plot.scatter(x='prank',y='scrank',s=0.5,
              xlabel='binomial test significance',
              ylabel='negative binomial regression residual')
)
# %%
