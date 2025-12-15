#%%
import pandas as pd
import numpy as np
from scipy.stats import binom
from scipy.stats import false_discovery_control
import xlmhglite
import itertools
from multiprocessing import Pool
from functools import partial
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, optimal_leaf_ordering
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances,manhattan_distances,pairwise_distances,cosine_similarity
from sklearn.manifold import MDS
import umap 
import scanpy as sc
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
with Pool(processes=10) as pool:
        # pool.map applies 'parallel_func' to every item in 'pairwise_combinations'
        df = pool.map(partial(get_cell_enrich,data_tbl=data_tbl), count_tbl.barcode_idx.drop_duplicates().to_list())

data_enrich_tbl = pd.concat(df)

# %%
(data_enrich_tbl
 .assign(enriched = lambda df: np.where(df.cell_vs_bulk_OR.gt(1),'enriched','depleted'))
 .groupby('enriched')
 .enrichment
 .plot.kde(legend=True))
#%%
gene_of_interest_idx = gene_label_tbl.query("name == 'LYZ'").index.to_list()[0] + 1

cells_of_interest_list = (count_tbl
 .query('gene_idx == @gene_of_interest_idx')
 .loc[:,['barcode_idx']]
 .drop_duplicates()
 .merge(cell_cov_tbl)
 .sort_values('tot_count')
 .barcode_idx.drop_duplicates().to_list()
 )
print(len(cells_of_interest_list))

(data_enrich_tbl

 .query('gene_idx == @gene_of_interest_idx')
 .assign(OR_rank = lambda df: df.cell_vs_bulk_OR.rank(pct=True))
 .plot.scatter(x='pscore',y='cell_vs_bulk_OR',c='tot_count',cmap='viridis',logy=True,logx=True)
)
# %%

(data_enrich_tbl
.query('barcode_idx in @cells_of_interest_list')
)


dummy_enrichment = (- np.log10(data_enrich_tbl
                              .query('barcode_idx in @cells_of_interest_list')
                              .query('enrichment > 0').enrichment.min())) + 1
enrichment_dense_matrix= (pd.pivot_table(data_enrich_tbl
                                         .query('barcode_idx in @cells_of_interest_list')
                                         .assign(corrected_enrichment= lambda df: -np.log10(df.enrichment))
                                         .assign(corrected_enrichment= lambda df: np.where(np.isinf(df.corrected_enrichment.to_numpy()),dummy_enrichment,df.corrected_enrichment)),
                                         index='barcode_idx',columns='gene_idx',values='corrected_enrichment'))
enrichment_dense_matrix = enrichment_dense_matrix.fillna(0)
cosine_sim_mat = cosine_similarity(enrichment_dense_matrix)

#%%
fig, ax = plt.subplots(figsize=(7, 6))

# 2. Display the matrix data as an image
# 'cmap' sets the color scheme (e.g., 'viridis', 'plasma', 'coolwarm', 'Greys')
# 'interpolation' determines how pixels are drawn (nearest is usually best for matrices)
im = ax.imshow(cosine_sim_mat, cmap='plasma_r', interpolation='nearest')
#%%
cosine_dist_mat = 1 - cosine_sim_mat
np.fill_diagonal(cosine_dist_mat, 0.0)

#%%
#cosine_dist_mat = np.round(euclidean_distances(enrichment_dense_matrix),decimals=5)
#%%
cosine_condensed_dist_matrix = squareform(cosine_dist_mat)
linked = linkage(cosine_condensed_dist_matrix, method='ward') # You can choose other methods like 'complete', 'average', 'single'
# Extract the leaf order from the linkage matrix
# The optimal_leaf_ordering function reorders the leaves for better visualization
cosine_ordered_linked = optimal_leaf_ordering(linked, cosine_condensed_dist_matrix)
cosine_leaf_order = dendrogram(cosine_ordered_linked, no_plot=True)['leaves']
cosine_reordered_matrix = cosine_dist_mat[cosine_leaf_order, :]
cosine_reordered_matrix = cosine_reordered_matrix[:, cosine_leaf_order]
#%%
fig, ax = plt.subplots(figsize=(7, 6))

# 2. Display the matrix data as an image
# 'cmap' sets the color scheme (e.g., 'viridis', 'plasma', 'coolwarm', 'Greys')
# 'interpolation' determines how pixels are drawn (nearest is usually best for matrices)
im = ax.imshow(cosine_reordered_matrix, cmap='plasma', interpolation='nearest')
#%%
reducer = umap.UMAP(metric='precomputed',n_neighbors= 45, random_state=42)
embedding = reducer.fit_transform(cosine_dist_mat)
#%%
gene_of_interest_idx = gene_label_tbl.query("name == 'LYZ'").index.to_list()[0] + 1

bio_lvl_convert_tbl = (data_enrich_tbl.query('gene_idx == @gene_of_interest_idx')
.loc[:,['enrichment']]
.assign(corrected_enrichment= lambda df: -np.log10(df.enrichment))
.assign(corrected_enrichment= lambda df: np.where(np.isinf(df.corrected_enrichment.to_numpy()),dummy_enrichment,df.corrected_enrichment))
.assign(bior = lambda df: df.corrected_enrichment.rank(pct=True))
.loc[:,['corrected_enrichment','bior']].reset_index(drop=True)
)
plt_tbl = (pd.DataFrame({'x':embedding[:, 0],
                        'y':embedding[:, 1],
                        'biomarker_lvl':enrichment_dense_matrix.loc[:,gene_of_interest_idx].to_numpy()})
                        .assign(bio_s = lambda df: np.where(df.biomarker_lvl.gt(0),10,0.1),
                                bio_c = lambda df: df.biomarker_lvl)
                        .sort_values('biomarker_lvl')
                        .merge(bio_lvl_convert_tbl,left_on='biomarker_lvl',right_on='corrected_enrichment',how='outer').fillna(0))

plt.figure(figsize=(8, 6))
plt.scatter(plt_tbl.x.to_numpy(), plt_tbl.y.to_numpy(),c=plt_tbl.bior.to_numpy(),s=plt_tbl.bio_s.to_numpy())
plt.title('UMAP Embedding with Precomputed Distance Matrix')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.grid(True)
plt.show()

# %%
