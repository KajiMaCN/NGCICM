# NGCICM: A novel deep learning-based method for predicting circRNA-miRNA interactions

### Abstract

The circRNAs and miRNAs play an important role in the development of human diseases, and they can be widely used as biomarkers of diseases for disease diagnosis. In particular, circRNAs can act as sponge adsorbers for miRNAs and act together in certain diseases. However, the associations between the vast majority of circRNAs and diseases and between miRNAs and diseases remain unclear. Computational-based approaches are urgently needed to discover the unknown interactions between circRNAs and miRNAs. In this paper, we propose a novel deep learning algorithm based on Node2vec and Graph ATtention network (GAT), Conditional Random Field (CRF) layer and Inductive Matrix Completion (IMC) to predict circRNAs and miRNAs interactions (NGCICM). We construct a GAT-based encoder for deep feature learning by fusing the talking-heads attention mechanism and the CRF layer. The IMC-based decoder is also constructed to obtain interaction scores. The Area Under the receiver operating characteristic Curve (AUC) of the NGCICM method is 0.9697, 0.9932 and 0.9980, and the Area Under the Precision-Recall curve (AUPR) is 0.9671, 0.9935 and 0.9981, respectively, using 2-fold, 5-fold and 10-fold Cross-Validation (CV) as the benchmark. The experimental results confirm the effectiveness of the NGCICM algorithm in predicting the interactions between circRNAs and miRNAs.

### Contributions

- We propose a deep learning-based method to predict the interactions between circRNAs and miRNAs through the circRNAs-cancers-miRNAs linkage, and the proposed algorithm is a novel computational-based method to explore potential interactions between circRNAs and miRNAs. The associations drawn through the case studies can be provided to support the wet experiment.
- We construct an encoder and decoder based on the GAT algorithm, which incorporates the talking-heads attention mechanism and CRF layers as well as IMC. This approach better preserves the high-level feature information and reduces redundant information.
- We use 2-fold, 5-fold and 10-fold CV to demonstrate that the NGCICM method outperforms existing state-of-the-art algorithms on a wide range of performance evaluation metrics.

### Citation

If you found this paper or code helpful, please cite our paper:

```
@article{Ma2023,
     author = {Ma, Zhihao and Kuang, Zhufang and Deng, Lei},
     journal = {IEEE/ACM TRANSACTIONS ON COMPUTATIONAL BIOLOGY AND BIOINFORMATICS},
     title = {{NGCICM: A novel deep learning-based method for predicting circRNA-miRNA interactions}},
     year = {2023}
     }
```

### Others

**If you have any questions, please submit your issues.**



***other works***

Ma Z, Kuang Z, Deng L. CRPGCN: predicting circRNA-disease associations using graph convolutional network based on heterogeneous network[J]. *BMC bioinformatics*, 2021, 22(1): 1-23. 【[paper](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-04467-z)】【[code](https://github.com/KajiMaCN/CRPGCN)】 

