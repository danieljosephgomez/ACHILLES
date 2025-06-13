# ⚔️🛡️ ACHILLES: Foundation Model for Multiplex imaging and Single-cell Spatial Multiomics for IO Therapeutics Mapping


This foundation model provides phenotypic information high-dimensional single-cell and spatial omics data, and is built to discover novel cell states and do in silico experimentation. It applies AI/ML, deep learning, and statistical geometrical analysis in a phenotypic and spatial information that requires preview and consortia biological datasets.



---



## 🚀 Overview



This repository features:


- **Data generation** for spatial single-cell transcriptomics  

- **Foundation model training** with PyTorch ([Autoencoder](training/train_model.py) / Transformer Encoder)  

- **Latent space embeddings** for clustering and embeddings 

- **Immuno-Therapeutics evaluation metrics** (e.g., ARI, NMI, F1, Silhouette)  

- **Spatial information** Spatial transcriptomics

- **Visualization of Spatial subcellular molecular localization and Phenotypic information** of latent space and clustering results using image science and dimension reduction

- **Interactive notebooks** for data exploration and visualization

- **LRI and Pathway enrichment analysis** for data exploration and visualization

- **Data download scripts** for 10X Genomics Xenium, Visium, Cell VDJ, and Vizgen MERSCOPE datasets


---
## 🔧 Setup



Create environment:

```bash

conda create -n ACHILLES python=3.11

conda activate ACHILLES 

```



Install dependencies:

```bash
pip install -r requirements.txt
```



---
---

🪓 10X Genomics Single-Cell and Spatial Data Download with xenium_dl.py
```bash
python xenium_dl.py --preset ['brain','lung','kidney']
```
---



## 🧪 Running the Project



1. **Generate and Map Synthetic, Spatial, and Single-cell Omics Data**

    ```bash
    ```



2. **Train Foundation Model**

    ```bash
    python ACHILLES/train_model.py
    ```



3. **Evaluate Latent Embeddings**

    ```bash
    python evaluation/evaluate_latent_space.py
    ```



Or open [`notebooks/`](notebooks/) to run interactively.



[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danjoegomez/ACHILLES/notebooks/01_visualize_hubmap_data.ipynb)



---



## 🧠 Key Concepts Demonstrated



| Skill                      | Implementation                             |

| -------------------------- | ------------------------------------------ |

| Python & ML Frameworks     | PyTorch models, training loop              |

| Applied Math & Statistics  | PCA, UMAP, clustering metrics              |

| Foundation Model Logic     | Latent space encoding & reuse              |

| Creative Problem-Solving   | Simulated ontologies + latent structure    |

| Scalability & Validation   | Train/test split, synthetic generalization |

| Collaboration & Modularity | Structured repo, clear API, doc-ready      |



---



## 📊 Example Outputs



- **spatial_ligand_map.png**: Spatial map of ligand expression across tissue  

- **spatial_receptor_map.png**: Spatial map of receptor expression across tissue  

- **interaction_matrix.csv**: Quantified ligand-receptor interaction matrix between cell types or regions  



---



## 📥 Data Download Scripts



- **10X Genomics Single-Cell and Spatial Data**  

    ```bash

    python download_10x.py --url https://cf.10xgenomics.com/samples/xenium/1.3.0/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs.zip --output data/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs.zip

    ```



- **Vizgen MERSCOPE Spatial Multiomics Data**  

    ```bash

    python download_merscope.py --gcs-path gs://vz-ffpe-showcase/HumanUterineCancerPatient2-ROCostain/cell_by_gene.csv --output data/merscope/cell_by_gene.csv

    python download_merscope.py --gcs-path gs://vz-ffpe-showcase/HumanUterineCancerPatient2-ROCostain/cell_metadata.csv --output data/merscope/cell_metadata.csv

    ```

---



## 🧬 Tissue/Region Classification 



Classify tissue or region using latent space embeddings and a random forest classifier:



```bash

python tissue_classification.py \

  --input_csv outputs/latent_with_umap_clusters.csv \

  --tissue_col Tissue \

  --output_report outputs/analysis/tissue_classification_report.csv

```



---



## 📜 License

MIT License


---



## 👤 Contact: Daniel J. Gomez  

[sfdanielgomez@gmail.com](mailto:sfdanielgomez@gmail.com) 
=======
👤 Author
Daniel J. Gomez
[sfdanielgomez@gmail.com | https://www.linkedin.com/in/danieljgomez1]
