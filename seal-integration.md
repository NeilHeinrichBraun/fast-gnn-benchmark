# Plan d'action — Intégration de SEAL dans fast-gnn-benchmark

SEAL fonctionne différemment des modèles GAE actuels — il opère sur des **sous-graphes enclosants** (un graphe par paire de nœuds candidate) au lieu du graphe global, avec un node labeling DRNL et un pooling de graphe (SortAggregation). Voici le plan d'action structuré.

---

## 1. Schémas (`src/fast_gnn_benchmark/schemas/`)

### `dataset_models.py`
- Ajouter `DataLoaderType.SEAL_LOADER = "seal_loader"`.
- Créer `SEALLoaderParameters` (champs : `num_hops`, `batch_size`, `node_label`="drnl" (DRNL / zero / hop / drnl+), `max_nodes_per_hop` (sous-échantillonnage), `use_features` bool, `cache_dir`, éventuellement `num_workers`/`pin_memory`).
- L'ajouter à l'union `DataLoaderParametersChoices`.

### `model.py`
- Ajouter `ArchitectureType.DGCNN = "dgcnn"` (le backbone par défaut du papier SEAL).
- Créer `DGCNNParameters(ArchitectureParameters)` : `num_layers`, `k` (sort pooling, percentile ou int), `gnn_type` (gcn/sage/gat — pour réutiliser tes backbones existants en interne), `conv1d_channels`, `dense_dim`, `dropout`.
- L'ajouter à `ArchitectureParametersChoices`.
- Créer `LinkPredictorType.GRAPH_CLASSIFIER` (head qui prend un embedding de graphe pooled et sort un logit) — alternative à `HADAMARD_MLP`/`COSINE_SIMILARITY` qui sont conçus pour deux embeddings de nœuds.
- Optionnel : ajouter un flag `task_subtype: Literal["whole_graph", "subgraph"]` sur `LinkPredictionModelParameters` pour distinguer le mode GAE (graphe entier) du mode SEAL (sous-graphes) — c'est ce flag qui pilotera le trainer.

---

## 2. Dataset / dataloader (`src/fast_gnn_benchmark/data/`)

### Nouveau `dataset/seal.py`
- Encapsuler `SEALDataset` du fichier extrait dans une classe qui prend en entrée un de tes datasets existants (Planetoid, OGBL, Amazon…) et produit un `InMemoryDataset` de sous-graphes.
- Externaliser :
  - `num_hops` paramétrable.
  - Stratégie de labeling pluggable (DRNL, zero, hop, distance encoding).
  - Option `use_features` pour concaténer ou non `data.x` original aux features one-hot du labeling.
  - Cache par splits (`SEAL_<dataset>_<num_hops>_<label>_{train,val,test}.pt`) basé sur les splits OGBL existants au lieu de `RandomLinkSplit` (important pour rester comparable avec tes runs `gae_*_ogbcollab.yml` / `gae_*_ppa.yml`).
- Construire le dataset à partir des `dataset.split["train"|"valid"|"test"]` de `FixLinkPropPredDataset` quand on est sur OGBL (pas de `RandomLinkSplit`).

### Nouveau `data/seal_dataloader.py`
- Wrapper minimal autour du `DataLoader` de `torch_geometric.loader` (qui sait batcher des `Data` en sous-graphes), avec la même interface que `LinkLoader` (`__iter__`, `__len__`, `split_type`) pour que `trainer.py` n'ait rien à changer en aval.
- Yield des batchs où chaque `Data` est un sous-graphe avec `x`, `edge_index`, `batch`, `y` (label 0/1 du lien target). C'est différent du format `target_edges` actuel — c'est pour ça qu'il faut un loader dédié.

### `data_models.py`
- Ajouter le case `SEAL_LOADER` dans `get_data_loader` : instancier le `SEALDataset` à partir du dataset déjà chargé, puis envelopper dans le `SEALDataLoader`.
- Comme le pré-processing SEAL est lourd, le faire **une seule fois** sur le dataset partagé (avec cache disque) — pas trois fois pour train/val/test.

---

## 3. Backbone (`src/fast_gnn_benchmark/models/backbones/`)

### Nouveau `dgcnn.py`
- Implémenter `DGCNN` : pile de convs (utiliser `load_gnn` avec tes `GNNParameters` existants pour le GNN interne → c'est ainsi qu'on rend l'archi GNN modulable) suivie de SortAggregation + Conv1d + MLP.
- Le forward prend `(x, edge_index, batch)` et renvoie un logit par graphe.
- Idéalement séparer en deux modules : un `SubgraphEncoder` (GNN stack + pooling → embedding par graphe) et la tête classification, pour pouvoir swap d'autres pools (mean/add/Set2Set) plus tard.

### `backbones/__init__.py`
- Ajouter le case `ArchitectureType.DGCNN` dans `load_backbone`.
- **Important** : aujourd'hui `load_backbone` retourne un module qui mappe `(x, edge_index) → x_node`. DGCNN retourne `(x, edge_index, batch) → logit_par_graphe`. Soit tu acceptes la divergence de signature (et le modèle SEAL appelle directement le backbone sans head séparée), soit tu introduis une convention : tout backbone produit un embedding (par nœud ou par graphe) et la head s'adapte. Je conseille la première option (plus simple, plus fidèle au papier).

---

## 4. Modèle (`src/fast_gnn_benchmark/models/`)

### Nouveau `seal.py` (ou étendre `link_prediction.py`)
- Créer `SEALLinkPredictionModel(BaseGNN[LinkPredictionModelParameters])` :
  - `load_model()` construit le backbone DGCNN (ou n'importe quel backbone de graph-level retourné par `load_backbone`).
  - `training_step` / `validation_step` / `test_step` font `pred = self.model(batch.x, batch.edge_index, batch.batch)` puis BCE contre `batch.y`. Plus de `target_edges` ni de classifier `Hadamard_MLPPredictor`.
- Si tu introduis `task_subtype`, le routing se fait dans `trainer.py:get_model` ; sinon, garde un task_type séparé `"link_prediction_seal"`.

### `link_prediction_heads.py`
- Ajouter (si tu veux rester proche du pattern existant) un `GraphClassifierHead` qui wrap le MLP final de DGCNN. C'est cosmétique — utile seulement si tu sors le pooling+MLP du backbone.

---

## 5. Trainer (`src/fast_gnn_benchmark/trainer.py`)

- Dans `get_model`, ajouter le routage vers `SEALLinkPredictionModel` quand `task_type == "link_prediction"` + `task_subtype == "subgraph"` (ou avec un nouveau task_type, au choix).
- `check_test_batch` : vérifier qu'il marche encore — les batchs SEAL n'ont pas de `target_edges` mais ont `batch`, donc l'appel `model.test_step` doit fonctionner naturellement avec le nouveau model class.

---

## 6. Metrics (`src/fast_gnn_benchmark/metrics/base_metrics.py`)

- `HitRate@K` et `MRR` sont actuellement calculés par batch sur des `target_edges` regroupés (positives + negatives ensemble). En SEAL, chaque sous-graphe donne un logit indépendant — il faut vérifier que ces metrics fonctionnent quand on les nourrit batch-par-batch avec `(pred, y)` flat. Probablement OK pour `BinaryAccuracy` et `ROC_AUC` ; à adapter pour Hit@K/MRR (regrouper toutes les preds de l'epoch via `update`/`compute` standard de torchmetrics, ce qui semble déjà être le cas).

---

## 7. Configs (`configs/link_prediction/`)

- Créer `seal_dgcnn_cora.yml` (sanity check rapide, équivalent du script PyG).
- Créer `seal_dgcnn_ogbcollab.yml` et `seal_dgcnn_ppa.yml` pour comparer avec tes `gae_*_ogbcollab.yml` / `gae_*_ppa.yml` existants.
- Créer des variantes avec d'autres GNN internes (`seal_sage_*.yml`, `seal_gcn_*.yml`, voire `seal_sgformer_*.yml`) pour exploiter ton catalogue d'architectures.

---

## 8. Ordre d'implémentation recommandé

1. Schémas (enum + pydantic models) — squelette qui guide le reste.
2. `SEALDataset` + `SEALDataLoader` testés en isolation sur Cora.
3. `DGCNN` backbone + intégration dans `load_backbone`.
4. `SEALLinkPredictionModel` + routage dans `trainer.py`.
5. Config Cora → vérifier qu'on reproduit ~les chiffres du papier.
6. Configs OGBL + variantes d'architectures.

---

## 9. Points d'attention

- **Coût mémoire/temps** : l'extraction des sous-graphes sur OGBL-PPA ou OGBL-COLLAB est très coûteuse (millions d'edges candidats × k-hop). Prévois un cache disque obligatoire et probablement un sous-échantillonnage des négatifs en train (`max_nodes_per_hop` + limiter le nombre de négatifs par epoch).
- **DRNL coûteux** : le `shortest_path` scipy est CPU-only et lent. Pour les gros graphes, prévoir une version sparse-batchée ou approximée.
- **Splits OGBL** : ne PAS utiliser `RandomLinkSplit` comme dans le script PyG — utiliser les splits officiels via `dataset.split` (cohérence avec tes runs GAE existants).
- **Features** : le script PyG drop `data.x` et n'utilise que le one-hot du labeling. Pour OGBL-PPA (features 58-dim) ou ton embedder learnable, il faut concaténer `data.x[sub_nodes]` + one-hot — c'est ce qui rend SEAL réellement comparable à tes baselines GAE.
