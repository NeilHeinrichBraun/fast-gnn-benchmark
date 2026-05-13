# Implémentation C++ SEAL — Plan single-threaded

## 1. Périmètre

**Remplacé par C++** : la boucle d'extraction dans `extract_enclosing_subgraphs` — k-hop neighborhood, retrait du lien cible, BFS distances, calcul DRNL.

**Reste Python** : construction des `Data`, `F.one_hot` + concat features, `self.save()`, orchestration dans `process()`.

---

## 2. Structure des fichiers

```
src/fast_gnn_benchmark/data/dataset/
├── seal.py                  ← 2 points de modification (voir §6)
└── seal_cpp/
    ├── __init__.py          ← JIT loader + fallback
    ├── seal_ops.cpp         ← implémentation principale + pybind11
    ├── csr.hpp              ← structure CSR
    ├── bfs.hpp              ← primitives BFS
    └── drnl.hpp             ← formule DRNL
```

---

## 3. Composants

### 3.1 `csr.hpp` — Représentation du graphe

**But** : convertir l'`edge_index` PyTorch (COO `[2, E]`) en CSR C++ pour accès O(1) aux voisins d'un nœud.

**Structure de données** :
```
struct CSRGraph {
    vector<int> row_ptr   // taille num_nodes + 1
    vector<int> col_idx   // taille num_edges
    int num_nodes
    int num_edges
}
```

**Algorithme de construction** :
1. Première passe sur `edge_index[0]` : compter le degré de chaque nœud source → vecteur `degree[num_nodes]`.
2. Prefix sum → `row_ptr` : `row_ptr[0] = 0`, `row_ptr[i+1] = row_ptr[i] + degree[i]`.
3. Deuxième passe : remplir `col_idx` en maintenant un curseur par ligne.

**Accès aux voisins d'un nœud `u`** : itérer `col_idx[row_ptr[u] .. row_ptr[u+1]]`.

> ⚠️ OGBL-COLLAB stocke chaque arête dans les deux sens dans `edge_index` — ne pas dupliquer, tout traiter directement.

---

### 3.2 `bfs.hpp` — Primitives BFS

Deux fonctions indépendantes.

#### A. `bfs_k_hop`

**But** : trouver tous les nœuds à distance ≤ k depuis un ensemble de nœuds seeds (ici `{src, dst}`).

**Entrée** : `CSRGraph`, `seeds: vector<int>`, `k: int`

**Sortie** : `sub_nodes: vector<int>` (indices globaux), `global_to_local: unordered_map<int, int>`

**Algorithme** :
- BFS standard avec profondeur. File : `deque<pair<int, int>>` (nœud, profondeur).
- Initialiser la file avec chaque seed à profondeur 0.
- Si profondeur == k : marquer le nœud mais ne pas explorer ses voisins.
- `sub_nodes` = nœuds découverts dans l'ordre de visite.
- `global_to_local[global_id] = index dans sub_nodes`.

#### B. `bfs_distances`

**But** : calculer les distances BFS depuis `source`, en traitant `excluded` comme inexistant.

**Entrée** : `local_adj: vector<vector<int>>`, `source_local: int`, `excluded_local: int`, `n: int`

**Sortie** : `distances: vector<int>` de taille n

**Algorithme** :
- BFS standard depuis `source_local`.
- Règle unique : quand `excluded_local` apparaît dans une liste de voisins, le sauter — ne jamais l'empiler ni le visiter.
- Nœuds non atteints → `distances[i] = INT_MAX`.

> **Pourquoi pas de retrait physique** : scipy retire le nœud de la matrice et réindexe. En BFS, sauter le nœud exclu est strictement équivalent mais sans reconstruction ni décalage d'indices.

---

### 3.3 `seal_ops.cpp` — Construction de la sous-adjacence locale

Logique interne à `seal_ops.cpp`, pas dans un header séparé.

**Entrée** : `sub_nodes`, `global_to_local`, `CSRGraph`, `src_local: int`, `dst_local: int`

**Sorties** :
- `local_adj: vector<vector<int>>` — adjacence locale avec lien cible retiré (utilisée pour les BFS DRNL)
- `sub_edge_index: at::Tensor [2, E_local]` — à retourner en Python

**Algorithme** :
1. Pour chaque nœud `u` dans `sub_nodes` (indice local `u_local`) :
   - Itérer ses voisins dans la CSR globale.
   - Pour chaque voisin `v` : si `v` est dans `global_to_local` → `v_local = global_to_local[v]`.
   - Si `(u_local, v_local)` ≠ `(src_local, dst_local)` et ≠ `(dst_local, src_local)` : ajouter `v_local` à `local_adj[u_local]`.
2. Construire `sub_edge_index` depuis `local_adj` : deux vecteurs `rows`, `cols`, convertis en `torch::Tensor`.

---

### 3.4 `drnl.hpp` — Formule DRNL

**Entrée** : `dist2src: vector<int>`, `dist2dst: vector<int>`, `src_local: int`, `dst_local: int`

**Sortie** : `z: at::Tensor` de type `torch::kLong`, taille n

**Algorithme** :
```
for i in 0..n:
    if dist2src[i] == INT_MAX OR dist2dst[i] == INT_MAX:
        z[i] = 0
    else:
        d      = dist2src[i] + dist2dst[i]
        d2     = d / 2
        d_mod  = d % 2
        z[i]   = 1 + min(dist2src[i], dist2dst[i]) + d2 * (d2 + d_mod - 1)

z[src_local] = 1   ← override systématique
z[dst_local] = 1   ← override systématique
```

**Edge cases** :
- `d2 * (d2 + d_mod - 1)` vaut 0 quand d=0 ou d=1 → correct.
- Vérifier `INT_MAX` avant le calcul de `d` pour éviter l'overflow.

---

### 3.5 `seal_ops.cpp` — Fonction principale et enregistrement pybind11

**Deux fonctions exposées à Python** :

#### `build_csr(edge_index: Tensor, num_nodes: int) -> tuple[Tensor, Tensor]`
- Construit la CSR et retourne `(row_ptr, col_idx)` comme tensors `torch::kInt32` ou `torch::kInt64`.
- Appelé une fois dans `process()`, résultats stockés dans `self._row_ptr`, `self._col_idx`.

#### `batch_extract(row_ptr, col_idx, src_batch, dst_batch, num_hops, num_nodes) -> tuple[list[Tensor], list[Tensor], list[Tensor]]`

**Corps single-threaded** :
```
Construire CSRGraph depuis row_ptr, col_idx (accès raw, pas de copie)
Initialiser 3 vecteurs de résultats de taille N

for i in 0..N:
    src = src_batch[i],  dst = dst_batch[i]

    sub_nodes, g2l = bfs_k_hop(csr, {src, dst}, num_hops)
    src_local = g2l[src],  dst_local = g2l[dst]

    local_adj, sub_edge_index = build_local_adj(sub_nodes, g2l, csr, src_local, dst_local)

    dist2src = bfs_distances(local_adj, src_local, dst_local, n)
    dist2dst = bfs_distances(local_adj, dst_local, src_local, n)

    z            = compute_drnl(dist2src, dist2dst, src_local, dst_local)
    sub_nodes_t  = tensor(sub_nodes, kLong)

    résultats[i] = (sub_edge_index, z, sub_nodes_t)

return (sub_edge_indices, z_labels, sub_nodes_list)
```

**Enregistrement pybind11** :
```
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batch_extract", &batch_extract)
    m.def("build_csr",     &build_csr_from_edge_index)
}
```

---

## 4. Build

**Approche** : JIT via `torch.utils.cpp_extension.load()` dans `seal_cpp/__init__.py`.

**Flags de compilation** : `-O3 -std=c++17`

**Pas d'OpenMP** : pas de `-fopenmp`, pas de dépendance à `libomp` → clang Apple Silicon standard.

**`seal_cpp/__init__.py`** :
```
Tenter load(name='seal_cpp', sources=[seal_ops.cpp], extra_compile_args=[...])
   Succès  → AVAILABLE = True, exposer batch_extract, build_csr
   Échec   → AVAILABLE = False, logger un warning, tout à None
```

Le module se compile au premier import, le binaire est caché dans `~/.cache/torch_extensions/seal_cpp/`. Recompilation automatique si le `.cpp` est modifié.

---

## 5. Schéma : `use_cpp_extension` dans `SEALLoaderParameters`

Ajouter `use_cpp_extension: bool = True` dans `SEALLoaderParameters` (`dataset_models.py`). Permet de désactiver le C++ dans le YAML pour comparer ou débugger sans recompiler :
```yaml
use_cpp_extension: false
```

---

## 6. Intégration dans `seal.py`

**Deux points de modification seulement.**

### Point 1 — `process()`

Avant les 6 appels à `extract_enclosing_subgraphs`, si `self.use_cpp` :
```
self._row_ptr, self._col_idx = seal_cpp.build_csr(full_edge_index, num_nodes)
```

### Point 2 — `extract_enclosing_subgraphs`

Deux chemins selon `self.use_cpp` :

**Chemin C++** :
1. Préparer `src_batch`, `dst_batch` depuis `edge_label_index` (deux vecteurs de taille N).
2. Appel unique : `sub_edges, z_labels, sub_nodes_list = seal_cpp.batch_extract(self._row_ptr, self._col_idx, src_batch, dst_batch, self.num_hops, num_nodes)`.
3. Boucle Python légère pour construire les `Data` :
   ```
   for i in range(N):
       x_sub = source_x[sub_nodes_list[i]] if self.use_features else None
       data_list.append(Data(x=x_sub, z=z_labels[i], edge_index=sub_edges[i], y=y))
   ```

**Chemin Python (inchangé)** : boucle `tqdm` + scipy existante.

---

## 7. Validation

### 7.1 Correctness

Sur 100 paires `(src, dst)` tirées aléatoirement depuis OGBL-COLLAB :
- Calculer `z_python` via `drnl_node_labeling` Python/scipy actuel.
- Calculer `z_cpp` via `seal_cpp.batch_extract` sur ces mêmes paires.
- Assertion : `torch.equal(z_python, z_cpp)` pour chaque paire.

Les résultats doivent être **bit-for-bit identiques** — DRNL est entièrement entier et déterministe, aucune tolérance flottante.

### 7.2 Performance

Mesurer sur 10 000 paires consécutives :
- Temps Python + scipy
- Temps C++ single-thread

Afficher le speedup. Cible réaliste : **15-30×**. Si inférieur à 5×, investiguer un overhead d'appel Python inattendu.

### 7.3 End-to-end

Générer le cache avec Python, générer avec C++ sur le même graphe, vérifier que les fichiers `.pt` produits ont des contenus identiques.

---

## 8. Ordre d'implémentation

| Étape | Fichier | Validation intermédiaire |
|---|---|---|
| 1 | `csr.hpp` | `row_ptr[-1] == num_edges`, lookup voisins manuels |
| 2 | `bfs.hpp` A — k-hop | Path graph 0-1-2-3, k=1, seed={1} → {0,1,2} |
| 3 | `bfs.hpp` B — distances | Path graph, BFS depuis 0 excluant 2 → [0, 1, INF, INF] |
| 4 | `drnl.hpp` | 3 paires à la main vs valeurs Python |
| 5 | Sous-adjacence dans `seal_ops.cpp` | Vérifier retrait lien cible |
| 6 | Assemblage + pybind11 | Import Python sans erreur |
| 7 | `seal_cpp/__init__.py` | JIT compile au premier import |
| 8 | Intégration `seal.py` | Test correctness §7.1 |
| 9 | Benchmark | §7.2 |
| 10 | End-to-end | §7.3 |

---

## 9. Extension future — OpenMP

Une fois la correctness validée en single-thread, ajouter le parallélisme est minimal :
- Remplacer la boucle `for i in 0..N` par `#pragma omp parallel for`.
- Sur macOS : installer `libomp` (`brew install libomp`) + flags `-fopenmp`.
- Sur Linux/cluster : `gcc -fopenmp`, zéro configuration.
- La logique algorithmique ne change pas d'une ligne.

Gain attendu avec OpenMP (8 cœurs physiques M-series) : **~8× additionnel** sur le single-thread.

| Configuration | Extraction 200k train | Full 1.18M train |
|---|---|---|
| Python + scipy (actuel) | ~38 min | ~4h |
| C++ single-thread | ~1-3 min | ~8-15 min |
| C++ + OpenMP 8 cores | ~15-30 sec | ~2-3 min |
