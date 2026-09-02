---
name: doc-graph-hypotheses
description: Pour une fonction de graph_pipeline_prototype.ipynb, propose une docstring et les lignes correspondantes des deux tableaux de graph_pipeline_documentation.tex (Hypothèse métier / Si elle est fausse, et Mécanisme / Ce qui disparaît), précises et concises, calibrées sur pipeline_hypotheses.tex. À utiliser quand l'utilisateur colle une fonction du notebook et demande de la documenter, de proposer les hypothèses, ou de remplir le document étape par étape.
---

# Documenter une fonction de graph_pipeline_prototype.ipynb

Pour une fonction collée par l'utilisateur, produit trois choses : une docstring,
des lignes proposées pour le tableau « Hypothèses métier », et des lignes proposées
pour le tableau « Ce qui peut être exclu du graphe ». Le document cible est
`graph_pipeline_documentation.tex` (tableaux `tab:hypotheses` et `tab:exclusions`).

## Principe : proposer, pas insérer

Ce skill **propose** dans la réponse — il ne modifie `graph_pipeline_documentation.tex`
que si l'utilisateur le demande explicitement ensuite. L'utilisateur valide ou corrige
avant qu'une ligne devienne définitive.

## 1. Lire le code, ne rien inventer

Même règle que pour toute documentation de ce dépôt : chaque phrase doit être
traçable à une ligne du code collé. Une hypothèse métier non vérifiable dans le
code se marque « à confirmer » plutôt que d'être affirmée.

## 2. Proposer la docstring

Style à suivre — Google docstring, en français, calibré sur l'exemple le plus
complet déjà présent dans le notebook (`get_customer`) :

```python
def get_customer(spark: SparkSession, customer_shortname: str) -> DataFrame:
    """Résout un client vers son publisherId Artemis.

    Args:
        spark: Session Spark active.
        customer_shortname: shortName du client dans t2s_gold_customer.

    Returns:
        DataFrame avec les colonnes customer_shortname, customerId, db_name,
        publisherId. Un client sans publisher Artemis correspondant est absent
        du résultat (inner join).
    """
```

- Une phrase de résumé, puis `Args:` (uniquement les paramètres non triviaux —
  inutile de documenter un `spark: SparkSession` répété partout) et `Returns:`.
- `Returns:` pour un `DataFrame` : colonnes produites + granularité (une ligne par
  quoi ?) + toute ligne silencieusement perdue en route (filtre, `inner join`).
  C'est le pont direct vers le tableau d'exclusion de l'étape 4.
- Ne documentez pas `Raises:` sauf exception explicite dans le code.
- Certaines fonctions du notebook n'ont qu'un résumé d'une ligne sans `Args`/`Returns` :
  ce n'est pas la cible, alignez-vous sur `get_customer`, pas sur ces fonctions-là.
- Livrez la docstring seule, prête à remplacer celle existante (ou à insérer s'il n'y
  en a pas), dans un bloc de code Python.

## 3. Proposer les lignes « Hypothèses métier »

Une ligne par **hypothèse distincte**, pas une par ligne de code. Une fonction
simple peut n'en avoir aucune ; une fonction dense (plusieurs filtres/jointures
indépendants) peut en avoir trois ou quatre. Ne forcez jamais une ligne s'il n'y a
rien à dire.

Format de chaque ligne (colonnes du tableau `tab:hypotheses`) :

| Colonne | Contenu |
|---|---|
| Id | `Hx` (placeholder — cf. étape 5 pour la numérotation réelle) |
| Étape | `\url{nom_fonction}` (voir encadré ci-dessous — pas `\code{}` ici) |
| Hypothèse métier | Le fait métier qui doit être vrai pour que l'étape ait un sens |
| Si elle est fausse | La conséquence concrète si ce fait est faux |

Précis et concis, mais pas édulcoré : nommez l'opération et l'identifiant exacts
qui font que l'hypothèse est vérifiable (le type de jointure, la fonction Spark en
cause, le nom de colonne), en `\code{}`. Le jargon à éviter, c'est le vocabulaire
d'analyse abstrait qui n'apporte rien de vérifiable (« granularité », « idempotent »,
« canonise ») — pas les noms d'opérations ou de colonnes du code, qui sont au
contraire ce qui rend la ligne exploitable. Une à deux phrases courtes par cellule,
au niveau de précision de `pipeline_hypotheses.tex`.

Calibrage — lignes H1/E1 réelles de `pipeline_hypotheses.tex` pour `get_customer`,
la référence à reproduire :

```
Hypothèse métier : Le \code{shortName} désigne un client unique, qui possède un
publisher dans le namespace \code{artemis-prod}.
Si elle est fausse : La jointure \code{inner} rend le client invisible, ou
\code{collect()} renvoie plusieurs \code{db\_name} et le graphe mélange deux
catalogues.
```

## 4. Proposer les lignes « Ce qui peut être exclu du graphe »

Une ligne par **mécanisme qui retire silencieusement des lignes** dans cette
fonction (filtre, jointure `inner`/`anti`, `dropna`, `dropDuplicates`, `distinct`,
seuil). Cherchez-les dans le code, ne les déduisez pas d'ailleurs. S'il n'y en a
aucun dans la fonction (ex. une fonction purement géométrique/tensorielle qui ne
filtre rien), dites-le au lieu de forcer une ligne.

**Ne dupliquez pas une ligne H déjà proposée pour la même ligne de code.** Si le
mécanisme d'exclusion est exactement ce que H décrit déjà (l'hypothèse EST le
filtre, sans fait mécanique distinct à en tirer), ne proposez pas de ligne E
correspondante — gardez seulement H. Une ligne E n'est justifiée que si elle
apporte quelque chose que H n'a pas (le volume concret, la condition exacte) — ou
si aucune ligne H n'a été proposée pour cette ligne de code.

Piège déjà raté une fois : une fonction qui **construit un ensemble d'identifiants
dérivé d'un sous-ensemble** (ex. `build_node_mapping` qui prend les nœuds des
*arêtes* fournies) plutôt que de l'univers amont complet (le catalogue) exclut
silencieusement tout ce qui est dans l'univers amont mais absent du sous-ensemble
— même sans filtre, jointure ou `dropna` visible. Demandez-vous systématiquement :
« est-ce que cette fonction part d'un univers plus restreint que celui d'où
viennent ses entrées amont (catalogue, embeddings, univers de vues) ? » avant de
conclure qu'il n'y a pas de ligne E.

| Colonne | Contenu |
|---|---|
| Id | `Ex` (placeholder) |
| Étape | `\url{nom_fonction}` (voir encadré ci-dessous — pas `\code{}` ici) |
| Mécanisme | L'opération qui retire des lignes, en une phrase simple |
| Ce qui disparaît | Ce qui est concrètement perdu (quels produits, quelles vues, quels clients) |

Calibrage — ligne E1 réelle de `pipeline_hypotheses.tex` pour `get_customer` :

```
Mécanisme : jointure \code{inner} clients / publishers.
Ce qui disparaît : Un client sans \emph{publisher} \code{artemis-prod} : le
notebook s'exécute alors sur des listes vides sans erreur.
```

## La colonne Étape : `\url{}`, jamais `\code{}`

La colonne Étape des deux tableaux est étroite (`p{2.45cm}` pour les hypothèses,
`p{3.2cm}` pour les exclusions). `\code{}` (= `\texttt{\small}`) ne coupe jamais un
identifiant en `\ttfamily` au milieu d'un mot — un nom de fonction un peu long
(`get_embeddings`, `sessionize_views`, et pire pour les fonctions plus loin dans le
notebook : `build_co_view_edges`, `recommended_products_for_t2s_user`) dépasse alors
silencieusement dans la colonne « Hypothèse métier »/« Mécanisme ». C'est passé
inaperçu deux fonctions de suite avant d'être corrigé (`Overfull \hbox` dans le
log, invisible tant qu'on ne regarde pas le rendu).

Le correctif : dans la colonne Étape uniquement, utilisez `\url{nom_fonction}`
(la commande vient de `url.sty`, déjà chargé via `hyperref` — aucun `\usepackage`
à ajouter). `\url{}` coupe automatiquement aux `_`, et surtout **ne prend pas les
identifiants échappés** : écrivez `\url{sessionize_views}`, avec un `_` littéral,
pas `\url{sessionize\_views}`. `\code{}` reste inchangé partout ailleurs (colonnes
Hypothèse métier / Si elle est fausse / Mécanisme / Ce qui disparaît, et dans la
docstring) — seule la colonne Étape change de commande.

```
H5 & \url{get_embeddings} & Un modèle d'embedding figé (\code{last\_model\_id}) ...
E4 & \url{get_embeddings} & \code{last\_model\_id} figé & ...
```

Même piège dans les colonnes Mécanisme / Ce qui disparaît (`p{4.4cm}`, `p{5.2cm}`),
mais `\url{}` n'y aide pas : ces cellules mélangent souvent du texte français et du
`\code{}`, et surtout le nom en cause est parfois un identifiant Spark/camelCase
**sans** `_` ni `-` (ex. `sponsoredProductPlacementExecutions`) — `\url{}` ne coupe
qu'aux séparateurs, donc un tel mot reste tout aussi incassable en `\url{}` qu'en
`\code{}`. Le correctif ici n'est pas une commande mais la formulation : élidez
l'identifiant complet avec `(...)` et décrivez la cible en français, comme le fait
déjà `E7` (`\code{emitted.between(...)}` plutôt que les deux arguments complets) —
```
E12 & ... & \code{F.explode(...)} sur le tableau des placements sponsorisés & ...
```
plutôt que `\code{F.explode("sponsoredProductPlacementExecutions")}`. Vérifiez le
log (`grep -i overfull`, cf. étape 6) sur ces deux colonnes aussi, pas seulement sur
Étape.

## 5. Numérotation

Avant de proposer les Id, lisez `graph_pipeline_documentation.tex` et repérez le
plus grand `Hn` déjà présent dans le tableau `tab:hypotheses` et le plus grand `En`
déjà présent dans `tab:exclusions` ; proposez la suite (`H{n+1}`, `E{n+1}`, …). Si le
document est encore vide, commencez à `H1` / `E1`. Vérifiez aussi qu'aucune ligne
existante ne documente déjà cette fonction, pour ne pas dupliquer.

## 6. Livrer

Trois blocs dans la réponse :
1. La docstring (bloc de code Python).
2. Les lignes proposées pour « Hypothèses métier », en LaTeX prêtes à coller
   (une ligne de tableau par hypothèse, séparées par `\addlinespace`), ou une phrase
   disant qu'il n'y en a pas.
3. Même chose pour « Ce qui peut être exclu du graphe ».

N'éditez `graph_pipeline_documentation.tex` que si l'utilisateur le demande — dans ce
cas insérez juste avant `\end{longtable}` du tableau concerné, avec `\addlinespace`
avant chaque nouvelle ligne sauf si le tableau est vide.

`graph_pipeline_documentation.tex` ne charge **pas** `amsmath` (contrairement à
`pipeline_hypotheses.tex`, qui elle le charge). `\land`, `\rightarrow`, l'indiçage
simple (`$n$`, `$n(n-1)/2$`) fonctionnent en TeX de base ; `\binom{}{}`, `\emph`
imbriqué dans certains contextes mathématiques ou tout autre symbole spécifique à
amsmath causeront une erreur `Undefined control sequence`. Écrivez le calcul en
notation simple (`n(n-1)/2` plutôt que `\binom{n}{2}`) au lieu d'ajouter le
`\usepackage`.

Dès que `graph_pipeline_documentation.tex` est modifié (une ou les deux tables),
recompilez-le immédiatement, sans attendre que l'utilisateur le demande :

```
pdflatex -interaction=nonstopmode graph_pipeline_documentation.tex
```

Relancez une seconde fois si le log affiche un avertissement `Rerun` (« Label(s) may
have changed », ou `longtable Warning: Table widths have changed ` — ce dernier
apparaît dès qu'une ligne ajoutée change la répartition des lignes sur les pages) :
c'est fréquent dès qu'on ajoute des lignes à un `longtable` déjà rempli, pas
seulement à cause des références croisées. Si la compilation échoue, montrez
l'erreur et corrigez l'insertion avant de rendre la main — n'annoncez pas les
lignes comme ajoutées si le PDF ne se génère pas.

Grep aussi le log pour `Overfull` (`pdflatex ... | grep -i overfull`) : c'est le
signal qu'une cellule déborde de sa colonne, en particulier la colonne Étape (voir
l'encadré `\url{}` ci-dessus) — ça ne fait pas échouer la compilation et passe
inaperçu si on ne vérifie pas. Si ça apparaît sur une ligne que vous venez
d'ajouter, corrigez avant de rendre la main. Une fois propre, supprimez les
auxiliaires (`.aux`, `.log`, `.out`) et ne gardez que le `.tex` et le `.pdf` à jour.

## Échappement

Mêmes règles que pour tout ce document : `_` → `\_`, `%` → `\%`, `&` → `\&`, `#` → `\#`,
`$` → `\$`, partout sauf dans un environnement verbatim. Un identifiant de fonction ou
de colonne dans une cellule de tableau doit être échappé.
