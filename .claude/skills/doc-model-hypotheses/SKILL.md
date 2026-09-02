---
name: doc-model-hypotheses
description: Pour une fonction de model_test_prototype_clean.ipynb, propose une docstring et les lignes correspondantes des deux tableaux de model_test_prototype_documentation.tex (Hypothèse métier / Si elle est fausse, et Mécanisme / Ce qui disparaît), précises et concises, calibrées sur l'exemple de run_inference ci-dessous. À utiliser quand l'utilisateur colle une fonction de model_test_prototype_clean, ou demande de la documenter, de proposer les hypothèses, ou de remplir le document étape par étape.
---

# Documenter une fonction de model_test_prototype_clean.ipynb

Pour une fonction collée par l'utilisateur, produit trois choses : une docstring,
des lignes proposées pour le tableau « Hypothèses métier », et des lignes proposées
pour le tableau « Ce qui peut être exclu de l'évaluation ». Le document cible est
`model_test_prototype_documentation.tex` (tableaux `tab:hypotheses` et `tab:exclusions`).

C'est le pendant, pour `model_test_prototype_clean.ipynb`, de ce que
`doc-graph-hypotheses` fait pour `graph_pipeline_prototype.ipynb` — même structure de
document, même niveau d'exigence. Les deux notebooks ne se recoupent pas (l'un
construit le graphe de co-vue, l'autre évalue un modèle déjà entraîné dessus) : ne
mélangez jamais leurs fonctions ou leurs documents.

## Principe : proposer, pas insérer

Ce skill **propose** dans la réponse — il ne modifie
`model_test_prototype_documentation.tex` que si l'utilisateur le demande explicitement
ensuite. L'utilisateur valide ou corrige avant qu'une ligne devienne définitive.

## 1. Lire le code, ne rien inventer

Même règle que pour toute documentation de ce dépôt : chaque phrase doit être
traçable à une ligne du code collé. Une hypothèse métier non vérifiable dans le
code se marque « à confirmer » plutôt que d'être affirmée.

Ce notebook manipule des objets propres à son domaine (`triggers`, `candidates_by_category`,
`top12_model`, `exec_code`) — vérifiez toujours contre la fonction réellement collée, pas
contre votre souvenir d'une fonction similaire ailleurs dans le notebook.

## 2. Proposer la docstring

Même convention Google docstring que `graph_pipeline_prototype.ipynb` (via
`doc-graph-hypotheses`) : une phrase de résumé, un paragraphe optionnel si
l'implémentation contient un choix non évident, puis `Args:` (uniquement les
paramètres non triviaux) et `Returns:`.

- `Returns:` pour un `list[dict]` : les clés produites, la granularité (un élément
  par quoi ?), et tout élément silencieusement absent du résultat (filtre, `continue`,
  `dropDuplicates`). C'est le pont direct vers le tableau d'exclusion de l'étape 4 —
  exactement le rôle que joue `Returns:` côté graphe pour un `DataFrame`.
- Ne documentez pas `Raises:` sauf exception explicite dans le code.

Calibrage — docstring de `extract_all_triggers`, corrigée pour inclure `Args:`/`Returns:`,
la référence à reproduire :

```python
def extract_all_triggers(prod_results: DataFrame) -> list[dict]:
    """Tous les triggers de la table prod prototype.

    Remplace extract_triggers (qui partait de val_split["edge"], donc des seules executions ayant
    au moins un positif dans le top-12 prod) et extract_negative_only_triggers. Le tenseur d'edges
    n'est plus utilise ici: la population d'analyse est definie independamment de l'etiquetage.

    Args:
        prod_results: DataFrame prod prototype, doit contenir exec_code et trigger_internal_id.

    Returns:
        Liste de dicts (un par exec_code distinct), avec les cles exec_code,
        trigger_internal_id et trigger_node_id. Un trigger dont le produit declencheur
        n'a pas de node_id dans node2idx (hors graphe) est absent du resultat.
    """
```

- Le résumé nomme ce que la fonction produit, pas comment.
- Le paragraphe de rationale (avant `Args:`), quand il existe, porte une information
  qui ne se lit pas déjà dans le nom de la fonction ou ses arguments — ici, un
  contraste avec l'ancienne version.
- Une fonction déjà simple (ex. `ids_in_rank_order`) n'a pas besoin de paragraphe de
  rationale, ni forcément d'`Args:`/`Returns:` si le type suffit déjà (ex. `-> int`
  trivial) : ne forcez pas.
- Livrez la docstring seule, prête à remplacer celle existante (ou à insérer s'il n'y
  en a pas), dans un bloc de code Python.

## 3. Proposer les lignes « Hypothèses métier »

Une ligne par **hypothèse distincte**, pas une par ligne de code. Une fonction
d'infrastructure (chargement de checkpoint, listing S3, affichage) n'en a
généralement aucune ; une fonction qui décide quels candidats scorer, quel filtre
appliquer, ou ce qui compte comme positif/négatif en a typiquement une à trois.
Ne forcez jamais une ligne s'il n'y a rien à dire.

Format de chaque ligne (colonnes du tableau `tab:hypotheses`) :

| Colonne | Contenu |
|---|---|
| Id | `Hx` (placeholder — cf. étape 5 pour la numérotation réelle) |
| Étape | `\url{nom_fonction}` (voir encadré ci-dessous — pas `\code{}` ici) |
| Hypothèse métier | Le fait métier qui doit être vrai pour que l'étape ait un sens |
| Si elle est fausse | La conséquence concrète si ce fait est faux |

Précis et concis, mais pas édulcoré : nommez l'opération et l'identifiant exacts qui
font que l'hypothèse est vérifiable (le nom de la structure de données, la clé du
dict, la fonction Spark en cause), en `\code{}`. Le jargon à éviter, c'est le
vocabulaire d'analyse abstrait qui n'apporte rien de vérifiable — pas les noms
d'opérations ou de champs du code, qui sont au contraire ce qui rend la ligne
exploitable.

**Une phrase par cellule, deux au grand maximum.** Le calibrage n'est pas
`pipeline_hypotheses.tex` ici (le document démarre vide), mais la longueur cible est
la même : regardez la taille des cellules H1/E1 de `graph_pipeline_documentation.tex`
(`get_customer`) et visez la même longueur. Signes à couper avant de livrer une
ligne :
- une clause conditionnelle emboîtée (« si X était nécessaire, alors Y ») au lieu
  d'énoncer directement le fait vérifiable — l'hypothèse est un fait sur le code, pas
  un scénario hypothétique sur un ancien protocole ;
- une parenthèse qui répète en français ce que le `\code{}` voisin dit déjà ;
- plus d'une sub-clause reliée par une virgule ou un point-virgule dans « Si elle est
  fausse » — une seule conséquence concrète, pas la liste de toutes les consequences
  possibles.

Calibrage — lignes H/E travaillées à partir de `run_inference`, la référence à
reproduire (le code correspondant : `candidates_to_score = candidates_by_category[trigger["category"]]`,
sans autre restriction que d'exclure le trigger lui-même) :

```
Hypothèse métier : Le pool de candidats scorés couvre toute la catégorie du trigger,
pas seulement les 12 candidats déjà proposés par la prod.
Si elle est fausse : Le score modèle et le score prod ne comparent plus deux
stratégies sur le même univers de candidats.
```

## 4. Proposer les lignes « Ce qui peut être exclu de l'évaluation »

Une ligne par **mécanisme qui retire silencieusement des triggers, des candidats ou
des lignes** dans cette fonction (`continue` sur une condition, filtre, jointure
`inner`/`anti`/`left_semi`, `dropna`, `dropDuplicates`, seuil). Cherchez-les dans le
code, ne les déduisez pas d'ailleurs. S'il n'y en a aucun dans la fonction, dites-le
au lieu de forcer une ligne.

**Ne dupliquez pas une ligne H déjà proposée pour la même ligne de code.** Si le
mécanisme d'exclusion est exactement ce que H décrit déjà (l'hypothèse EST le
filtre, sans justification métier distincte à en tirer), ne proposez pas de ligne E
correspondante — gardez seulement H. Ce piège s'est produit avec
`extract_all_triggers` : `dropDuplicates(["exec_code"])` et le filtre `node2idx` ont
d'abord généré une ligne H et une ligne E quasi identiques chacun ; les deux lignes
E ont été retirées, H seule suffisait. Une ligne E n'est justifiée que si elle
apporte un fait mécanique (quel volume, quelle condition exacte) qui n'était pas
déjà dans la formulation métier de H — ou si aucune ligne H n'a été proposée pour
cette ligne de code.

Même piège que côté graphe : une fonction qui **construit un sous-ensemble dérivé
d'un univers plus restreint que ses entrées amont** (ex. un pool de candidats limité
à ce qui a un `node_id`, alors que le catalogue amont est plus large) exclut
silencieusement tout ce qui est hors de ce sous-ensemble — même sans filtre,
jointure ou `dropna` visible dans la fonction elle-même. Demandez-vous
systématiquement : « est-ce que cette fonction part d'un univers plus restreint que
celui d'où viennent ses entrées amont (catalogue, triggers, pool) ? » avant de
conclure qu'il n'y a pas de ligne E.

| Colonne | Contenu |
|---|---|
| Id | `Ex` (placeholder) |
| Étape | `\url{nom_fonction}` (voir encadré ci-dessous — pas `\code{}` ici) |
| Mécanisme | L'opération qui retire des triggers/lignes, en une phrase simple |
| Ce qui disparaît | Ce qui est concrètement perdu (quels triggers, quels candidats, quelles paires) |

Même règle de brièveté qu'à l'étape 3 : une phrase par cellule, deux au maximum.

Calibrage — ligne E travaillée à partir de `run_inference` (le code correspondant :
`if trigger["category"] is None: skipped_no_candidates += 1; continue` et
`if candidates_to_score.numel() == 0: skipped_no_candidates += 1; continue`) :

```
Mécanisme : \code{trigger["category"]} absent, ou pool de candidats vide après
retrait du trigger.
Ce qui disparaît : Les triggers sans catégorie résolue ou sans autre produit dans
leur catégorie --- comptés (\code{skipped\_no\_candidates}) mais absents de
\code{top12\_model}.
```

## La colonne Étape : `\url{}`, jamais `\code{}`

La colonne Étape des deux tableaux est étroite (`p{2.45cm}` pour les hypothèses,
`p{3.2cm}` pour les exclusions). `\code{}` (= `\texttt{\small}`) ne coupe jamais un
identifiant en `\ttfamily` au milieu d'un mot — un nom de fonction un peu long
(`extract_all_triggers`, `build_candidates_by_category`, `stage_model_rows`) dépasse
alors silencieusement dans la colonne « Hypothèse métier »/« Mécanisme ».

Le correctif : dans la colonne Étape uniquement, utilisez `\url{nom_fonction}` (la
commande vient de `url.sty`, déjà chargé via `hyperref` — aucun `\usepackage` à
ajouter). `\url{}` coupe automatiquement aux `_`, et surtout **ne prend pas les
identifiants échappés** : écrivez `\url{build_candidates_by_category}`, avec un `_`
littéral, pas `\url{build\_candidates\_by\_category}`. `\code{}` reste inchangé
partout ailleurs (colonnes Hypothèse métier / Si elle est fausse / Mécanisme / Ce
qui disparaît, et dans la docstring) — seule la colonne Étape change de commande.

```
H4 & \url{build_candidates_by_category} & Le pool de candidats ...
E4 & \url{build_candidates_by_category} & \code{if i in node2idx} & ...
```

Même piège dans les colonnes Mécanisme / Ce qui disparaît (`p{4.4cm}`, `p{5.2cm}`),
mais `\url{}` n'y aide pas : ces cellules mélangent souvent du texte français et du
`\code{}`, et un accès à une clé de dict ou un appel Spark peut rester long et
incassable même en `\url{}`. Le correctif ici n'est pas une commande mais la
formulation : élidez la partie non essentielle avec `(...)` et décrivez la cible en
français plutôt que de citer l'expression complète.

## 5. Numérotation

Avant de proposer les Id, lisez `model_test_prototype_documentation.tex` et repérez
le plus grand `Hn` déjà présent dans le tableau `tab:hypotheses` et le plus grand `En`
déjà présent dans `tab:exclusions` ; proposez la suite (`H{n+1}`, `E{n+1}`, …). Le
document démarre vide : commencez à `H1` / `E1` s'il n'y a encore aucune ligne.
Vérifiez aussi qu'aucune ligne existante ne documente déjà cette fonction, pour ne
pas dupliquer.

## 6. Livrer

Trois blocs dans la réponse :
1. La docstring (bloc de code Python).
2. Les lignes proposées pour « Hypothèses métier », en LaTeX prêtes à coller (une
   ligne de tableau par hypothèse, séparées par `\addlinespace`), ou une phrase
   disant qu'il n'y en a pas.
3. Même chose pour « Ce qui peut être exclu de l'évaluation ».

N'éditez `model_test_prototype_documentation.tex` que si l'utilisateur le demande —
dans ce cas insérez juste avant `\end{longtable}` du tableau concerné, avec
`\addlinespace` avant chaque nouvelle ligne sauf si le tableau est vide.

`model_test_prototype_documentation.tex` ne charge **pas** `amsmath`. `\land`,
`\rightarrow`, l'indiçage simple (`$n$`) fonctionnent en TeX de base ; `\binom{}{}`,
`\emph` imbriqué dans certains contextes mathématiques ou tout autre symbole
spécifique à amsmath causeront une erreur `Undefined control sequence`. Écrivez le
calcul en notation simple au lieu d'ajouter le `\usepackage`.

Dès que `model_test_prototype_documentation.tex` est modifié (une ou les deux
tables), recompilez-le immédiatement, sans attendre que l'utilisateur le demande :

```
pdflatex -interaction=nonstopmode model_test_prototype_documentation.tex
```

Relancez une seconde fois si le log affiche un avertissement `Rerun` (« Label(s) may
have changed », ou `longtable Warning: Table widths have changed ` — ce dernier
apparaît dès qu'une ligne ajoutée change la répartition des lignes sur les pages).
Si la compilation échoue, montrez l'erreur et corrigez l'insertion avant de rendre
la main — n'annoncez pas les lignes comme ajoutées si le PDF ne se génère pas.

Grep aussi le log pour `Overfull` (`pdflatex ... | grep -i overfull`) : c'est le
signal qu'une cellule déborde de sa colonne, en particulier la colonne Étape (voir
l'encadré `\url{}` ci-dessus). Si ça apparaît sur une ligne que vous venez d'ajouter,
corrigez avant de rendre la main. Une fois propre, supprimez les auxiliaires
(`.aux`, `.log`, `.out`) et ne gardez que le `.tex` et le `.pdf` à jour.

## Échappement

Mêmes règles que pour tout ce document : `_` → `\_`, `%` → `\%`, `&` → `\&`, `#` → `\#`,
`$` → `\$`, partout sauf dans un environnement verbatim. Un identifiant de fonction ou
de colonne dans une cellule de tableau doit être échappé.
