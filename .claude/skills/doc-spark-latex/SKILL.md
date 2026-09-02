---
name: doc-spark-latex
description: Documente une fonction PySpark en LaTeX prêt à coller dans rapport_stage.tex — nom, signature et arguments, ce qu'elle fait, tables lues et schéma de la table produite. À utiliser dès que l'utilisateur colle du code Spark/PySpark (une ou plusieurs fonctions, une cellule de notebook) et demande de la doc, de la documentation, du LaTeX, ou de quoi remplir le rapport.
---

# Documenter une fonction Spark en LaTeX

Produit un **fragment LaTeX insérable** (pas de préambule, pas de `\begin{document}`)
documentant une ou plusieurs fonctions PySpark, aligné sur le préambule de
`rapport_stage.tex`.

## Ce que la sortie peut utiliser

Le document cible fournit déjà : `booktabs`, `listings` (style `conf`), `xcolor`,
`enumitem`, `amsmath`, `hyperref`, `babel` français, et la commande maison
`\code{...}` (= `\texttt{\small ...}`). N'introduisez **aucun** autre package.
Si le fragment a besoin d'autre chose, signalez-le en prose après le fragment
plutôt que d'ajouter un `\usepackage`.

## Procédure

1. **Lire le code, ne rien inventer.** Toute affirmation doit être traçable à une
   ligne du code collé. Si une information manque (sémantique métier d'une colonne,
   unité, granularité), écrivez-la comme hypothèse explicite (« vraisemblablement… »)
   ou omettez-la — ne la fabriquez pas.

2. **Extraire les entrées.** Pour chaque paramètre : nom, type annoté (ou déduit du
   corps si non annoté — le dire), valeur par défaut, rôle. Distinguez :
   - les paramètres scalaires / de configuration (`customer_shortname`, `cutoff_train`) ;
   - les paramètres `DataFrame`, qui sont des **tables amont** produites par une autre
     fonction du pipeline — nommez la fonction productrice si le code collé ou le repo
     permet de l'identifier, sinon décrivez le schéma attendu (colonnes lues sur ce
     DataFrame dans le corps).

3. **Extraire les tables lues.** Cherchez `spark.table(...)`, `spark.read...`,
   `.load(...)`, `spark.sql(...)`. Notez le nom Unity Catalog complet en trois parties
   et, pour chacune, les colonnes réellement utilisées (`select`, `where`, clés de
   jointure). C'est la partie la plus utile du document : listez les colonnes, pas
   seulement les tables.

4. **Décrire le traitement** en tirets (`itemize`), un par **bloc logique du code**
   (une variable nommée, une colonne calculée, l'expression finale) — jamais un
   tiret par appel Spark. Chaque tiret est **une phrase courte et précise** qui dit
   ce que le bloc représente et retient l'info clé (filtre décisif, type de
   jointure, clé, colonnes conservées), pas une paraphrase de l'API ligne à ligne.
   Une fonction à une seule expression chaînée n'a droit qu'à un ou deux tirets au
   total, même si elle enchaîne filtre + jointure + select. Référence de calibrage
   (à partir de `get_customer`) :

   ```
   \item \code{customers} : client T2S filtré sur \code{shortName}, projeté sur
         \code{publicId} et \code{databaseName} (renommé \code{db\_name}).
   \item \code{publishers} : clients Artemis rattachés aux \emph{publishers} par
         jointure \code{left} sur \code{insights\_customer\_id}, restreints au
         namespace \code{artemis-prod}, projetés sur \code{publisherId} et
         \code{public\_id}.
   \item Jointure finale \code{inner} \code{customers}/\code{publishers} sur
         \code{publicId} = \code{public\_id} ; sélection des quatre colonnes de
         sortie.
   ```

   Trois tirets pour toute la fonction — pas onze. Aucune prose narrative, aucune
   justification métier sauf si elle change le sens d'un filtre. Un `cache()` sans
   effet fonctionnel se glisse dans la phrase du bloc qui le porte, il n'a pas son
   propre tiret. Pour une fonction à ~10-20 blocs nommés (variables intermédiaires
   nombreuses), gardez un tiret par bloc mais une seule phrase chacun — la longueur
   totale vient du nombre de blocs, pas de la verbosité de chaque phrase.

5. **Décrire la table de sortie.** Dérivez le schéma du `.select(...)` final : nom de
   colonne après `alias`, type, provenance (table + colonne d'origine, ou expression).
   Précisez la **granularité** de la ligne de sortie (une ligne par quoi ?) — c'est
   déductible des `groupBy` / `distinct` / clés de jointure.

   Attention : les **types** ne se déduisent pas du code. Sauf cast explicite
   (`.cast(...)`) ou expression au type évident (`F.count`, `F.lit`), ils viennent du
   schéma des tables amont, que le code collé ne contient pas.

   **Vérifiez-les plutôt que de mettre `?`.** Le CLI Databricks est authentifié dans
   cet environnement (profil `mirakl-ai` — `databricks auth profiles` pour
   confirmer). Pour chaque table lue ou colonne dont le type est incertain :

   ```
   databricks --profile mirakl-ai tables get <catalog>.<schema>.<table>
   ```

   renvoie le schéma Unity Catalog complet en JSON (`columns[].type_name` /
   `type_text`). Pour un champ `STRUCT` ou `ARRAY<STRUCT>` imbriqué (ex.
   `sponsoredProductPlacementExecutions`), le `type_text` brut peut être énorme et
   imbriqué sur plusieurs niveaux — parsez `type_json` en Python plutôt que de lire
   la chaîne à l'œil, et gardez la colonne **Type** du tableau LaTeX courte
   (`array<struct>`, pas le `type_text` complet, qui ferait déborder une colonne de
   `tabular`) ; détaillez les champs exploités en prose dans la colonne
   **Provenance et sens** seulement s'ils tiennent en une ligne courte.

   Deux colonnes de tables différentes (ou même d'une seule table à deux niveaux
   d'imbrication différents) peuvent porter le même nom avec des schémas
   distincts — vérifiez toujours plutôt que de supposer qu'un nom déjà vu ailleurs
   a le même type ici.

   Seulement si la table n'existe pas encore, ou si l'utilisateur n'a pas donné
   accès au catalogue, annoncez le type comme déduit du nommage — ne le présentez
   jamais comme relevé dans ce cas.

6. **Écrire le fragment** en suivant `references/template.tex`, dont
   `references/exemple.tex` montre un remplissage complet sur une vraie fonction du
   dépôt. Respectez les règles d'échappement ci-dessous.

   L'intro (avant \code{Signature}) fait **une à deux phrases**, jamais plus : ce que
   la fonction produit et à partir de quoi, sans reformuler ce que le tableau des
   arguments ou les tirets du traitement disent déjà. Pas de mise en contexte du
   pipeline, pas de « cette fonction permet de… » — direct : sujet, verbe,
   complément.

7. **Livrer** le fragment dans un bloc de code LaTeX dans la réponse. N'éditez
   `rapport_stage.tex` que si l'utilisateur le demande — dans ce cas, insérez-le à
   l'endroit cohérent avec le plan existant et proposez un `\label` unique.

## Échappement — non négociable

Les noms Unity Catalog sont pleins de `_`, qui font échouer la compilation en mode
texte. Dans tout `\code{}`, `\caption{}`, cellule de tableau ou prose :

| Caractère | Écrire |
|---|---|
| `_` | `\_` |
| `%` | `\%` |
| `&` | `\&` |
| `#` | `\#` |
| `$` | `\$` |

Donc `\code{mirakl\_ai.ds\_etl\_prod.t2s\_gold\_customer}`, jamais
`\code{mirakl_ai...}`. Seule exception : à l'intérieur d'un environnement
`lstlisting`, le code est verbatim et **ne doit pas** être échappé.

Les noms de tables en trois parties sont longs et ne se coupent pas en `\ttfamily` :
mettez-les dans un `itemize`, pas dans une colonne de `tabular` étroite (le template
suit déjà cette règle).

Même règle pour une colonne `l`/`c`/`r` de `tabular` : ces types de colonne n'ont
**aucune largeur maximale** et ne coupent jamais leur contenu, même s'il y a des
espaces — seul `p{<largeur>}` le fait. Si une cellule combine plusieurs identifiants
(`\code{train\_pos\_edge\_index}, \code{train\_neg\_edge\_index}`), mettez cette
colonne en `p{<largeur>}`, sinon la colonne s'élargit pour tenir tout le texte sur
une seule ligne et le tableau déborde de la page. C'est **silencieux** : dans un
`\begin{center}`, `pdflatex` ne produit **aucun** warning *Overfull hbox* même quand
le contenu dépasse largement la marge — `center` centre la boîte quelle que soit sa
largeur, il ne la contraint pas. Donc pour tout tableau dont une colonne combine
plusieurs `\code{}` ou dont vous doutez de la largeur, ne vous fiez pas au log
`pdflatex` : rendez la page en image (`gs -sDEVICE=png16m ...` sur le PDF, cf.
section Vérification) et regardez-la.

Les tableaux (arguments, table produite) vont dans `\begin{center} ... \end{center}`,
**jamais** dans `\begin{table}[h] ... \end{table}`. `table` est un environnement
flottant : sans `\caption`/`\label` (il n'y en a pas besoin ici), LaTeX n'a aucune
raison de le garder collé à son paragraphe, et avec plusieurs de ces tableaux à la
suite dans un même document, ils dérivent et peuvent s'empiler sous la mauvaise
sous-section. `center` reste fixé exactement là où il est écrit.

`\paragraph{Signature.}` doit **toujours** être suivi de `\leavevmode\par` avant le
`\begin{lstlisting}`, jamais de `\begin{lstlisting}` directement. Sans ce `\par`,
`\paragraph` ne force aucun saut de ligne : la bordure supérieure du cadre du listing
se plaque contre la ligne du titre au lieu de démarrer dessous, ce qui donne un trait
gris flottant collé à « Signature. » dans le PDF rendu. Le template et l'exemple
appliquent déjà ce correctif — ne le retirez pas en copiant le motif.

## Plusieurs fonctions d'un coup

Si l'utilisateur colle un pipeline entier, produisez une `\subsection` par fonction
dans l'ordre du flux de données, et ouvrez par une phrase qui donne la chaîne
(`get_customer` → `get_products` → `sessionize_views` → …). Ne dupliquez pas la
description d'une table déjà documentée : renvoyez-y par `\ref{}`.

## Vérification optionnelle

Si l'utilisateur veut la garantie que ça compile, écrivez le fragment dans le
scratchpad entouré d'un préambule minimal reprenant celui de `rapport_stage.tex`,
puis lancez `pdflatex -interaction=nonstopmode`. Ne le faites que sur demande, ou si
le fragment contient une construction inhabituelle dont vous doutez.

Piège : le shell est `zsh`, où `echo` interprète les échappements. `echo
'\begin{document}'` écrit un octet backspace (`\b`) et `echo '\end{document}'` un
ESC (`\e`), ce qui fait échouer la compilation sur une erreur *Unicode character
^^H* trompeuse, à une ligne du préambule et non du fragment. Utilisez `printf
'%s\n'` ou un heredoc `<<'EOF'` pour assembler le fichier de test.
