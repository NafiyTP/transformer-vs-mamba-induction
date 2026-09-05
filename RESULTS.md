# Résultats (état au 2026-09-05, projet complet)

## Stage 0 — Transformer baseline : confirmé

Transformer attention-only 2 couches, entraîné sur la tâche d'induction synthétique (checkpoint fourni par Yifan).

- **induction_accuracy sur un batch de test** (seed différent de l'entraînement) : **0.854**
- Scan des patterns d'attention par (couche, tête) : les deux têtes de la couche 1 ont une masse d'attention moyenne de **0.49** et **0.56** sur la position source attendue par l'induction, contre **0.02-0.03** pour la couche 0 (niveau chance).
- Voir `results/stage0_induction_head_scores.png` (heatmap couche x tête) et `results/stage0_best_head_attention_pattern.png` (pattern d'attention complet de la meilleure tête, montre la diagonale décalée caractéristique).

Conclusion : le schéma classique de la littérature (previous-token head en couche 0 → induction head en couche 1) est bien présent.

## Stage 2 (côté transformer) — Activation patching : confirmé, causalement

Corruption : les tokens de la source du bigramme répété sont remplacés par des tokens aléatoires frais, la cible et le masque d'induction restent identiques au batch clean.

- induction_accuracy clean : **0.840**
- induction_accuracy corrupted : **0.022** (= niveau chance, la corruption casse bien le signal)
- Score de récupération par patching (1.0 = récupération totale du niveau clean) :
  - couche 0, tête 0 : +0.008
  - couche 0, tête 1 : -0.011
  - **couche 1, tête 0 : +0.679**
  - **couche 1, tête 1 : +0.797**

Conclusion : accord total entre l'évidence corrélationnelle (patterns d'attention, Stage 0) et l'évidence causale (patching, Stage 2) — les deux têtes de couche 1 sont bien le circuit d'induction.

Code : `scripts/01b_analyze_baseline.py`, `scripts/03a_localize_transformer.py`, `src/analysis/patching.py::transformer_patch_heads`.

## Stage 1 — Mamba : mécanisme confirmé, entraînement à pleine échelle en cours

Journal de debug complet, honnête :

1. Premier entraînement (2000 steps, lr=1e-3) : loss reste bloquée exactement au niveau chance (ln(50)=3.91), aucune amélioration.
2. Vérifié que l'architecture peut apprendre du tout (memorisation d'un batch fixe : converge en ~30 steps, 98% accuracy) → pas un bug d'implémentation trivial.
3. Diagnostic ciblé : sur le modèle non-entraîné, l'effet d'un token modifié en position 2 sur la sortie s'effondre après ~5-6 positions, alors que la tâche demande de la mémoire sur 20-40 positions.
4. Cause identifiée : `dt_proj.weight` était initialisé aléatoirement, ce qui rendait le delta (donc la constante de temps de mémoire par canal) essentiellement aléatoire à chaque pas au lieu d'être contrôlé par le biais par canal. Fix : initialiser `dt_proj.weight` à zéro (comme dans l'implémentation officielle Mamba), pour que le delta soit purement piloté par le biais au départ.
5. Après le fix : vérifié que l'état caché h préserve bien l'info sur 20+ positions (`h_diff` décroît lentement). Mais la lecture (via C_t) de cette info restait quasi nulle au-delà de la fenêtre de la conv1d — normal à l'initialisation aléatoire, mais le signal de gradient pour apprendre ce readout était donc très faible au départ.
6. Toujours bloqué avec batch=128, lr=3e-3 : au niveau chance après 1800 steps, même avec la loss calculée uniquement sur les positions d'induction (élimine la confusion avec le "plancher" des positions non-prédictibles).
7. **Test décisif qui a débloqué le problème** : sur un batch FIXE (la même séquence répétée à chaque step, plutôt qu'un batch neuf à chaque fois), le modèle apprend l'induction parfaitement en ~100 steps (accuracy 1.0). Ça prouve que l'architecture *peut* représenter le mécanisme — le problème est spécifiquement la généralisation à travers des batchs différents.
8. Cause trouvée : le **weight decay par défaut d'AdamW (0.01)** empêchait le circuit de se former. En le mettant à 0 (`weight_decay=0.0`), une version simplifiée de la tâche (vocab=10, seq_len=20) montre une transition de phase nette (chance → 85% d'accuracy) en ~600-900 steps.
9. Nouveau problème rencontré à pleine échelle : la loss devient `nan` vers le step 550. Cause : sans weight decay, rien ne retient `A_log` de dériver sans limite ; une fois assez grand, `exp(A_log)` déborde vers l'infini, et si `delta` s'arrondit à exactement 0.0 pour un canal (arrondi float32), `delta * A` devient `0 * (-inf) = nan`. Fix : clamp de sécurité sur `A_log` (max 20) dans `effective_A()`, plus un plancher de 1e-6 sur `delta`, plus un garde-fou dans la boucle d'entraînement qui saute un step si la loss/le gradient devient non-fini.
10. Validation à difficulté intermédiaire (vocab=20, seq_len=32, d_model=64, mêmes réglages) : transition de phase nette, plateau à 82% d'accuracy d'induction, comparable au transformer (85%). Le mécanisme fonctionne bien sur cette implémentation.
11. **Entraînement à pleine échelle terminé** (vocab=50, seq_len=64, `AdamW(lr=5e-3, weight_decay=0.0)`, batch_size=256, gradient clipping à 1.0, `A_log` clampé à max 20, plancher 1e-6 sur delta) : transition de phase nette entre le step ~1000 et ~3000, plateau stable ensuite.

**Résultat final Stage 1** : **induction_accuracy = 0.869** sur batch de test (seed=123, protocole identique à l'évaluation du transformer) — légèrement au-dessus du transformer (0.854). Script reprenable par tranches (checkpoint modèle+optimiseur+historique tous les 200 steps) dans `scripts/02_train_mamba.py`.

## Stage 2 (côté Mamba) — Activation patching : circuit distribué, pas localisé

Même recette de corruption que côté transformer (source du bigramme remplacée par des tokens aléatoires), pour comparaison directe.

- induction_accuracy clean : **0.840**, corrupted : **0.024**
- Patcher un groupe de 16 canaux (sur 128) à la fois, dans une seule couche : **aucun groupe ne dépasse +3.5%** de récupération, y compris en couche 1 — très différent des têtes du transformer qui récupèrent seules 70-80%.
- Patcher la couche 1 **entière** (ses 128 canaux) seule : **+51.7%** de récupération.
- Patcher la couche 0 entière seule : quasi nul (+0.5%).
- Patcher les **deux couches entières** ensemble : **+100.3%** (récupération totale).
- Balayage de la fraction de canaux de la couche 1 patchés (couche 0 tenue clean) : 12%→+1.6%, 25%→+18%, 50%→+36%, 75%→+75%, 100%→+100%. Croissance à peu près régulière, pas de palier qui indiquerait un petit sous-ensemble suffisant.
- Score d'importance canal par canal (un seul canal patché à la fois) : aucun canal individuel ne dépasse +2.1%, moyenne à +0.7%.

**Conclusion** : contrairement au transformer, qui concentre l'induction dans 2 têtes clairement identifiables, ce Mamba implémente le même comportement de façon **distribuée** sur l'ensemble de la couche 1 : chaque canal contribue un petit peu, aucun n'est individuellement nécessaire ou suffisant, mais collectivement ils sont indispensables (et la couche 0 seule ne sert presque à rien, mais devient nécessaire en complément une fois la couche 1 patchée). C'est une différence qualitative réelle entre les deux architectures, pas juste une question d'échelle.

Code : `scripts/03b_localize_mamba.py`, `src/analysis/patching.py::mamba_patch_channel_groups`.

## Stage 3 — Analyse des pôles : résultat négatif informatif

Hypothèse testée (celle qui motivait le projet au départ, reliée à TSIA202b) : les canaux importants pour l'induction devraient avoir un pôle plus lent (plus proche de 1, donc plus proche du cercle unité comme un filtre à mémoire longue) que les canaux sans importance, puisqu'ils doivent conserver l'information sur tout l'écart entre les deux occurrences du bigramme.

- Pôle discret le plus lent par canal, calculé avec le delta moyen réellement observé aux positions d'induction sur un batch réel.
- **Corrélation (pôle, score d'importance individuel) : +0.046** — quasiment nulle.
- Pôle moyen des 20% de canaux les moins importants : **0.991**. Pôle moyen des 20% les plus importants : **0.992**. Essentiellement identiques.

**Conclusion** : l'hypothèse naïve ne tient pas. La quasi-totalité des canaux de la couche 1 ont déjà un pôle très proche de 1 (mémoire largement suffisante pour la tâche), donc la vitesse de mémorisation ne différencie pas les canaux importants des autres. Ce qui différencie probablement un canal important, c'est le contenu de ses projections B_t/C_t (est-ce qu'il code une direction utile pour identifier/retrouver un token), pas sa constante de temps. C'est un résultat négatif, mais un résultat négatif propre et interprété reste un vrai résultat scientifique — ça affine l'intuition de départ plutôt que de la confirmer aveuglément.

Code : `scripts/04_eigen_analysis.py`.

## Synthèse pour le portfolio

Trois résultats concrets et complets :
1. Le transformer et le Mamba atteignent tous les deux ~85-87% sur la tâche d'induction — capacité comparable.
2. Le mécanisme interne est qualitativement différent : circuit localisé (2 têtes) chez le transformer, circuit distribué (toute une couche) chez Mamba.
3. La théorie "mémoire longue = canal important" (motivée par l'intuition state-space/filtrage) est infirmée empiriquement : presque tous les canaux ont déjà une mémoire suffisante, ce n'est pas ce qui les différencie.

Journal de debug complet conservé ci-dessus : bug d'initialisation, effet du weight decay, instabilité numérique. Ce processus de découverte fait partie du résultat, pas juste la conclusion finale.
