**📅 Mercredi 17 juin 2026 — Brief quotidien « Artisan Prévisionniste »**

**Thème du jour : Modèles de fondation pour séries temporelles (TSFM) — du zero-shot au few-shot, et la question de la viabilité opérationnelle (routage de complexité).**

Bonjour. Aujourd'hui on attaque le sujet qui sépare le praticien junior du senior en 2026 : non pas « savoir lancer un TSFM », mais savoir **quand il bat un spécialiste supervisé et quand il  perd**, puis industrialiser cette décision. C'est exactement là que se trouve ta valeur de freelance.

---

## 📋 LIVRABLE 1 — Plan d'action du jour (~55 min)

### 1. Veille technologique & scientifique (12 min)

- **IBM's March 2026 Time-Series Refresh (FlowState r1.1, TTM-r3, PatchTST-FM r1, TSPulse r1)** — TSFM.ai, 12 avril 2026 → https://tsfm.ai/blog/ibm-march-2026-time-series-refresh*Pertinent aujourd'hui :* les trois modèles de prévision émettent désormais des quantiles nativement — la sortie point-only devient l'exception. À retenir aussi : le leader GIFT-Eval en prévision ponctuelle ne fait que 18,5 M de paramètres, preuve que sur les séries temporelles l'innovation architecturale prime sur le scaling. C'est le contre-pied exact du monde LLM. TsfmTsfm
- **Assessing the Operational Viability of Foundation Models for Time Series Forecasting** — arXiv:2605.24381, 23 mai 2026 → https://arxiv.org/abs/2605.24381*Pertinent aujourd'hui :* l'étude ne regarde pas la précision agrégée mais la performance par régime opérationnel. Conclusion exploitable : les modèles de fondation excellent sur des structures périodiques transférables et sur les scénarios cold-start ou long-tail, tandis que les spécialistes supervisés gardent l'avantage sur les systèmes régis par des contraintes physiques strictes. Ils proposent un **Complexity Router** — on s'en sert dans le bloc 3. arXiv
- **Time Series Foundation Models: A Deep Dive into Strengths and Limitations** — aihorizonforecast, mai 2026 → https://aihorizonforecast.substack.com/p/time-series-foundation-models-a-deep*Pertinent aujourd'hui :* une lecture technique et sans hype des vraies limites des TSFM — lesquelles sont contournables, lesquelles ne le sont pas, lesquelles restent ouvertes. L'antidote au marketing des éditeurs ; à lire avant de vendre un TSFM à un client. Substack

### 2. Ingénierie & code (25 min)

**Objectif :** construire un harnais de **backtest rolling-origin** qui compare un TSFM zero-shot à des baselines statistiques fortes, sur une vraie série de charge électrique — et en sortir une décision argumentée.

**Consigne précise :**

1. Récupère une série de charge horaire (un pays via le package OPSD, cf. Livrable 2).
2. Baselines : `SeasonalNaive` (saison = 24) + `AutoETS` ou `MSTL` via `statsforecast`.
3. Challenger : `Chronos-Bolt` (zero-shot) via `chronos-forecasting`.
4. Protocole : fenêtre glissante, **≥ 6 origines**, horizon = 24 pas, refit baseline à chaque origine (le TSFM reste zero-shot).
5. Métriques via `utilsforecast` : **MASE**, **sMAPE**, **WAPE**, et **couverture de l'intervalle à 90 %** (le TSFM sort des quantiles — utilise-les).

**Stack :** Python 3.11, `statsforecast`, `utilsforecast`, `chronos-forecasting` (PyTorch), `pandas`. GPU optionnel (Chronos-Bolt tourne sur CPU pour une série).

**Critère de réussite — tu sais que c'est fini quand :** ton script imprime un tableau métriques × modèles reproductible (seed figée), **et** tu peux formuler en une phrase *sur quel horizon / quel régime* le TSFM bat ou perd contre AutoETS, en t'appuyant sur la couverture d'intervalle et pas seulement le MASE moyen. Si le TSFM gagne en MASE mais sa couverture 90 % réelle est à 70 %, tu dois savoir le dire — c'est ça le niveau senior.

### 3. System Design & MLOps (15 min)

**Mini-décision d'architecture :** tu dois mettre en production le *Complexity Router* du papier arXiv du jour. Le routeur aiguille **chaque série** soit vers un TSFM zero-shot mutualisé, soit vers un spécialiste supervisé dédié.

Produis un **raisonnement structuré (pas de code)** couvrant :

- **Features de routage** : sur quelles caractéristiques empiriques décides-tu ? (force/stabilité de la saisonnalité, longueur d'historique → cold-start, intermittence, présence de contraintes physiques, volatilité). Le papier montre que ce routage sélectif atteint une meilleure précision et un coût d'inférence nettement inférieur au déploiement d'un TSFM universel — quel est ton arbitrage précision/latence/coût GPU ? arXiv
- **Contrat de service** : SLA latence par branche, mode batch vs temps réel, et **fallback** si la branche choisie échoue (timeout TSFM → spécialiste, et inversement).
- **Le piège senior** : comment monitores-tu la **dérive de la décision de routage elle-même** ? (distribution des routes dans le temps, séries qui « basculent » de branche, et qui arbitre quand le router se trompe ?). Un router non monitoré devient une boîte noire de second ordre.

Écris ce raisonnement en 8-12 lignes ou en schéma textuel. Si tu ne sais pas justifier un seuil de bascule, c'est le signal que la feature de routage est mal choisie.

### 4. Lecture de référence

**Hyndman & Koehler (2006), *Another look at measures of forecast accuracy*** — le papier fondateur du **MASE**, accessible via le site de Rob Hyndman → https://robjhyndman.com/papers/mase.pdf

*Ce que tu en retires :* pourquoi le MASE est scale-free et robuste aux séries intermittentes là où le MAPE explose — exactement la métrique que tu manipules au bloc 2. Pour le contexte backtesting complet, le manuel de référence libre reste **FPP3** (Hyndman & Athanasopoulos, chap. « The forecaster's toolbox », section évaluation de la précision) → https://otexts.com/fpp3/
