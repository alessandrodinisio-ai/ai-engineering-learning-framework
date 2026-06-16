# Fairness Criteria — Group, Individual, Counterfactual

> Three families form the backbone of the fairness literature. Group fairness: demographic parity, equalized odds, conditional use accuracy equality — rates equal on average across protected groups. Individual fairness (Dwork et al. 2012): similar individuals receive similar decisions; a Lipschitz condition on the decision mapping. Counterfactual fairness (Kusner et al. 2017): a decision is fair to an individual if it remains unchanged when the sensitive attribute is counterfactually altered. 2024 theoretical result (NeurIPS 2024): there is an inherent "counterfactual vs accuracy" tradeoff; a model-agnostic method can convert an "optimal but unfair" predictor into a counterfactually fair one with bounded accuracy loss. Backtracking counterfactuals (arXiv:2401.13935, January 2024): a new paradigm that avoids interventions on legally protected attributes. Philosophical reconciliation (ICLR Blogposts 2024): with a causal graph, satisfying certain group fairness metrics implies counterfactual fairness.

**Type:** Learn
**Languages:** Python (standard library, three-criteria comparison)
**Prerequisites:** Phase 18 · 20 (bias), Phase 02 (classical ML)
**Time:** ~60 minutes

## Learning Objectives

- Name three group fairness criteria (demographic parity, equalized odds, conditional use accuracy equality) and one impossibility result.
- Describe individual fairness via the Lipschitz formulation of Dwork et al. 2012.
- Describe counterfactual fairness and its dependence on a causal graph.
- Explain backtracking counterfactuals, and why they sidestep the "intervening on protected attributes" problem.

## The Problem

Lesson 20 covered measuring bias. Lesson 21 covers defining the fairness standard that measurement serves. Three families offer structurally different criteria — a model can be group-fair but individually unfair, or counterfactually fair but group-unfair. Choosing a criterion is a policy decision; no criterion is universally optimal.

## The Concept

### Group Fairness

- **Demographic parity.** For all groups P(Y=1 | A=a) = P(Y=1 | A=a'). Equal acceptance rates.
- **Equalized odds.** P(Y=1 | Y*=y, A=a) = P(Y=1 | Y*=y, A=a'). Equal TPR and FPR across groups.
- **Conditional use accuracy equality.** P(Y*=y | Y=y, A=a) = P(Y*=y | Y=y, A=a'). Equal predictive values across groups.

Impossibility (Chouldechova, Kleinberg-Mullainathan-Raghavan 2017): when base rates are unequal, these three cannot hold simultaneously.

### Individual Fairness

Dwork et al. 2012. A decision mapping f is individually fair if it satisfies |f(x) - f(x')| <= L * d(x, x') (for some Lipschitz constant L) with respect to a task-specific similarity metric d. Similar individuals receive similar decisions.

Defining d is required. This is a policy question, not a statistical one.

### Counterfactual Fairness

Kusner et al. 2017. A decision is counterfactually fair to individual i if, under a causal model of the population, the decision remains unchanged when individual i's sensitive attribute is counterfactually altered.

Requires a causal DAG. That DAG is a modeling choice. The legitimacy of counterfactual fairness depends entirely on the legitimacy of that DAG.

### Counterfactual vs Accuracy Tradeoff

NeurIPS 2024 theory: there is an inherent tradeoff between counterfactual fairness and predictive accuracy. A model-agnostic method can convert an "optimal but unfair" predictor into a counterfactually fair one with bounded accuracy cost. That accuracy cost depends on the magnitude of the sensitive attribute coefficients in the "optimal unfair predictor."

### Backtracking Counterfactuals

arXiv:2401.13935 (January 2024). Traditional counterfactuals require intervening on the sensitive attribute — "if this person were a different gender, would the decision change?" This is legally problematic: anti-discrimination law does not permit interventions on protected attributes.

Backtracking counterfactuals reverse the direction: instead of intervening on the attribute, ask which combination of the individual's actual features would have produced that counterfactual outcome. This sidesteps the legal objection.

### Philosophical Reconciliation

ICLR Blogposts 2024. With a causal graph in hand, satisfying certain group fairness metrics implies counterfactual fairness. The three families are not orthogonal; they are different facets of the same underlying causal structure.

This does not resolve the impossibility theorem (unequal base rates still prevent group criteria from holding simultaneously). But it shows that the apparent opposition between "group" and "individual/counterfactual" is partly an artifact of not making the causal model explicit.

### Position in Phase 18

Lesson 20 is bias measurement. Lesson 21 is fairness definitions. Lesson 22 is privacy (differential privacy). Lesson 23 is watermarking. These are allocation-adjacent lessons, complementing the deception-adjacent content of Lessons 7–11.

## Build It

`code/main.py` creates a toy binary classification dataset with a sensitive attribute and unequal base rates. Computes demographic parity, equalized odds, and conditional use accuracy equality on a simple classifier. Observes the three metrics fighting each other. Applies a reweighting for demographic parity and observes its cost on the other two metrics.

## Use It

This lesson produces `outputs/skill-fairness-criterion.md`. Given a fairness claim or policy, it identifies which criterion is being claimed, whether the model can satisfy the remaining criteria under the stated unequal base rates, and which causal DAG the claim depends on.

## Exercises

1. Run `code/main.py`. Report the three group metrics on the default data. Apply the demographic parity reweighting and report again.

2. Implement the individual fairness metric from Dwork et al. 2012 using L2 on non-sensitive features. Report how many pairs violate the Lipschitz condition at constant L=1.

3. Read Kusner et al. 2017. Construct a simple two-feature causal DAG for resume scoring, and identify the counterfactual fairness conditions it implies.

4. The 2024 backtracking counterfactuals paper avoids intervening on protected attributes. Describe a scenario where this matters for legal compliance.

5. The ICLR 2024 reconciliation claims group fairness and counterfactual fairness are facets of the same structure. In `code/main.py`, pick two criteria and state the causal assumption that would make them equivalent.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|---------------------|
| Demographic parity | "Equal rates" | P(Y=1 \| A=a) equal across groups |
| Equalized odds | "Equal TPR/FPR" | True positive and false positive rates equal across groups |
| Conditional use accuracy | "Equal PPV/NPV" | Predictive values equal across groups |
| Individual fairness | "Lipschitz condition" | Similar individuals receive similar decisions |
| Counterfactual fairness | "Invariant to causal change" | Decision unchanged when attribute is counterfactually altered |
| Backtracking counterfactual | "Explain with actual values" | Counterfactual reasoning backward from outcome rather than forward from attribute |
| Impossibility theorem | "The three conflict" | Chouldechova / KMR 2017: group criteria are mutually exclusive when base rates differ |

## Further Reading

- [Dwork et al. — Fairness through Awareness (arXiv:1104.3913)](https://arxiv.org/abs/1104.3913) — individual fairness
- [Kusner, Loftus, Russell, Silva — Counterfactual Fairness (arXiv:1703.06856)](https://arxiv.org/abs/1703.06856) — counterfactual fairness
- [Chouldechova — Fair prediction with disparate impact (arXiv:1703.00056)](https://arxiv.org/abs/1703.00056) — impossibility
- [Backtracking Counterfactuals (arXiv:2401.13935)](https://arxiv.org/abs/2401.13935) — new paradigm for protected attribute intervention
