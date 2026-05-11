# Causal-Event Graphs for Context-Grounded Explainability in Supply-Chain Sales Forecasting

This repository contains the implementation of a framework for context-grounded explanation of supply-chain sales forecasting. The study combines predictive modeling, feature attribution, time-series causal discovery, Event Registry-based event graph construction, causal-event graph integration, and large-language-model-based explanation generation.

The main goal is not only to predict sales, but to explain selected predictions by linking model-important features to estimated causal relations and to real-world events extracted from external news sources.

## Overview of the process

The workflow follows a multi-stage pipeline:

1. **Data preparation and feature engineering**  
   The SupplyGraph benchmark dataset is enriched with contextual information from macroeconomic, financial, holiday, weather, and event-related data sources. Operational and derived supply-chain features are created, including lagged sales variables, production-to-sales ratios, delivery-to-sales ratios, factory issue rates, and inventory proxies.

2. **Predictive modeling**  
   Several ensemble learning models are trained for sales forecasting, including Random Forest, XGBoost, LightGBM, and CatBoost. Model performance is evaluated using RMSE, MAE, MAPE, MASE, and the coefficient of determination.

3. **Feature attribution**  
   Local feature attribution is applied to selected prediction instances to identify the features that contributed most strongly to the model prediction. These model-important features become the anchors for later event retrieval and explanation.

4. **Causal discovery**  
   Time-series causal discovery is applied to the structured supply-chain and contextual variables to estimate potential structural relations among them. The resulting causal graph is used as a data-driven causal layer for explanation, not as definitive proof of causality.

5. **Event Registry-based event graph construction**  
   Model-important features are translated into semantic search queries and used to retrieve relevant Event Registry articles published before the prediction date. A large language model extracts grounded event-feature relations from the retrieved articles. These relations are checked for textual evidence and scope consistency before being retained in the event graph.

6. **LLM-as-Judge verification**  
   Candidate event-feature relations are verified using an LLM-as-Judge procedure. The judge checks whether the supporting evidence appears in the article, whether the article supports the proposed cause-effect relation, and whether the event matches the feature definition at the appropriate scope.

7. **Causal-event graph integration**  
   The causal graph and event graph are merged into a unified causal-event graph. This graph connects model-important features, estimated causal relations, news-derived event entities, and semantic attribute categories into one explanation object.

8. **Explanation generation**  
   The final causal-event graph is converted into a plain-language explanation for supply-chain decision-makers. The generated explanation summarizes relevant events, explains the selected causal relations, and highlights potential risks or opportunities.

## Data sources

The study uses publicly accessible and externally available data sources, including:

- SupplyGraph benchmark dataset
- Trading Economics macroeconomic and financial indicators
- Bangladesh national holiday data
- Bangladesh weather data
- Event Registry news articles

Full article texts retrieved from Event Registry may be subject to source-specific copyright or access restrictions and should not be redistributed unless permitted by the original source.

## Environment setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

## API keys and configuration

Some parts of the pipeline require external API access. Store credentials outside the codebase, for example in environment variables or a local configuration file that is not committed to version control.

Typical required credentials include:

```bash
OPENROUTER_API_KEY=your_openrouter_key
EVENT_REGISTRY_API_KEY=your_event_registry_key
```

If a local configuration file is used, add it to `.gitignore` before making the repository public.

## Running the workflow

The notebooks should be executed in the logical order of the pipeline:

1. Prepare and merge supply-chain and contextual data.
2. Train and evaluate forecasting models.
3. Compute local feature attribution for selected prediction instances.
4. Estimate the causal graph.
5. Retrieve Event Registry articles and construct the event graph.
6. Verify extracted event-feature relations.
7. Merge the causal graph and event graph.
8. Generate visualizations and final explanations.

The exact file paths may need to be adjusted depending on the local directory structure and where datasets, saved models, and intermediate outputs are stored.

## Main outputs

The pipeline can produce:

- processed feature tables;
- trained forecasting model outputs;
- local feature-importance results;
- estimated causal graphs;
- Event Registry-derived event graphs;
- LLM-as-Judge audit logs;
- unified causal-event graphs;
- static and dynamic graph visualizations;
- plain-language supply-chain explanations;


## Reproducibility notes

The workflow depends on external data availability, API responses, and large-language-model outputs. For reproducibility, it is recommended to save:

- retrieved article metadata;
- extracted event-feature relations;
- LLM-as-Judge decisions;
- selected prediction-instance identifiers;
- model parameters;
- train-test split information;
- random seeds where applicable.

Because large-language-model outputs may vary between runs, saved intermediate JSON and CSV files should be retained when preparing final manuscript results.

## Citation

If this repository is used or extended, please cite the associated study:

**Causal-Event Graphs for Context-Grounded Explainability in Supply-Chain Sales Forecasting**
