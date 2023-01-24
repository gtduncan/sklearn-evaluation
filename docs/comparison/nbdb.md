---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Querying notebooks with SQL

*Added in sklearn-evaluation version 0.6*. Questions? [Join our community!](https://ploomber.io/community)

+++

`NotebookDatabase` indexes outputs from a collection of notebooks in a SQLite database so you can query them. Any tagged cells will be captured and indexed by the database.

Requirements:

```sh
pip install scikit-learn sklearn-evaluation ploomber ploomber-engine jupysql
```

```{code-cell} ipython3
from pathlib import Path

from ploomber.products import File
from ploomber_engine import execute_notebook

# to produce parameter grid
from sklearn.model_selection import ParameterGrid

# to create SQLite database
from sklearn_evaluation import NotebookDatabase
from sklearn_evaluation import SQLiteTracker
```

## Code

`NotebookDatabase` indexes the output of tagged cells. In this example, we're using `.ipynb` notebooks (and tag cells using `# %% tags=["some-tag"]`), [see here](https://docs.ploomber.io/en/latest/user-guide/faq_index.html#parameterizing-notebooks) to learn how to tag cells in `.ipynb` files.

<b>data.ipynb</b>

```{code-cell} ipython3
from sklearn import datasets
```
```{code-cell} ipython3
ca_housing = datasets.fetch_california_housing(as_frame=True)
df = ca_housing['frame']
df.to_csv('data.csv', index=False)
```

<b>model.ipynb</b>
```{code-cell} ipython3
# %% tags=["parameters"]
model = None
params = None
```

```{code-cell} ipython3
import importlib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
```

```{code-cell} ipython3
df = pd.read_csv('data.csv')
```

```{code-cell} ipython3
X = df.drop('MedHouseVal', axis='columns')
y = df.MedHouseVal
```

```{code-cell} ipython3
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.33,
                                                    random_state=0)
```

```{code-cell} ipython3
mod, _, attr = model.rpartition('.')
reg = getattr(importlib.import_module(mod), attr)(**params)
reg.fit(X_train, y_train)
print(model)
```

```{code-cell} ipython3
print(reg.get_params())
```

```{code-cell} ipython3
y_pred = reg.predict(X_test)
mean_squared_error(y_test, y_pred)
```

## Executing notebooks

Using <a href='https://ploomber-engine.readthedocs.io/en/latest/quick-start.html'>ploomber-engine</a>. Each experiment will create an output `.ipynb` file.

```{code-cell} ipython3

experiments = {
    'sklearn.tree.DecisionTreeRegressor': ParameterGrid(dict(criterion=['squared_error', 'friedman_mse'], splitter=['best', 'random'], max_depth=[3, 5])),
    'sklearn.linear_model.Lasso': ParameterGrid(dict(alpha=[1.0, 2.0, 3.0], fit_intercept=[True, False])),
    'sklearn.linear_model.Ridge':ParameterGrid(dict(alpha=[1.0, 2.0, 3.0], fit_intercept=[True, False])), 
    'sklearn.linear_model.ElasticNet': ParameterGrid(dict(alpha=[1.0, 2.0, 3.0], fit_intercept=[True, False])), 
}

# execute data.ipynb

execute_notebook(Path('data.ipynb'), File('output.ipynb'))

# generate one task per set of parameter

for model, grid in experiments.items():
    for i, params in enumerate(grid):
        name = f'{model}-{i}'
        task = execute_notebook(Path('model.ipynb'), File(f'{name}.ipynb'), parameters=dict(model=model, params=params))

```

## Indexing notebooks

```{code-cell} ipython3
:tags: ["hide-output"]

# initialize db with notebooks in the outputs directory
db = NotebookDatabase('nb.db', 'output/models/*.ipynb')

# Note: pass update=True if you want to update the database if
# the output notebook changes
db.index(verbose=True, update=False);
```

*Note: the `update` argument in `index()` was added in sklearn-evaluation version `0.7`*

+++

## Querying notebooks

`NotebookDatabase` uses SQLite. Here we use [SQLiteTracker](https://jupysql.readthedocs.io/en/latest/intro.html) to query our experiments.

```{code-cell} ipython3
tracker = SQLiteTracker("nb.db")
```

### Best performing models

```{code-cell} ipython3
df = tracker.query(
    """
SELECT
    path,
    json_extract(c, '$.model') AS model,
    json_extract(c, '$.mse') AS mse
FROM nbs
ORDER BY 3 ASC
LIMIT 3
"""
)

df
```

*Note:* If using SQLite 3.38.0 (which ships with Python >=3.10) or higher, you can use the shorter `->>` operator:

```sql
SELECT
    path,
    c ->> '$.model' AS model,
    c ->> '$.mse' AS mse
FROM nbs
ORDER BY 3 ASC
LIMIT 3
```

See SQLite's [documentation](https://www.sqlite.org/json1.html#jptr) for details.

+++

### Average error by model type

```{code-cell} ipython3
df = tracker.query(
    """SELECT
    json_extract(c, '$.model') AS model,
    AVG(json_extract(c, '$.mse')) AS avg_mse
FROM nbs
GROUP BY 1
ORDER BY 2 ASC""")

df
```

### DecisionTree by performance

```{code-cell} ipython3
df = tracker.query(
    """SELECT
    json_extract(c, '$.model') AS model,
    json_extract(c, '$.mse') AS mse,
    json_extract(c, '$.params.max_depth') AS max_depth,
    json_extract(c, '$.params.criterion') AS criterion,
    json_extract(c, '$.params.splitter') AS splitter
FROM nbs
WHERE json_extract(c, '$.model') = 'sklearn.tree.DecisionTreeRegressor'
ORDER BY mse ASC
LIMIT 5""")

df
```
