## Coursework Resources

To assist in your implementation, you are provided with this structured repository containing the necessary data and code scaffolding. The `data` directory contains a pre-partitioned train-validation-test split of our generated dataset, ensuring a consistent evaluation environment for all students. The `src` directory contains three primary Jupyter Notebooks designed to guide your workflow:

*   `logistic_regression.ipynb`: This file provides a template including data loading and a standard `scikit-learn` training pipeline. It also includes an accuracy evaluation cell and descriptions of the additional evaluations you need to do, such as implementing your fairness metric and visualizing the ROC curve.
*   `decision_tree.ipynb` and `neural_network.ipynb`: These notebooks are intentionally sparse. While they include data loading logic, the remaining sections consist of high-level descriptions for the cells you must develop to complete the comparative analysis required for the project.
*   Additionally, the `tex` directory contains a LaTeX template that serves as an example of the expected final submission format.

The repository follows the structure outlined below:

```bash
automated_interview_screening/
|-- data/
|   |-- train.csv
|   |-- val.csv
|   \-- test.csv
|-- src/
|   |-- logistic_regression.ipynb
|   |-- decision_tree.ipynb
|   \-- neural_network.ipynb
\-- tex/
    \-- submission_template.tex
```
