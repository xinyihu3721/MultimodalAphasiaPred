# SHAP Value Computation

## Description

This Jupyter Notebook, `compute_SHAP.ipynb`, is designed to compute **SHAP** values for predictors/features by analyzing test RMSE results from models with various predictor combinations. SHAP values offer insights into feature importance and aid in interpreting the contributions of predictors to the model's performance and predictions.
SHAP values for each predictor were calculated for SVR and RF model families separately, based on how much, on average, each predictor contributes to increasing or decreasing the prediction accuracy, as measured by average RMSE across folds, when included with other predictors. Specifically, the SHAP value for each predictor, $i$, is computed using the following formula:

$$\phi_i(F, v) = \frac{1}{|F|!} \sum_{S \subseteq F \setminus \{i\}} |S|! (|F| - |S| - 1)! \big[\nu(S) - \nu(S \cup \{i\}) \big],$$

$$
= \frac{1}{|F|} \sum_{L = 0}^{|F|-1} \frac{1}{\binom{|F|-1}{L}} \sum_{S \subseteq F \setminus \{i\} \text{ and} |S| = L}  \big[\nu(S) - \nu(S \cup \{i\}) \big],
$$

where $\phi_i$ is the SHAP function, $F$ is the set of all predictors (7 in total), $S$ is any subset of $F$ that does not include predictor $i$, $\nu(S)$ is the so-called characteristic function given by the average RMSE score for the model using the subset of predictors $S$, and $\big[\nu(S) - \nu(S \cup \{i\}) \big]$ is the **marginal contribution** of predictor $i$ to the subset $S$. 

We note that a positive value of $\big[\nu(S) - \nu(S \cup \{i\}) \big]$ would imply that RMSE decreases (prediction accuracy improves) when predictor $i$ is used in conjunction with the predictors in $S$ than otherwise. For $S = \{\}$, the empty set, the best RMSE predictor for any outer test-training fold is the constant predictor equal to the mean value of the outer training fold.

SHAP values were computed for each predictor using the following Python script.


**The notebook includes:**
- Computing SHAP values.
- Visualizing Marginal Contribution Values
---

## Requirements

To run this notebook, ensure the following Python libraries are installed:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `math`

Install these packages using `pip`:

```bash
pip install numpy pandas matplotlib seaborn
```
---
## Usage
1. **Load the Notebook:** Open compute_SHAP.ipynb in Jupyter Notebook, JupyterLab, or any compatible environment.
2. **Prepare the Data:** Replace the placeholder data loading section with your own dataset or ensure the existing data is compatible with your project. See the following subsubsection for input data requirements.
3. **Run the Notebook:** Execute the cells in order to:
   - Compute SHAP values for feature importance analysis.
   - Visualize results through SHAP plots.
### Input Data Requirements
   The imported CSV file should contain the following columns:
   - **Data Combination**: Lists the combinations of predictors (e.g., `PSW`, `RS-trans`, `PSG`, etc.) used in the model.
   - **Mean-RMSE**: Displays the corresponding mean RMSE values for each data combination.
   The structure of the CSV file should resemble the table below:
      
      | Data Combination       | Mean-RMSE |
      |-------------------------|-----------|
      | PSW                    | 19.539402 |
      | RS-trans               | 23.916031 |
      | PSG                    | 18.806939 |
      | DM                     | 23.977912 |
      | FA                     | 22.115956 |
      | ...                    | ...       |
      | DM LV PSG RS-trans     | 18.954037 |
      | DM FA PSG RS           | 16.206378 |
      | LV PSG RS RS-trans     | 16.381748 |
      | FA LV PSW              | 19.440104 |
      | Null                   | 22.256337 |

  Ensure the column names and data format match this structure for the notebook to process the data correctly. The Null model, representing no predictors, should be named as `Null`.

