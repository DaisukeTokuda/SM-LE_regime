# Code structure

- tool/: Write processed files
- prod/: Reproduce each figure
- *.py, stat/, vs_atm/: Local library

# Workflow

## Detection of SM-LE regimes and calculation of their corresponding indicators

- tool/smle/est_regimes.py: Estimate parameters for all candidates
- tool/smle/select_regime.py: Select the best parameters and write the SM-LE indicators
- tool/smle/enstat_write.py: Calculate the ensemble mean and standard deviation of coupled simulations

## SM-LE example

- prod/fig1_smle_example.py

## Variability among offline and coupled simulations

- prod/fig2_boxplot_cpl_std.py
- prod/fig3_map_cpl_enstat.py
- prod/fig4_boxplot_ofl_std.py

## Comparison of SM-LE indicators if defined between offline and coupled simulations

- prod/fig5_scatter_cpl_vs_ofl.py
- prod/fig6_boxplot_similar_intra_vs_inter.py

## Impacts of atmospheric fields on the spatial distribution in SM-LE breakpoints

- tool/random_forest/model_save.py: Save the estimated model
- tool/random_forest/importance.py: Save the permutation importances
- tool/random_forest/score.py: Save the prediction score
- prod/fig7_boxplot_bpdef_eval.py
- prod/fig8_scatter_bpdef_importance.py

## Impacts of atmospheric fields on the values of SM-LE indicators

- tool/lasso/model_save.py: Save the estimated model
- tool/lasso/coef.py: Get the coefficient from each estimated model
- prod/fig9_scatter_lasso_coef.py

# Note

- Some original libraries are not included (e.g., searching for available files on a local server)