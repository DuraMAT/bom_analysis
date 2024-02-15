# bom_analysis
This repository is published together with the article entitled "Analyzing the Impact of Design Factors on Solar Module Thermomechanical Durability Using Interpretable Machine Learning Techniques". The data set used in this study is available at [DuraMat Datahub](https://datahub.duramat.org/dataset/bom_thermal_cycling_degradation).

`public_bom_tc_modeling.ipynb` records the machine learning modeling and SHAP interpretation. \
`public_bom_tc_stats_validation.ipynb` records the post-hoc statistical validation. \
`bom_tc_pkg.py` provides the functions used in the two notebooks.

`model_publish` contains the pretrained weights of random forest model. `rf_train.pkl` is the weight trained on the training set and to be tested on the testing set. `rf_shap.pkl` shares the same model structure with `rf_train.pkl` but trained on the whole data set, which was used for SHAP interpretation.