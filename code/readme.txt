=======baseline=======================================
1. The path of dataset should be "./dataset/unary(or binary)/*", where "*" means the specific dataset
   GoogleNews-vectors-negative300.bin.gz should also be included in the path "dataset/"

2. unary_classify.py is used for unary template prediction (UT).
      
   binary_predict.py is used for binary template prediction (BT).
   
   combined_features.py is for the experiments BRI_cm.

   Use the command "python *.py" to run the codes, after setting the parameters in the file. 


3. The libraries, including 
   
   * gensim, 
   * re, 
   * scikit-learn, 
   * scipy, 
   * csv, 
   * pandas, 
   * numpy 
   
   should be installed before running the codes.
     
=========gcn============================================

**** Unary template prediction ****

1. The path of the dataset is "./dataset/*", where "*" means the specific dataset.
   GoogleNews-vectors-negative300.bin.gz should also be included in the path "dataset/"

2. unary_classify.py is used for the unary template prediction with the separate features.
   Use the command "python unary_classify.py -d wine -f embedding + other parameters" to run the codes. 
   
   combined_pca_and_we.py is used for the experiment that combines the output rules of word embedding and analogy space features.
   Use the command "python combined_pca_and we.py" to run the codes. 
   
   concatenated_feature_classify.py is used for the experiment that concatenates the word embedding and analogy space features as input.
   Use the command "python concatenated_feature_classify.py -d wine + other parameters" to run the codes. There is no need to set the feature type here.  
   
   all_data_used.py is used for checking whether our method can find rules that should have been in the ontology and were simply not there.
   Use the command "python all_data_used.py -d wine -f embedding + other parameters" to run the codes. 
   
3. The libraries, including 
   
   * dgl, 
   * pytorch, 
   * pandas, 
   * scikit-learn, 
   * csv, 
   * collections 
   
   should be installed before running the codes.
   
   
**** Binary template prediction ****

1. The path of the dataset is "./dataset/*/", where "*" means the specific dataset.
   GoogleNews-vectors-negative300.bin.gz should also be included in the path "dataset/"

2. binary_predict.py is used for the binary template prediction with the separate features.
   Use the command "python binary_predict.py -d wine -f embedding + other parameters" to run the codes. 
   
   combined_pca_and_we.py is used for the experiment that combines the output rules of word embedding and analogy space features.
   Use the command "python combined_pca_and we.py" to run the codes. 
   
   concatenated_feature_predict.py is used for the experiment that concatenates the word embedding and analogy space features as input.
   Use the command "python concatenated_feature_predict.py -d wine + other parameters" to run the codes. There is no need to set the feature type here.  
   
   all_data_used.py is used for checking whether our method can find rules that should have been in the ontology and were simply not there.
   Use the command "python all_data_used.py -d wine -f embedding + other parameters" to run the codes. 


3. The libraries, including 
   
   * dgl, 
   * pytorch, 
   * pandas, 
   * scikit-learn, 
   * csv, 
   * collections 
   
   should be installed before running the codes.
   



