# Test Task from KazanExpress
### by Liliya Kazykhanova
Marketplace Product categorization
***
**Business Task**: Marketplace's client-seller may make a mistake when filling product category field. It can be a reason of low potential profit. Task: to create an instrument to check these cases at the stage when client-seller just fill out data into marketplace system.

**Technical Task**: using the data to build a model that predicts the product category: text, image, other numerical and categorical features

**Type of ML task**:
* Classification: multiclass
    - Common Machine Learning Algorithms: Naive Bayes, LinearSVC, Ensemble (Random Forest)
    - Deep Learning: fine-tuned BERT model from HuggingFace

**Metrics:** F1-weighted score (micro)

**Data:** Train and test parquet file + 2 folders with train and test sets of images
- Train dataset contains 6 features, 1 target value - category_id and category_name column is text representation of category_id
- Test dataset - 6 features

*Training on CPU classical ML models*
*Training on GPU Text classification Neural Network*

**NOTES**:
* I run it based on Kaggle resources (without accelerator)
* Git has limit: 100 MB. That's why some of the files are not pushed (datasets, best_model.pt, train_data.pkl, valid_data.pkl)
* This project (input and output data) is available on Kaggle by the links: `https://www.kaggle.com/code/liliyak/solution-ke/notebook` and `https://www.kaggle.com/liliyak/ke-text-transformers`
***

### **CONTENT**
[Introduction](https://github.com/LiliyaKazykhanova/Test-Cases-for-KazanExpress/blob/main/README.md#Introduction)

[Checking dataset: outliers, duplicates](https://github.com/LiliyaKazykhanova/Test-Cases-for-KazanExpress/blob/main/README.md#Checking-dataset-:-outliers-,-duplicates)

[Feature engineering](https://github.com/LiliyaKazykhanova/Test-Cases-for-KazanExpress/blob/main/README.md#Feature-engineering)

[Models and Results](https://github.com/LiliyaKazykhanova/Test-Cases-for-KazanExpress/blob/main/README.md#Models-and-Results)

[Further research steps](https://github.com/LiliyaKazykhanova/Test-Cases-for-KazanExpress/blob/main/README.md#Further-research-steps)
***

#### **INTRODUCTION**
I joined train and test data into one main dataset for the correct feature processing.
- dataset shape: 107980 rows, 8 columns (+ 1 added 'sample' column in order to distinguish train and test data)
- target value: category_id

*category_name - text representation of category tree*
***

#### **CHECKING DATASET: outliers, duplicates**
- Number of found duplicates: 0
- Columns with missing values: None (category_id and category_name - are target values)
***

#### **FEATURE ENGINEERING**
- Rating:
    * converted to numerical type and rounded
    * ~76% data has 5 marks
- Sale:
    * converted to 0/1
    * ~1% data is flaged as 'in sale'
- Category Name:
    * getting 6 levels of category tree
        - ~35% products belong to 'Товары для дома' category
        - ~29% products belong to 'Электроника' category
        - ~19% products belong to 'Одежда' category
        - ~16% products belong to 'Хобби и творчество' category
        - ~2% products belong to 'Обувь' category
- Text fields:
    * getting 6 new columns:
        - title
        - description
        - attributes
        - custom_characteristics
        - defined_characteristics
        - filters
    * More than 50% data of custom/defined characteristics and filters columns are nan value (null). By rules, column with more than 30-40% of missing values should be dropped.
    * 'title' column consits of product name (key words). It's necessary to preprocess text data for using in algorithm:
        - tokenization — convert sentences to words
        - removing unnecessary punctuation, tags
        - removing stop words (frequent words which have not any semantic sense)
        - lemmatization — reducing words to their root word
        - vectorization — numerically representation of text (tf-idf)
***

#### **MODELS and RESULTS**
- Total feature number: 1 (title_lem column - it is processed 'title' column)
- Total number of predicted classes: 870 (after dropping small categories (less than 2))

**CLASSICAL models**

| Model | Params | F_1 score |
| :-: | :-: | :-: |
| **Naive Bayes** | by default | 0.67 |
| <font color='LightSeaGreen'>**LinearSVC**</font> | by default | 0.94 |
| <font color='LightSeaGreen'>**Random Forest**</font> | by default (n_estimators=15) | 0.97 |
| **GridSearchCV<br>Random Forest** | n_estimators=100<br>max_depth=900<br>max_features=400<br>min_samples_leaf=3 |
|  |  |  |

RESULTS of F1-weighted score on valid: 83% (final model - LinearSVC)

**DEEP LEARNING**
- Multilingual Pretrained Model: BertModel
- Get word embeddings: XLMRobertaTokenizer

RESULTS of F1-weighted score on valid after 3 epochs: 72%
***

#### **FURTHER RESEARCH**
As a further research step I can suggest next points:
- Text preprocessing:
    * using another type of algorithm for vectorization (word2vec - Skip-Gram)
- Build a multimodal model based on text and image as features to predict product category:
    * Text-model + CNN model for image data (VGG, ResNet, InceptionV3)
