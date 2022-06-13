# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Pre-processing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold

# Performance measures
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer

from sklearn.compose import ColumnTransformer


seed = 42
scorer = make_scorer(precision_score, average='micro') # Teste outras

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
gscv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

numerical_features = [
    'Idade', 
    'Filhos',
    'Horas_Trabalhadas_Semana',
    'Tempo_Empresa_anos',
    'Trabalho_Entre_Ferias_meses',
    'Tempo_Pe_horas',
    'Tempo_Sentado_horas',
    'Tempo_Corcoras_horas',
    'Cigarros_dia',
    'Horas_Dormidas_dia',
    'Pausas_Programadas_minutos',
    'Superiores_Desconfortavel_horas',
    'Inferiores_Desconfortavel_horas',
    'Tronco_Curvado_horas',
    'Tronco_Torcido_horas',
    'Maos_Dedos_horas',
    'Carga_6_horas',
    'Carga_15_horas',
    'Carga_25_horas',
    'Movimentos_repetitivos_horas',
    'Movimentos_Rapidos_horas',
    'Ferramentas_Maos_horas',
    'Ferramentas_Corpo_horas',
    'Horas_Carregando_Carga'
]

categorical_features = [
    'IMC_categoria'    
]

first_preprocessing = ColumnTransformer(
    transformers = [
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

second_numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', KBinsDiscretizer(n_bins = 5))
])
second_preprocessing = ColumnTransformer(
    transformers = [
        ('num', second_numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])


algorithms = {
    'kNN':  GridSearchCV(
        estimator = Pipeline(steps = [
            ('preprocessing', first_preprocessing)
            ,('knn', KNeighborsClassifier())
        ]), 
        param_grid={
            'knn__n_neighbors': [1, 3, 5],
            'knn__p': [1, 2],
        },
        scoring=scorer,
        cv=gscv)
    
    ,'kNN_2nd':  GridSearchCV(
        estimator = Pipeline(steps = [
            ('preprocessing', second_preprocessing)
            ,('knn', KNeighborsClassifier())
        ]), 
        param_grid={
            'knn__n_neighbors': [1, 3, 5],
            'knn__p': [1, 2],
        },
        scoring=scorer,
        cv=gscv)
    ,

    'nb_1st':  GridSearchCV(
        estimator = Pipeline(steps = [
            ('preprocessing', first_preprocessing)
            ,('knn', GaussianNB())
        ]), 
        param_grid = None,
        scoring=scorer,
        cv=gscv)
    
    ,'nb_2nd':  GridSearchCV(
        estimator = Pipeline(steps = [
            ('preprocessing', second_preprocessing)
            ,('knn', GaussianNB())
        ]), 
        param_grid = None,
        scoring=scorer,
        cv=gscv)
    ,
}





