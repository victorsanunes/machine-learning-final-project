# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# Pre-processing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA


# Performance measures
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer

from sklearn.compose import ColumnTransformer


seed = 42
precision_scorer = make_scorer(precision_score, average='weighted') # Teste outras
recall_scorer = make_scorer(recall_score, average='weighted') # Teste outras
f1_scorer = make_scorer(f1_score, average='weighted') # Teste outras
accuracy_scorer = make_scorer(accuracy_score) # Teste outras

scorers = {
    'precision': precision_scorer
    ,'recall': recall_scorer
    ,'f1': f1_scorer
    ,'accuracy': accuracy_scorer
}


cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
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

['Genero', 
'Idade', 
'Escolaridade', 
'Estado_Civil',
 'Filhos',
 'Atividade_fisica_semana',
  'Cigarros_dia',
   'Horas_Dormidas_dia',
       'Se_Capacitando', 'Outro_emprego', 'Ambiente_Trabalho',
       'Categoria_Profissional', 'Tempo_Empresa_anos', 'Tempo_Funcao_anos',
       'Horas_Trabalhadas_Semana', 'Pausas_Programadas_minutos',
       'Trabalho_Entre_Ferias_meses', 'Recebeu_Treinamento',
       'Como_foi_treinado', 'Trabalho_Chato', 'Trabalho_Organizado',
       'Prazos_Apertados', 'Satisfação_Gerencia', 'Satisfação_Dificuldades',
       'Tempo_Pe_horas', 'Tempo_Sentado_horas', 'Tempo_Corcoras_horas',
       'Superiores_Desconfortavel_horas', 'Inferiores_Desconfortavel_horas',
       'Tronco_Curvado_horas', 'Tronco_Torcido_horas', 'Maos_Dedos_horas',
       'Movimentos_repetitivos_horas', 'Movimentos_Rapidos_horas',
       'Ferramentas_Maos_horas', 'Ferramentas_Corpo_horas',
       'dados.Costa_superior', 'dados.Costa_media', 'dados.Costa_inferior',
       'bmi', 'carry_weight']

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

def build_algorithms(scorer = accuracy_score):

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
            # scoring = scorer,
            # cv = cv
            )
        ,'tree':  GridSearchCV(
            Pipeline([
                ('preprocessing', first_preprocessing),
                ('tree', DecisionTreeClassifier(random_state=seed))]), 
            param_grid={
                'tree__max_depth': [5, 10, 20],
                'tree__criterion': ['entropy', 'gini'],
            },
            )

    ,'svmlinear': GridSearchCV(
        Pipeline([
            ('preprocessing', first_preprocessing),
            ('pca', PCA()),
            ('svm', SVC(kernel='linear', random_state=seed))]), 
        param_grid={
            'pca__n_components': [2, 5, 10],
            'svm__C': [1.0, 2.0],
        },
        )
    ,'ann': GridSearchCV(
        Pipeline([
            ('preprocessing', first_preprocessing),
            ('ann', MLPClassifier(random_state = 1, max_iter=500))]), 
        param_grid = {
            #'activation': ['tanh', 'logistic'],
            #'hidden_layer_sizes': [(10,)],
            #'solver': ['adam'],
        },
        )

        
        # ,'kNN_2nd':  GridSearchCV(
        #     estimator = Pipeline(steps = [
        #         ('preprocessing', second_preprocessing)
        #         ,('knn', KNeighborsClassifier())
        #     ]), 
        #     param_grid={
        #         'knn__n_neighbors': [1, 3, 5],
        #         'knn__p': [1, 2],
        #     },
        #     scoring = f1_scorer,
        #     cv = gscv)
        # ,

        # 'nb_1st':  GridSearchCV(
        #     estimator = Pipeline(steps = [
        #         ('preprocessing', first_preprocessing)
        #         ,('nb', GaussianNB())
        #     ]), 
        #     param_grid = {'nb__var_smoothing': [1e-9]},
        #     scoring = f1_scorer,
        #     cv = gscv)
        
        # ,'nb_2nd':  GridSearchCV(
        #     estimator = Pipeline(steps = [
        #         ('preprocessing', second_preprocessing)
        #         ,('nb', GaussianNB())
        #     ]), 
        #     param_grid = {'nb__var_smoothing': [1e-9]},
        #     scoring = f1_scorer,
        #     cv = gscv)
        # ,
    }

    return algorithms





