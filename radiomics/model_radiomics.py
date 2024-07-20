import pandas as pd
import numpy as np
import os
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None
from sklearn.model_selection import GridSearchCV,cross_val_score
import matplotlib.pyplot as plt
from sklearn import model_selection
import matplotlib
import warnings
import random
warnings.filterwarnings('ignore')



def select_model_radiomics(modality:str, label:str):
    val_rate = 0.3
    if modality == 'all':
        print('Using all modality!')
    else:
        path = ('./' + modality + '_' + label + '.xlsx')
        assert os.path.exists(path), "dataset root:{} does not exist.".format(path)
        data = pd.read_excel(path, engine='openpyxl')
        df = data.iloc[:, 1:]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.max(), inplace=True)
        mri_id = data[['mri_id']].drop_duplicates(keep='first')
        random.seed(1)
        val_num = random.sample(range(len(mri_id)), k=int(len(mri_id) * val_rate))
        test_id = []
        train_id = []

        for i in range(len(mri_id)):
            if i in val_num:  # 如果该路径在采样的验证集样本中则存入验证集
                test_id.append(mri_id.iloc[i, :])
            else:  # 否则存入训练集
                train_id.append(mri_id.iloc[i, :])

        train_id = pd.DataFrame(train_id)
        test_id = pd.DataFrame(test_id)
        train = train_id.merge(df, on='mri_id', how='left').iloc[:, 1:]
        test = test_id.merge(df, on='mri_id', how='left').iloc[:, 1:]

        print('train shape:', train.shape)
        print('test shape:', test.shape)

        x_train = train.drop([label], axis=1)
        y_train = train[label]
        x_test = test.drop([label], axis=1)
        y_test = test[label]

        log_reg = LogisticRegression(solver="sag")
        model = log_reg.fit(x_train, y_train)
        predict_train = model.predict(x_test)
        print('Base_line Train AUC:', metrics.roc_auc_score(y_test, predict_train))

        models = [LogisticRegression(solver="sag"),
                  SVC(kernel="rbf", probability=True),
                  DecisionTreeClassifier(),
                  RandomForestClassifier(),
                  GradientBoostingClassifier(),
                  MLPClassifier(solver='lbfgs', max_iter=100),
                  XGBClassifier(n_estimators=100, objective='reg:squarederror'),
                  LGBMClassifier(n_estimators=50)]

        result = dict()
        for model in models:
            model_name = str(model).split('(')[0]
            scores = cross_val_score(model, X=x_train, y=y_train, verbose=0, cv=5,
                                     scoring=make_scorer(metrics.accuracy_score))
            result[model_name] = scores
            print(model_name + ' is finished')

        result = pd.DataFrame(result)
        result.index = ['cv' + str(x) for x in range(1, 6)]
        result.to_excel(label+modality+'_result.xlsx')

        matplotlib.rcParams['figure.figsize'] = (18.0, 18.0)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        i = 0
        plt.xticks(np.arange(0, 1.1, step=0.1))
        plt.yticks(np.arange(0, 1.1, step=0.1))
        result = dict()
        cv = model_selection.ShuffleSplit(n_splits=3, test_size=0.3, random_state=1)
        cs = ['red', 'orange', 'yellow', 'green', 'cyan',
              'blue', 'purple', 'pink', 'magenta', 'brown']
        c = 0
        for model in models:
            model_name = str(model).split('(')[0]
            for train, test in cv.split(x_train, y_train):
                probas_ = model.fit(x_train.iloc[train], y_train.iloc[train]).predict_proba(x_train.iloc[test])
                # Compute ROC curve and area the curve
                fpr, tpr, thresholds = roc_curve(y_train.iloc[test], probas_[:, 1])
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                # plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
                i += 1

            plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray', alpha=.8)
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            plt.plot(mean_fpr, mean_tpr, color=cs[c],
                     label=model_name + r' ROC AUC = %0.2f $\pm$ %0.2f' % (mean_auc, std_auc), lw=2, alpha=.8)
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            c += 1

            # 相关的设置被注释掉，可以恢复并查看画图效果。
            # plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')
            # plt.xlim([-0, 1])
            # plt.ylim([-0, 1])
            plt.xlabel('Specificity', fontsize='x-large')
            plt.ylabel('Sensitivity', fontsize='x-large')
            # plt.title('Receiver operating characteristic example', fontsize = 'x-large')
            plt.legend(loc="lower right", prop={"size": 22})

        plt.savefig((label+modality+'Train-ROC.jpg'), dpi=300)

        # test
        plt.clf()
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        lw = 2
        i = 0
        matplotlib.rcParams['figure.figsize'] = (18.0, 18.0)

        plt.xticks(np.arange(0, 1.1, step=0.1))
        plt.yticks(np.arange(0, 1.1, step=0.1))
        result = dict()
        cv = model_selection.ShuffleSplit(n_splits=3, test_size=0.3, random_state=1)
        cs = ['red', 'orange', 'yellow', 'green', 'cyan',
              'blue', 'purple', 'pink', 'magenta', 'brown']
        c = 0
        for model in models:
            model_name = str(model).split('(')[0]
            for train, test in cv.split(x_train, y_train):
                trainporbas_ = model.fit(x_train.iloc[train], y_train.iloc[train]).predict_proba(x_train.iloc[test])
                probas_ = model.predict_proba(x_test)  # 需要修改的是clf，即训练得到的model；以及测试集的X_test和y_test.
                fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
                fpr = fpr
                tpr = tpr
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
                # plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
            plt.plot(fpr, tpr, color=cs[c], alpha=.8, lw=lw, linestyle='-',
                     label=model_name + r' ROC AUC = %0.2f' % roc_auc)
            # plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',alpha=.6)
            plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray', alpha=.8)

            # plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')
            # plt.xlim([-0, 1])
            # plt.ylim([-0, 1])
            plt.xlabel('Specificity', fontsize='x-large')
            plt.ylabel('Sensitivity', fontsize='x-large')
            # plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right", prop={"size": 22})
            c += 1
            plt.savefig((label+modality+'Test-ROC.jpg'), dpi=300)


if __name__ == '__main__':
    # egfr del19 del21
    label = 'del19'
    # t1 t1c t2
    modality = 't1c'
    select_model_radiomics(modality, label)
