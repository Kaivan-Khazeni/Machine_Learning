import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/34836777/print-complete-key-path-for-all-the-values-of-a-python-nested-dictionary
def dict_path(my_dict, path=None):
    if path is None:
        path = []
    for k, v in my_dict.items():
        newpath = path + [k]
        if isinstance(v, dict):
            for u in dict_path(v, newpath):
                yield u
        else:
            yield newpath, v

def find_total_entropy(max_labels,variant,S):
  if variant == 1:
    #use entropy
    type_name = 'y'
    error = 0
    label_names = S[type_name].unique()

    #print(S)
    p = S.loc[S['y'] == 1, 'D_t'].sum()
    _p = S.loc[S['y'] == -1, 'D_t'].sum()
    error = -1*p * np.log(p) - _p * np.log(_p)

    return error
  elif variant == 2:
    #use ME
    type_name = S.keys()[-1]
    #ME = 0
    label_names = S[type_name].unique()
    list_of_prob = []
    for val in label_names:
      p = S[type_name].value_counts()[val] / len(S)
      list_of_prob.append(p)
    if len(list_of_prob) == max_labels:
      return np.min(list_of_prob)
    else:
      return 0
  elif variant == 3:
    #use gini
    type_name = S.keys()[-1]
    label_names = S[type_name].unique()
    sum_p = 0
    for val in label_names:
      p = S[type_name].value_counts()[val] / len(S[type_name])
      sum_p += np.square(p)

    return 1 - sum_p


def find_best_split(max_labels, variant, attr, S, total_ent):

    list_of_IG = {}
    all_attr_entropy = {}
    for A1 in attr:
        all_attr_entropy[A1] = 0
        curr_attr = A1

        curr_df = S[[curr_attr, 'y','D_t']]

        #print(curr_df)
        attr_entropy = find_total_entropy(max_labels,variant,curr_df)
        len_A1_df = len(curr_df[curr_attr])
        found_sum_expected_entropy = 0
        all_attr_values = curr_df[curr_attr].unique()
        prob_ = {}
        list_of_IG = {val: 0 for val in curr_df[curr_attr].unique()}
        for A2 in all_attr_values:
            attribute_specific = curr_df[curr_df[curr_attr] == A2]
            prob_[A2] = {}
            prob_[A2]["pos"] = sum(attribute_specific[attribute_specific.y == 1]['D_t'])/sum(attribute_specific['D_t'])
            prob_[A2]["neg"] = sum(attribute_specific[attribute_specific.y == -1]['D_t'])/sum(attribute_specific['D_t'])

        for k,v in prob_.items():
            for val,prob in v.items():
                list_of_IG[k] += -1 * (prob * np.log(prob))
        total_entropy_attribute = {val: 0 for val in curr_df[curr_attr].unique()}

        for val in curr_df[curr_attr].unique():
            prob_val = len(curr_df[curr_df[curr_attr] == val])/len(curr_df)
            total_entropy_attribute[val] = prob_val * list_of_IG[val]

        all_attr_entropy[A1] = total_ent - sum(total_entropy_attribute.values())

    return all_attr_entropy

def check_for_numerical(df_S):
    import statistics
    columns_with_number = df_S.select_dtypes(include=np.number).columns.tolist()

    df_return = df_S
    for attr in columns_with_number:
        median = statistics.median(df_S[attr])
        df_return.loc[(df_S[attr] <= median), attr] = 0.0
        df_return.loc[(df_S[attr] > median), attr] = 1.0
    return df_return


# max depth will range from 1 - 6
def ID3(max_labels, max_depth, variant, S, A, tree=None):
    # if len(S[S.keys()[-1]].unique()) > max_labels:
    max_labels = len(S['y'].unique())
    labels_value = S['y'].unique()
    Class = 'y'
    # Step 1: check if all examples have the same label
    # keep counter to make sure iterations do not exceed the max depth
    if max_depth >= 0:
        # Finding root based off of gain
        # A. Find Total entropy using variant
        total_entropy = find_total_entropy(max_labels, variant, S)
        # B. Now find the best split, which is A.
        table_attributes = S[S.columns[~S.columns.isin(['y','D_t','Weighted_y','pred','D_t+1'])]].columns
        best_A = find_best_split(max_labels, variant, table_attributes, S, total_entropy)
        A = sorted(best_A.items(), key=lambda x: x[1], reverse=True)[0][0]

        tree = {}
        tree[A] = {}
        max_depth -= 1
        try:
            values_of_A = S[A].unique()
        except:
            print('error')

        for A_val in values_of_A:
            Sv = S.loc[S[A] == A_val, S.columns != A]
            common, counts = np.unique(Sv['y'], return_counts=True)
            p = S.loc[S['y'] == 1, 'D_t'].sum()
            _p = S.loc[S['y'] == -1, 'D_t'].sum()
            if p > _p:
                tree[A][A_val] = 1
            else:
                tree[A][A_val] = -1

        return total_entropy,tree


def adaboost(S,S_test,t):
    dict_ = {}
    sum_of_vote_pred = []
    errors_per_t = []
    training_error = []
    errors_ = {}
    errors_["train"] = []
    errors_["test"] = []

    for i in range(1,t):
        dict_[i-1] = {}
        entropy , train_tree = ID3(-1 * float("inf"), 1, 1, S, "none", tree=None)
        print(train_tree)
        paths = list(dict_path(train_tree, path=None))
        for path in paths:
            attr = path[0][0]
            value = path[0][1]
            label = path[1]
            S.loc[S[attr] == value , 'pred'] = label
            S_test.loc[S_test[attr] == value, 'pred'] = label
        error_train = S.loc[S['y'] != S['pred'], 'D_t'].sum()
        error_test = S_test.loc[S_test['y'] != S_test['pred'], 'D_t'].sum()

        vote = 0.5*np.log((1-error_train)/error_train)
        dict_[i-1]['vote'] = vote
        dict_[i-1]['pred'] = S['pred'].values
        dict_[i-1]['pred_test'] = S_test['pred'].values


        S['D_t+1'] = (S['D_t']/sum(S['D_t'])) * np.power(2.718281 , (-1 * round(vote,3) * S['y'] * S['pred']))
        #S['D_t+1'] = S['D_t+1']/sum(S['D_t'])
        S['D_t'] = S['D_t+1']
        #print(i)
    #here I will do error after we get the final H
        final_pred_train = []
        final_pred_test = []
        count_train = 0
        count_test = 0
        for j in range(len(S)):
            h_i = []
            h_i_test = []
            for T in range(0,i):
                vote_A = dict_[T]['vote']
                h_pred = dict_[T]['pred'][j]
                h_pred_test = dict_[T]['pred_test'][j]
                h_i.append(vote_A * h_pred)
                h_i_test.append(vote_A * h_pred_test)
            final_pred_train.append(np.sign(sum(h_i)))
            final_pred_test.append(np.sign(sum(h_i_test)))

            if final_pred_train[j] != S['y'].loc[j]:
                count_train += 1
            if final_pred_test[j] != S_test['y'].loc[j]:
                count_test += 1
        #print(final_pred_train)
        errors_['train'].append(count_train)
        errors_['test'].append(count_test)
    return errors_['train'], errors_['test']

if __name__ == '__main__':
    bank_attributes = ["age", "job", "marital", "education", "default", "balance", "housing", "loan", "contact", "day",
                       "month", "duration", "campaign", "pdays", "previous", "poutcome", "y"]

    bank_df_train = pd.read_csv('Bank_Data/train.csv', names=bank_attributes)
    bank_df_test = pd.read_csv('Bank_Data/test.csv', names=bank_attributes)

    temp = bank_df_train
    #Print Errors with Unknown in table, but fix numericals to be binary
    bank_df_train = check_for_numerical(temp)
    bank_df_train.loc[(bank_df_train["y"] == "no"), "y"] = -1
    bank_df_train.loc[(bank_df_train["y"] == "yes"), "y"] = 1

    tempB = bank_df_test

    bank_df_test = check_for_numerical(tempB)
    bank_df_test.loc[(bank_df_test["y"] == "no"), "y"] = -1
    bank_df_test.loc[(bank_df_test["y"] == "yes"), "y"] = 1
    bank_df_test['D_t'] = 1 / len(bank_df_test)
    bank_df_test['Weighted_y'] = bank_df_test['y'] * bank_df_test['D_t']

    bank_df_test['pred'] = 0
    bank_df_test['D_t+1'] = 0

    bank_df_train['D_t'] = 1/len(bank_df_train)
    bank_df_train['Weighted_y'] = bank_df_train['y'] * bank_df_train['D_t']
    #print(bank_df_train.head(5))
   # print(bank_df_train[bank_df_train.y == 1])
    bank_df_train['pred'] = 0
    bank_df_train['D_t+1'] = 0

    sub_df = bank_df_train[['age','y','D_t']]

    errors_per_t_train = []
    errors_per_t_test = []
    axis_y = []

    train_error,test_error = adaboost(bank_df_train, bank_df_test, 40)
    print(train_error)
    print(test_error)


    #print(errors_per_t)

    data_a = errors_per_t_train
    data_b = errors_per_t_test
   # Shape is now: (10, 80)
    plt.plot(axis_y,data_a)
    plt.plot(axis_y,data_b) # plotting by columns

    plt.show()










