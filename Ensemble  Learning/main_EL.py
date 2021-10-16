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
    error = -p * np.log(p) - _p * np.log(_p)

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
    # Need to check the

    list_of_IG = {}
    for A1 in attr:
        curr_attr = A1

        curr_df = S[[curr_attr, 'y','D_t']]
        #print(curr_df)

        len_A1_df = len(curr_df[curr_attr])
        found_sum_expected_entropy = 0
        all_attr_values = curr_df[curr_attr].unique()
        for A2 in all_attr_values:
            attribute_specific = curr_df[curr_df[curr_attr] == A2]
            found_expected_entropy = 0
            entropy = find_total_entropy(max_labels, variant, attribute_specific)
            found_sum_expected_entropy += (entropy * (len(attribute_specific) / len_A1_df))
        list_of_IG[A1] = total_ent - found_sum_expected_entropy
        #print(list_of_IG)
    return list_of_IG

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
    # Step 1: check if all examplesa have the same label
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
        if max_depth == 0:
            tree[A] = S['y'].mode()[0]
            return tree
        try:
            values_of_A = S[A].unique()
        except:
            print(S)
        for A_val in values_of_A:
            Sv = S.loc[S[A] == A_val, S.columns != A]
            common, counts = np.unique(Sv['y'], return_counts=True)
            if len(Sv.columns) == 1 or len(counts) == 1:
                tree[A][A_val] = common[0]
            else:
                p = Sv.loc[Sv['y'] == 1, 'D_t'].sum()
                _p = Sv.loc[Sv['y'] == -1, 'D_t'].sum()
                if p > _p:
                    tree[A][A_val] = 1
                else:
                    tree[A][A_val] = -1
        return total_entropy,tree

def get_error_of_trees(df_A,df_test_A):
    # TRAINING ERROR
    # DEPTH 1 -> 6
    # RECORD ERROR
    list_of_variants = ["Entropy", "MajorityError", "Gini Index"]
    error_dict_per_variant = {}

    testing_error = {}
    for b in range(1, 4):
        error_dict_per_variant[list_of_variants[b - 1]] = {}
        total = []
        total_test = []
        testing_error[list_of_variants[b - 1]] = {}
        for i in range(1, len(df_A.columns)):

            error_counter = 0
            error_counter_test = 0

            train_tree = ID3(-1 * float("inf"), i, b, df_A, "none", tree=None)
            train_tree_path = list(dict_path(train_tree, path=None))
            for j in range(len(train_tree_path)):
                df_filter = df_A
                df_filter_test = df_test_A
                curr_path = train_tree_path[j]
                for a in range(0, len(curr_path[0])):
                    if (a % 2) == 0:
                        df_filter = df_filter[df_filter[curr_path[0][a]] == curr_path[0][a + 1]]
                        df_filter_test = df_filter_test[df_filter_test[curr_path[0][a]] == curr_path[0][a + 1]]

                length = len(df_filter)
                length_test = len(df_filter_test)

                count_of_accuracies = len(df_filter[df_filter['y'] == curr_path[1]])
                count_of_accuracies_test = len(
                    df_filter_test[df_filter_test['y'] == curr_path[1]])

                count_of_inaccuracies = length - count_of_accuracies
                count_of_inaccuracies_test = length_test - count_of_accuracies_test

                if length == 0:
                    error_counter += 0
                else:
                    error_counter += count_of_inaccuracies / length
                if length_test == 0:
                    error_counter_test += 0
                else:
                    error_counter_test += count_of_inaccuracies_test / length_test

            total.append(error_counter / len(train_tree_path))
            total_test.append(error_counter_test/len(train_tree_path))

        error_dict_per_variant[list_of_variants[b - 1]] = np.mean(total)
        testing_error[list_of_variants[b - 1]] = np.mean(total_test)

    Errors = {'Train Average Error': [error_dict_per_variant['Entropy'], error_dict_per_variant['MajorityError'],
                                      error_dict_per_variant['Gini Index']],
              'Test Average Error': [testing_error['Entropy'], testing_error['MajorityError'],
                                     testing_error['Gini Index']]}

    error_df = pd.DataFrame(Errors, index=['Entropy', 'MajorityError', 'Gini Index'])
    return error_df

def adaboost(S,S_test,t):
    dict_ = {}
    sum_of_vote_pred = []
    errors_per_t = []
    training_error = []
    errors_ = {}

    errors_["train"] = []
    errors_["test"] = []

    for i in range(0,t):
        dict_[i] = {}
        entropy , train_tree = ID3(-1 * float("inf"), 1, 1, S, "none", tree=None)
        paths = list(dict_path(train_tree, path=None))
        for path in paths:
            attr = path[0][0]
            value = path[0][1]
            label = path[1]
            S.loc[S[attr] == value , 'pred'] = label
            S_test.loc[S_test[attr] == value, 'pred'] = label

        error_train = S.loc[S['y'] != S['pred'], 'D_t'].sum()
        error_test = S_test.loc[S_test['y'] != S_test['pred'], 'D_t'].sum()
        errors_['train'].append(error_train / np.sum(S['D_t']))
        errors_['test'].append(error_test/ np.sum(S_test['D_t']))


        errors_per_t.append(error_train)

        vote = 0.5*np.log((1-error_train)/error_train)
        dict_[i]['vote'] = vote
        dict_[i]['pred'] = S['pred'].values


        S['D_t+1'] = S['D_t'] * np.power(2.718281 , (-1 * round(vote,3) * S['y'] * S['pred']))
        S['D_t+1'] = S['D_t+1']/sum(S['D_t'])
        S['D_t'] = S['D_t+1']
        #print(i)
    #here I will do error after we get the final H
    final_pred = []
    #isSame = True
    #training error
    #err = []
    #for t_ in range(t):
    # training_error = []
    count_train = 0
    count_test = 0
    for i in range(len(S)):
        h_i = []
        for T in range(0,t):
            vote_A = dict_[T]['vote']
            h_pred = dict_[T]['pred'][i]
            h_i.append(vote_A * h_pred)
        final_pred.append(np.sign(sum(h_i)))
        if np.sign(sum(h_i)) != S['y'][i]:
            count_train += 1
        if np.sign(sum(h_i)) != S_test['y'][i]:
            count_test += 1
        #training_error.append(np.power(2.71828, S['D_t'][i] * final_pred[i] * S['y'][i]))
    #err.append(np.sum(training_error))
    #print(err)
    return errors_, count_train,count_test



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


    for i in range(1,10):

        e, e1 = adaboost(bank_df_train,bank_df_test,i)

        e = (e/5000)
        e1 = (e1/5000)


        errors_per_t_train.append(1 / np.power(2.71828 , e * i))
        errors_per_t_test.append(1 / np.power(2.71828, e1* i))
        axis_y.append(i)

    #print(errors_per_t)


    data_a = errors_per_t_train
    data_b = errors_per_t_test
   # Shape is now: (10, 80)
    plt.plot(axis_y,data_a)
    plt.plot(axis_y,data_b) # plotting by columns

    plt.show()










