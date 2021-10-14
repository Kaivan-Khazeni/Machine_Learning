import math

import pandas as pd
import numpy as np

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
        print(best_A)
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
                tree[A][A_val] = common[0]
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

def adaboost(S,t):
    for i in range(0,t):
        entropy , train_tree = ID3(-1 * float("inf"), 1, 1, S, "none", tree=None)
        print(train_tree)
        paths = list(dict_path(train_tree, path=None))
        #print(paths)
        for path in paths:
            attr = path[0][0]
            value = path[0][1]
            label = path[1]
            S.loc[S[attr] == value , 'pred'] = label
        #print(-vote * S['y'] * S['pred'])

        error = S.loc[S['y'] != S['pred'], 'D_t'].sum()
        print(error)
        vote = 0.5*np.log((1-error)/error)
        #print(vote)
        for j in range(0,len(S)):
            S['D_t+1'].loc[j] = S['D_t'].loc[j] * math.exp(-vote * S['y'].loc[j] * S['pred'].loc[j])
            S['D_t+1'].loc[j] = S['D_t+1'].loc[j]/sum(S['D_t+1'])
            S['D_t'].loc[j] = S['D_t+1'].loc[j]







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


    bank_df_train['D_t'] = 1/len(bank_df_train)
    bank_df_train['Weighted_y'] = bank_df_train['y'] * bank_df_train['D_t']
    #print(bank_df_train.head(5))
   # print(bank_df_train[bank_df_train.y == 1])
    bank_df_train['pred'] = 0
    bank_df_train['D_t+1'] = 0

    sub_df = bank_df_train[['age','y','D_t']]






    adaboost(bank_df_train, 500)





