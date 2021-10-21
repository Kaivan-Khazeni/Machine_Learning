import math

from DecisionTree.HW1 import main as DT



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
    for A1 in attr:
        curr_attr = A1
        curr_df = S[[curr_attr, 'y', 'D_t']]
        # print(curr_df)
        len_A1_df = len(curr_df[curr_attr])
        found_sum_expected_entropy = 0
        all_attr_values = curr_df[curr_attr].unique()
        for A2 in all_attr_values:
            attribute_specific = curr_df[curr_df[curr_attr] == A2]
            pos = sum(attribute_specific[attribute_specific.y == 1]['D_t']) / sum(
                attribute_specific['D_t'])
            neg = sum(attribute_specific[attribute_specific.y == -1]['D_t']) / sum(
                attribute_specific['D_t'])
            found_sum_expected_entropy += (-1 * (pos * np.log(pos)) - 1 * (neg * np.log(neg)))\
                                          *  len(curr_df[curr_df[curr_attr] == A2]) / len(curr_df)
        list_of_IG[A1] = total_ent - found_sum_expected_entropy
        # print(list_of_IG)
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
            p = Sv.loc[Sv['y'] == 1, 'D_t'].sum()
            _p = Sv.loc[Sv['y'] == -1, 'D_t'].sum()
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
    errors_train_arr = []
    stump_error_train = []
    stump_error_test = []
    for i in range(1,t):
        if i > 1:
            S['D_t'] = S['D_t+1']

        dict_[i] = {}
        entropy , train_tree = ID3(-1 * float("inf"), 1, 1, S, "none", tree=None)

        #print(train_tree)

        paths = list(dict_path(train_tree, path=None))
        #print(S.pred)
        for path in paths:
            #print(path)
            attr = path[0][0]
            value = path[0][1]
            label = path[1]
            S.loc[S[attr] == value , 'pred'] = label
            S_test.loc[S_test[attr] == value, 'pred'] = label
       # print(S.pred)
        stump_error_train.append(len(S.loc[S.y != S.pred])/len(S))
        stump_error_test.append(len(S_test.loc[S_test.y != S_test.pred])/len(S_test))
        error_train = sum(S.loc[S['y'] != S['pred'], 'D_t'])
        errors_train_arr.append(error_train / sum(S['D_t']))
        #print(errors_train_arr[i-1])
        error_test = sum(S_test.loc[S_test['y'] != S_test['pred'], 'D_t'])
        vote = 0.5*np.log((1-error_train)/error_train)

        dict_[i]['vote'] = vote
        dict_[i]['pred'] = [val for val in S['pred']]
        dict_[i]['pred_test'] = [val for val in S_test['pred']]


        S['D_t+1'] = (S['D_t']/sum(S['D_t'])) * np.power(2.718281 ,(-1 * vote * S['y'] * S['pred']))

        final_pred_train = []
        final_pred_test = []
        count_train = 0
        count_test = 0
        for j in range(len(S)):
            h_i = []
            h_i_test = []
            for T in range(1,i+1):
                vote_A = dict_[T]['vote']
                h_pred = dict_[T]['pred'][j]
                h_pred_test = dict_[T]['pred_test'][j]
                h_i.append(vote_A * h_pred)
                h_i_test.append(vote_A * h_pred_test)
            final_pred_train.append(np.sign(sum(h_i)))
            final_pred_test.append(np.sign(sum(h_i_test)))

        for j in range(len(final_pred_train)):
            if final_pred_test[j] != S_test.y[j]:
                count_test = count_test + 1
            if final_pred_train[j] != S.y[j]:
                count_train = count_train + 1

        errors_['train'].append(count_train/5000)
        errors_['test'].append(count_test/5000)

    #return errors_train_arr
    return stump_error_train,stump_error_test, errors_['train'], errors_['test']

def print_errors_per_iteration(train_error,test_error):
    axis_y = [i for i in range(1, 500)]
    plt.plot(axis_y, train_error, label="Train")
    plt.plot(axis_y, test_error, label="Test")  # plotting by columns
    plt.ylabel("Error")
    plt.xlabel("Iteration")
    plt.legend()
    plt.show()

def print_errors_per_stump(stump_err_train,stump_err_test):
    axis_y = [i for i in range(1, 500)]
    plt.plot(axis_y, stump_err_train, label="Train")
    plt.plot(axis_y, stump_err_test, label="Test")  # plotting by columns
    plt.ylabel("Error")
    plt.xlabel("Iteration")
    plt.legend()
    plt.show()

def bagging(S, S_test,T):
    # need to randomly sample 5000 rows for each iteration T then run
    # a decision tree for it.

    #no pruning or stopping so depth should be high for this call to ID3.
    row_numbers = list(range(0,len(S)))
    errors_per_t_train = []
    errors_per_t_test = []
    accumulation_train = 0
    accumulation_test = 0
    for t in range(1,T):
        new_df = S.sample(len(S),replace=True)
        dt = DT.ID3(-1 * float("inf"), len(S.columns), 1, new_df, "none", tree=None)
        train_tree_path = list(dict_path(dt, path=None))
        for j in range(len(train_tree_path)):
            df_filter = S
            df_filter_test = S_test
            curr_path = train_tree_path[j]
            for a in range(0, len(curr_path[0])):
                if (a % 2) == 0:
                    df_filter = df_filter[df_filter[curr_path[0][a]] == curr_path[0][a + 1]]
                    df_filter_test = df_filter_test[df_filter_test[curr_path[0][a]] == curr_path[0][a + 1]]

            length = len(df_filter)
            length_test = len(df_filter_test)

            count_of_inaccuracies = len(df_filter[df_filter.y != curr_path[1]])
            count_of_inaccuracies_test = len(df_filter_test[df_filter_test.y != curr_path[1]])
            if length == 0 or len(train_tree_path) == 0:
                accumulation_train += 0
            else:
                accumulation_train += (count_of_inaccuracies/length)/len(train_tree_path)
            if length_test == 0:
                accumulation_test += 0
            else:
                accumulation_test += (count_of_inaccuracies_test / length_test) / len(train_tree_path)

        errors_per_t_train.append(accumulation_train/t)
        errors_per_t_test.append(accumulation_test/t)
    return errors_per_t_train, errors_per_t_test


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


    # This will call adaboost and return the 4 arrays we want, below
    # I CALL EITHER ERRORS PER ITER OR ERRORS PER STUMP.  NEED TO
    # COMMENT ONE OUT AND RUN THE OTHER

    #stump_err_train, stump_err_test, train_error,test_error= adaboost(bank_df_train, bank_df_test, 500)

    #print_errors_per_iteration(train_error,test_error)
    #print_errors_per_stump(stump_err_train,stump_err_test)



    # BELOW will do bagging
    bank_df_train = pd.read_csv('Bank_Data/train.csv', names=bank_attributes)
    bank_df_test = pd.read_csv('Bank_Data/test.csv', names=bank_attributes)

    temp = bank_df_train
    bank_df_train = check_for_numerical(temp)
    bank_df_train.loc[(bank_df_train["y"] == "no"), "y"] = -1
    bank_df_train.loc[(bank_df_train["y"] == "yes"), "y"] = 1
    tempB = bank_df_test
    bank_df_test = check_for_numerical(tempB)
    bank_df_test.loc[(bank_df_test["y"] == "no"), "y"] = -1
    bank_df_test.loc[(bank_df_test["y"] == "yes"), "y"] = 1

    axis_x = [i for i in range(1, 10)]

    train, test = bagging(bank_df_train,bank_df_test, 10)
    plt.plot(axis_x,train)
    plt.plot(axis_x,test)
    plt.show()








