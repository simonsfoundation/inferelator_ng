import os
from . import utils
import pandas as pd
import numpy as np


class PythonDRDriver:


    def __init__(self):
        self.special_char_dictionary = {'+' : 'specialplus','-' : 'specialminus', '.' : 'specialperiod' , '/':'specialslash','\\':'special_back_slash',')':'special_paren_backward',
        '(':'special_paren_forward', ',':'special_comma', ':':'special_colon',';':'special_semicoloon','@':'special_at','=':'special_equal',
         '>':'special_great','<':'special_less','[':'special_left_bracket',']':'special_right_bracket',"%":'special_percent',"*":'special_star',
        '&':'special_ampersand','^':'special_arrow','?':'special_question','!':'special_exclamation','#':'special_hashtag',"{":'special_left_curly',
        '}':'special_right_curly','~':'special_tilde','`':'special_tildesib','$':'special_dollar','|':'special_vert_bar'}
        pass

    def get_following_conditions(self, current):
        return list(np.where(self.list_of_previouses.str.contains(current)==True)[0])

    def replace_special_characters(self, df):
        special_char_inv_map = {v: k for k, v in list(self.special_char_dictionary.items())}
        cols = df.columns.tolist()
        for sch in special_char_inv_map:
            cols = [item.replace(sch, special_char_inv_map[sch]) for item in cols]
        df.columns = cols
        return df

    def run(self, expression_mat, metadata_dataframe):

        meta_data = metadata_dataframe.copy()
        meta_data = meta_data.replace('NA', np.nan, regex=False)
        exp_mat = expression_mat.copy()

        cols=exp_mat.columns.tolist()
        for ch in self.special_char_dictionary.keys():
            #need this edge case for passing micro test
            if len(meta_data['condName'][~meta_data['condName'].isnull()]) > 0:
                meta_data['condName']= meta_data['condName'].str.replace(ch,self.special_char_dictionary[ch])
            if len(meta_data['prevCol'][~meta_data['prevCol'].isnull()]) > 0:
                meta_data['prevCol']=meta_data['prevCol'].str.replace(ch,self.special_char_dictionary[ch])
            cols=[item.replace(ch,self.special_char_dictionary[ch]) for item in cols]
        exp_mat.columns=cols

        cond = meta_data['condName'].copy()
        prev = meta_data['prevCol'].copy()
        self.list_of_previouses = prev
        delt = meta_data['del.t'].copy()
        prev.loc[delt > self.delTmax] = np.nan
        delt.loc[delt > self.delTmax] = np.nan
        not_in_mat=set(cond)-set(exp_mat)
        cond_dup = cond.duplicated()

        if len(not_in_mat) > 0:
            cond = cond.str.replace('[/+-]', '.')
            prev = cond.str.replace('[/+-]', '.')
            if cond_dup != cond.duplicated():
                raise ValueError('Tried to fix condition names in meta data so that they would match column names in expression matrix, but failed')

        # check if there are condition names missing in expression matrix
        not_in_mat=set(cond)-set(exp_mat)
        if len(not_in_mat) > 0:
            print(not_in_mat)
            raise ValueError('Error when creating design and response. The conditions printed above are in the meta data, but not in the expression matrix')

        cond_n_na = cond[~cond.isnull()]
        steady = prev.isnull() & ~(cond_n_na.isin(prev.replace(np.nan,"NA")))

        des_mat=pd.DataFrame(exp_mat[cond[steady]])
        res_mat=pd.DataFrame(exp_mat[cond[steady]]) 

        # Construct dictionary of following conditions
        following_dictionary = {}
        for i in list(np.where(~steady)[0]):
            following_dictionary[i] = self.get_following_conditions(cond[i])

        for i in list(np.where(~steady)[0]):
            following = following_dictionary[i]
            following_delt = list(delt[following])

            # Create a list of local indices to access following_delt,
            # which is a dynamic list with values that need to be recomputed for each i
            following_under_delt = []
            if len(following_delt) > 0:
                following_under_delt = list(np.where(following_delt[0] < self.delTmin)[0])

            while len(following_under_delt) > 0:
                current = following_under_delt[0]
                following_current = following_dictionary[following[current]]
                if len(following_current) == 0:
                    sum_delt_list = []
                else:
                    # Use numpy array to broadcast addition across all branches
                    sum_delt_list = np.array(delt[following_current]) + float(following_delt[current])
                # Update the following list to no longer include the current condition
                # but instead includes the conditions following the current
                following = utils.get_all_except(following, current) + following_current
                # Update the following_delt list to no longer include the current condition but replace it 
                # with the delt of the conditions following the current summed with its delt
                following_delt = list(utils.get_all_except(following_delt, current)) + list(sum_delt_list)
                following_under_delt = list(np.where(np.array(following_delt) < self.delTmin)[0])

            for (idx, j) in enumerate(following):
                column_name = cond[i]
                if len(following) > 1:
                    column_name = "%s_dupl%02d" % (cond[i], idx+1)

                    # Handle column name collision with other columns in the expression matrix
                    original_column_name = column_name
                    k = 1
                    while column_name in res_mat.columns :
                        column_name = original_column_name + '.{}'.format(int(k))
                        k = k + 1

                exp_i = exp_mat[cond[i]]
                exp_j = exp_mat[cond[j]]
                des_mat[column_name] = exp_i
                res_mat[column_name] = self.tau * (exp_j - exp_i) / following_delt[idx] + exp_i

            # special case: nothing is following this condition within delT.min
            # and it is the first of a time series --- treat as steady state
            if len(following) == 0 and prev.isnull()[i]:
                des_mat = pd.concat([des_mat, exp_mat[cond[i]]], axis=1)
                des_mat.rename(columns={des_mat.columns.values[len(des_mat.columns)-1]:cond[i]}, inplace=True)
                res_mat = pd.concat([res_mat, exp_mat[cond[i]]], axis=1)
                res_mat.rename(columns={res_mat.columns.values[len(res_mat.columns)-1]:cond[i]}, inplace=True)

        des_mat = self.replace_special_characters(des_mat)
        res_mat = self.replace_special_characters(res_mat)
        return (des_mat, res_mat)
