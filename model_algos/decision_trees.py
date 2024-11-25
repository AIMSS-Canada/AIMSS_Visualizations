import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
st.set_page_config(layout="wide")
config = {'displayModeBar': False}
st.title("Decision Trees")

if 'Catroot' not in st.session_state:
    st.session_state['Catroot'] = 'Age'
if 'CompControot' not in st.session_state:
    st.session_state['CompControot'] = '>'
if 'ValControot' not in st.session_state:
    st.session_state['ValControot'] = 60
if 'Cat1' not in st.session_state:
    st.session_state['Cat1'] = 'ChestPainType'
if 'CompCat1' not in st.session_state:
    st.session_state['CompCat1'] = 'is'
if 'Cat2' not in st.session_state:
    st.session_state['Cat2'] = 'MaxHR'
if 'CompCont2' not in st.session_state:
    st.session_state['CompCont2'] = '<'
if 'ValCont2' not in st.session_state:
    st.session_state['ValCont2'] = 120

class TreeNode:
    def __init__(self, feature=None, condition=None, threshold=None, left=None, right=None, value=None, cvd_count=0, non_cvd_count=0):
        self.feature = feature
        self.condition = condition
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.cvd_count = cvd_count
        self.non_cvd_count = non_cvd_count

def plot_tree(node, ax, x=0.5, y=1.0, dx=0.2, dy=0.2, text_kwargs=None):
    if text_kwargs is None:
        text_kwargs = dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white')

    if node.value is not None:
        ax.text(x, y, f'CVD: {node.cvd_count}\nNon-CVD: {node.non_cvd_count}', ha='center', va='center', bbox=text_kwargs)
    else:
        text = f'{node.feature} {node.condition} {node.threshold}\nCVD: {node.cvd_count}\nNon-CVD: {node.non_cvd_count}'
        ax.text(x, y, text, ha='center', va='center', bbox=text_kwargs)

    if node.left:
        ax.plot([x, x - dx], [y, y - dy], 'k-')
        plot_tree(node.left, ax, x - dx, y - dy, dx / 2, dy, text_kwargs)

    if node.right:
        ax.plot([x, x + dx], [y, y - dy], 'k-')
        plot_tree(node.right, ax, x + dx, y - dy, dx / 2, dy, text_kwargs)

def visualize_tree(tree):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plot_tree(tree, ax)
    plt.tight_layout()
    return fig

def cont_filter(df, key, comparison, value):
    if comparison == '<':
        return df[df[key]<value], df[df[key]>=value]
    if comparison == '<=':
        return df[df[key]<=value], df[df[key]>value]
    if comparison == '>':
        return df[df[key]>value], df[df[key]<=value]
    if comparison == '>=':
        return df[df[key]>=value], df[df[key]<value]
    
def bin_filter(df, key, comparison, value):
    if comparison == 'is':
        return df[df[key].isin(value)], df[~df[key].isin(value)]
    if comparison == 'not':
        return df[~df[key].isin(value)], df[df[key].isin(value)]
    
def split_conds(text, df, cols, cat_cols, id):
    # Im sorry for this monstrosity
    col11, col12, col13 = st.columns([1,0.6,1])
    with col11:
        st.selectbox(text, options=cols[:-1], key=f'Cat{id}')
    with col12:
        if st.session_state[f'Cat{id}'] not in cat_cols:
            comp = st.selectbox('', options=['<', '<=', '>=', '>'], key=f'CompCont{id}', label_visibility='hidden')
            with col13:
                val = st.number_input(f'ValCont{id}', step=1, key=f'ValCont{id}', label_visibility='hidden')
            df_true, df_false = cont_filter(df, st.session_state[f'Cat{id}'], comp, val)

        else:
            comp = st.selectbox('', options=['is', 'not'], key=f'CompCat{id}', label_visibility='hidden')
            with col13:
                val = st.multiselect(
                    f'ValCat{id}', 
                    options=set(df[st.session_state[f'Cat{id}']]), 
                    default=df[st.session_state[f'Cat{id}']].head(1), 
                    key=f'ValCat{id}', 
                    label_visibility='hidden'
                )
            df_true, df_false = bin_filter(df, st.session_state[f'Cat{id}'], comp, val)

    return df_true, df_false, st.session_state[f'Cat{id}'], comp, val

df = pd.read_csv('./data/heart.csv')

cols = df.columns[:-1]
cat_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
target = df['HeartDisease']

with st.expander('Help'):
    st.write('''
    The dataset was obtained from Kaggle and contains 918 samples, combined from 5 datasets (Cleveland, Hungarian, Switzerland, 
    Long Beach VA, and the Stalog (Heart) Data Set), to predict the likelihood of cardiovascular disease (CVD) based on 11 features. 
          
    The goal for a decision tree is to split the data into buckets that are mostly homogenous in class. Decision trees are
    normally deeper (having more branching conditions) than this example, but it gives you an idea of how the branches work 
    and how predictions are made.
             
    **Hint:** Left is true and right is false.
    ''')

    st.dataframe(df[:5], hide_index=True)

col1, col2 = st.columns([1.5, 1])

with col2:
    root_true, root_false, root_cat, root_thresh, root_val = split_conds('Top Split Condition', df, cols, cat_cols, 'root')
    node1_true, node1_false, node1_cat, node1_thresh, node1_val = split_conds('Left Split Condition', root_true, cols, cat_cols, '1')
    node2_true, node2_false, node2_cat, node2_thresh, node2_val = split_conds('Right Split Condition', root_false, cols, cat_cols, '2')

with col1:
    leaf1 = TreeNode(value='', cvd_count=len(node1_true[node1_true['HeartDisease']==1]), non_cvd_count=len(node1_true[node1_true['HeartDisease']==0]))
    leaf2 = TreeNode(value='', cvd_count=len(node1_false[node1_false['HeartDisease']==1]), non_cvd_count=len(node1_false[node1_false['HeartDisease']==0]))
    leaf3 = TreeNode(value='', cvd_count=len(node2_true[node2_true['HeartDisease']==1]), non_cvd_count=len(node2_true[node2_true['HeartDisease']==0]))
    leaf4 = TreeNode(value='', cvd_count=len(node2_false[node2_false['HeartDisease']==1]), non_cvd_count=len(node2_false[node2_false['HeartDisease']==0]))
    node1 = TreeNode(feature=node1_cat, condition=node1_thresh, threshold=node1_val, left=leaf1, right=leaf2,
                     cvd_count=len(root_true[root_true['HeartDisease']==1]), non_cvd_count=len(root_true[root_true['HeartDisease']==0]))
    node2 = TreeNode(feature=node2_cat, condition=node2_thresh, threshold=node2_val, left=leaf3, right=leaf4,
                     cvd_count=len(root_false[root_false['HeartDisease']==1]), non_cvd_count=len(root_false[root_false['HeartDisease']==0]))
    root = TreeNode(feature=root_cat, condition=root_thresh, threshold=root_val, left=node1, right=node2,
                    cvd_count=len(df[df['HeartDisease']==1]), non_cvd_count=len(df[df['HeartDisease']==0]))
    st.write('')
    st.write('')
    fig = visualize_tree(root)

    st.pyplot(fig, use_container_width=True)
