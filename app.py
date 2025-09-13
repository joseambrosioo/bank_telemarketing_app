import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)
from imblearn.over_sampling import SMOTE
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
import joblib
import dash_table # Add this import for the DataTable

# --- GLOBAL DATA LOADING & PREPROCESSING (for Model Training) ---
# This code runs once when the app starts, handling the full data pipeline.
def preprocess_data_for_modeling():
    # Part 2: Load the data
    d1 = pd.read_csv('dataset/Bank Marketing Data Set.csv')
    
    # Part 3: Data Cleaning & Imputation
    significant_cat_variables = ['education', 'job']
    for var in significant_cat_variables:
        d1[var + '_un'] = (d1[var] == 'unknown').astype(int)

    d1.loc[(d1['age'] > 60) & (d1['job'] == 'unknown'), 'job'] = 'retired'
    d1.loc[(d1['education'] == 'unknown') & (d1['job'] == 'admin.'), 'education'] = 'secondary'
    d1.loc[(d1['education'] == 'unknown') & (d1['job'] == 'blue-collar'), 'education'] = 'secondary'
    d1.loc[(d1['education'] == 'unknown') & (d1['job'] == 'entrepreneur'), 'education'] = 'tertiary'
    d1.loc[(d1['education'] == 'unknown') & (d1['job'] == 'housemaid'), 'education'] = 'primary'
    d1.loc[(d1['education'] == 'unknown') & (d1['job'] == 'management'), 'education'] = 'tertiary'
    d1.loc[(d1['education'] == 'unknown') & (d1['job'] == 'retired'), 'education'] = 'secondary'
    d1.loc[(d1['education'] == 'unknown') & (d1['job'] == 'self-employed'), 'education'] = 'tertiary'
    d1.loc[(d1['education'] == 'unknown') & (d1['job'] == 'services'), 'education'] = 'secondary'
    d1.loc[(d1['education'] == 'unknown') & (d1['job'] == 'student'), 'education'] = 'secondary'
    d1.loc[(d1['education'] == 'unknown') & (d1['job'] == 'technician'), 'education'] = 'secondary'
    d1.loc[(d1['education'] == 'unknown') & (d1['job'] == 'unemployed'), 'education'] = 'secondary'
    d1.loc[(d1['education'] == 'unknown') & (d1['job'] == 'unknown'), 'education'] = 'secondary'
    d1.loc[(d1['education'] == 'secondary') & (d1['job'] == 'unknown'), 'job'] = 'blue-collar'
    d1.loc[(d1['education'] == 'tertiary') & (d1['job'] == 'unknown'), 'job'] = 'blue-collar'
    d1.loc[(d1['education'] == 'primary') & (d1['job'] == 'unknown'), 'job'] = 'management'
    
    d1['pdays'] = d1['pdays'].replace(to_replace=-1, value=0)
    d1.rename(columns={'class': 'deposit', 'campain': 'campaign'}, inplace=True)
    d1['deposit'] = d1['deposit'].replace(to_replace=[1, 2], value=[0, 1])
    d1.drop(['education_un', 'job_un'], axis=1, inplace=True)
    
    # Cube root transformation
    num_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous', 'day']
    for col in num_cols:
        d1[col] = d1[col].apply(lambda x: np.cbrt(x) if x >= 0 else -np.cbrt(abs(x)))

    # Dummy Encoding
    d2 = d1.copy()
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    d2 = pd.get_dummies(data=d2, columns=cat_cols, drop_first=True)
    
    cols_to_drop_pre_modeling = ['day', 'previous', 'job_retired', 'marital_single', 'education_secondary', 'default_yes', 'contact_telephone', 'poutcome_other', 'poutcome_unknown', 'month_may']
    final_features = [col for col in d2.columns if col not in cols_to_drop_pre_modeling + ['deposit']]
    
    X = d2[final_features]
    y = d2['deposit']

    sm = SMOTE(random_state=2)
    X_resampled, y_resampled = sm.fit_resample(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=1)
    
    return X, y, X_resampled, y_resampled, X_train, X_test, y_train, y_test, final_features

# Run the preprocessing for modeling
X_data, y_data, X_resampled, y_resampled, X_train, X_test, y_train, y_test, final_features = preprocess_data_for_modeling()

# --- Model Loading (instead of training) ---
# Assuming the models have been pre-trained and saved using a separate script.
models = {
    'Logistic Regression': joblib.load('models/trained_model_logistic_regression.joblib'),
    'Decision Tree': joblib.load('models/trained_model_decision_tree.joblib'),
    'KNN Classifier': joblib.load('models/trained_model_knn_classifier.joblib'),
    'Random Forest': joblib.load('models/trained_model_random_forest.joblib'),
    'Bagging Classifier': joblib.load('models/trained_model_bagging_classifier.joblib'),
    'AdaBoost': joblib.load('models/trained_model_adaboost.joblib'),
    'Gradient Boosting': joblib.load('models/trained_model_gradient_boosting.joblib'),
    'Stacked Classifier': joblib.load('models/trained_model_stacked_classifier.joblib')
}
model_results = models
final_model = model_results['Random Forest']
y_pred_final = final_model.predict(X_test)
final_cm = confusion_matrix(y_test, y_pred_final)
final_cr = classification_report(y_test, y_pred_final, output_dict=True)

# Helper function to get model metrics
def get_metrics_df(X, y):
    df_rows = []
    for name, model in model_results.items():
        predictions = model.predict(X)
        try:
            probabilities = model.predict_proba(X)[:, 1]
            roc_auc = roc_auc_score(y, probabilities)
        except (AttributeError, IndexError):
            roc_auc = "N/A"
        
        df_rows.append({
            'Model': name,
            'Accuracy': accuracy_score(y, predictions),
            'Recall': recall_score(y, predictions, zero_division=0),
            'Precision': precision_score(y, predictions, zero_division=0),
            'F1-Score': f1_score(y, predictions, zero_division=0),
            'ROC-AUC': roc_auc,
        })
    return pd.DataFrame(df_rows).round(4)
metrics_train_df = get_metrics_df(X_test, y_test)

# --- GLOBAL DATA LOADING & PREPROCESSING (for EDA) ---
# This function loads and cleans data specifically for the EDA plots.
def preprocess_eda_data():
    d1 = pd.read_csv('dataset/Bank Marketing Data Set.csv')
    d1.rename(columns={'class': 'deposit', 'campain': 'campaign'}, inplace=True)
    d1['deposit'] = d1['deposit'].replace(to_replace=[1, 2], value=[0, 1])
    d1['pdays'] = d1['pdays'].replace(to_replace=-1, value=0)
    return d1
d1_eda = preprocess_eda_data()

# Helper function to generate column definitions with types
def get_column_definitions(df):
    cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append({"name": col, "id": col, "type": "numeric"})
        else:
            cols.append({"name": col, "id": col, "type": "text"})
    return cols

# Get the column definitions with correct types
columns_with_types = get_column_definitions(d1_eda)


# --- Dashboard Layout ---
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

header = dbc.Navbar(
    dbc.Container(
        [
            html.Div(
                [
                    html.Span("🏦", className="me-2"),
                    dbc.NavbarBrand("Bank Marketing Campaign Success", class_name="fw-bold text-wrap", style={"color": "black"}),
                ], className="d-flex align-items-center"
            ),
            dbc.Badge("Dashboard", color="primary", className="ms-auto")
        ]
    ),
    color="light",
    class_name="shadow-sm mb-3"
)

# 1. ASK Tab
ask_tab = dcc.Markdown(
    """
    ### ❓ **ASK** — Defining the Business Problem
    This project's main goal is to help a bank optimize its marketing strategy and increase the effectiveness of its telemarketing campaigns. By analyzing past customer data, we aim to predict which customers are most likely to subscribe to a term deposit. This will allow the bank to focus its efforts on high-potential customers, saving time and resources while increasing customer satisfaction by reducing unwanted calls.

    **Key Objectives:**
    - Predict customers' responses to a telemarketing campaign.
    - Identify a target customer profile for future campaigns.
    - Increase the overall success rate of term deposit subscriptions.
    """, className="p-4"
)

# 2. PREPARE Tab
prepare_tab = html.Div(
    children=[
        html.H4(["📝 ", html.B("PREPARE"), " — Getting the Data Ready"], className="mt-4"),
        html.P("To build a reliable predictive model, we first need to clean and prepare our data. This involves handling missing values, standardizing formats, and transforming variables."),
        
        html.H5("Data Source and Cleaning"),
        html.P("Our dataset contains 45,211 records of existing bank customers who were contacted via phone calls. Each record includes 17 variables such as demographics, job, education, and past campaign results. A key part of our preparation involved addressing 'unknown' values in categorical columns like 'job' and 'education.' We used a logical approach to impute these values, for instance, by inferring a person's education level based on their job. We also converted the target variable, 'deposit' (whether they subscribed or not), to a binary format (1 for 'yes', 0 for 'no') for easier modeling."),
        
        html.H5("Handling Outliers and Data Distribution"),
        html.P("Many of the numerical features, like `age` and `balance`, were not normally distributed and contained many extreme values, or 'outliers.' To correct this, we applied a **cube root transformation** to the data. This technique is especially useful for handling both positive and negative outliers and making the data more suitable for statistical analysis and machine learning models."),
        
        html.H5("Feature Engineering and Selection"),
        html.P("We converted all categorical variables (like `job` and `marital`) into numerical format using **dummy encoding**. This process creates a new binary column for each category, allowing machine learning models to understand them. For example, a single `job` column becomes multiple columns like `job_blue-collar` and `job_management`. We also performed statistical tests (Chi-square and t-tests) and analyzed **Variance Inflation Factor (VIF)** to ensure that our features were not overly correlated with each other, a condition known as **multicollinearity**. This step is crucial for building a stable and reliable model."),
        
        html.H5("Addressing Data Imbalance"),
        html.P([
            "A critical finding was that our target variable, `deposit`, was heavily imbalanced: only about ",
            html.B(f"{round(y_data.value_counts(normalize=True).loc[1] * 100, 2)}%"),
            " of customers subscribed. A simple model could get a high accuracy score by just predicting 'no' for everyone, which would be useless. To solve this, we used **SMOTE (Synthetic Minority Over-sampling Technique)**. This technique creates synthetic data points for the minority class ('yes') to balance the dataset. This ensures our models learn to recognize and predict both outcomes equally well."
        ]),
        
        html.H5("Dataset Sample (First 10 Rows)"),
        dash_table.DataTable(
            id='sample-table',
            columns=columns_with_types,
            data=d1_eda.head(10).to_dict('records'),
            sort_action="native",
            filter_action="native",
            page_action="none",
            style_table={'overflowX': 'auto', 'width': '100%'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold',
                'textAlign': 'center',
            },
            style_cell={
                'textAlign': 'left',
                'padding': '5px',
                'font-size': '12px',
                'minWidth': '80px', 'width': 'auto', 'maxWidth': '150px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
            },
        ),
        html.Br(),
    ], className="p-4"
)

# 3. ANALYZE Tab with sub-tabs
analyze_tab = html.Div(
    children=[
        html.H4(["📈 ", html.B("ANALYZE"), " — Finding Patterns and Building Models"], className="mt-4"),
        html.P("This is where we explore the data to find insights and build our predictive models."),
        dbc.Tabs([
            dbc.Tab(label="Exploratory Data Analysis", children=[
                html.Div(
                    children=[
                        html.H5("Subscription and Contact Rate by Customer Age", className="mt-4"),
                        html.P([
                            "This bar chart shows the subscription rate across different age groups. We can see a clear pattern: customers under 30 and over 60 have a significantly higher subscription rate than the middle-aged group. This insight is valuable as it helps us identify our most responsive customers."
                        ]),
                        dcc.Graph(id="age-subscription-plot", figure=go.Figure()),
                        
                        html.H5("Subscription Rate by Balance Level", className="mt-4"),
                        html.P([
                            "This plot reveals a strong positive correlation between a customer's balance and their likelihood to subscribe. Customers with higher balances are much more likely to make a term deposit. The bank can prioritize its marketing to clients with average to high balances."
                        ]),
                        dcc.Graph(id="balance-subscription-plot", figure=go.Figure()),
                        
                        html.H5("Subscription Rate by Month", className="mt-4"),
                        html.P([
                            "This line chart shows a fascinating trend. While the bank contacted most customers between May and August, the highest subscription rates occurred in the fall and spring months (March, September, October, and December). This suggests a misalignment in the bank's strategy and highlights a key opportunity for improvement."
                        ]),
                        dcc.Graph(id="month-subscription-plot", figure=go.Figure()),
                    ], className="p-4"
                )
            ]),
            dbc.Tab(label="Model Performance", children=[
                html.Div(
                    children=[
                        html.H5("Model Comparison", className="mt-4"),
                        html.P("We trained several machine learning models to see which one performs best at predicting term deposit subscriptions. We used a balanced dataset (thanks to SMOTE) to ensure our results are not skewed."),
                        html.P([
                            html.B("Key Metrics Explanation:"),
                            html.Ul([
                                html.Li("Accuracy: The percentage of correct predictions. While simple, it can be misleading on imbalanced datasets."),
                                html.Li("Precision: Of all the customers the model predicted would subscribe, how many actually did? High precision means fewer wasted marketing efforts."),
                                html.Li("Recall: Of all the customers who actually subscribed, how many did our model successfully identify? High recall means we're not missing out on potential customers."),
                                html.Li("F1-Score: A balance between Precision and Recall. It's a great single metric for evaluating performance on imbalanced datasets."),
                                html.Li("ROC-AUC: Measures the model's ability to distinguish between the two classes (subscribers and non-subscribers). A score closer to 1 is better.")
                            ])
                        ]),
                        
                        html.H5("Comparing All Models"),
                        dcc.Graph(id="model-comparison-plot", figure=go.Figure()),
                        
                        html.P([
                            "Based on the F1-Score and ROC-AUC, the **Random Forest**, **Bagging**, and **Stacked** classifiers performed exceptionally well. We selected the **Random Forest Classifier** as our final model due to its high performance and strong interpretability."
                        ]),
                        
                        html.H5("Final Model Performance: Random Forest Classifier"),
                        dcc.Graph(id="final-confusion-matrix", figure=go.Figure()),
                        
                        html.P([
                            "The Confusion Matrix visually summarizes our model's performance. The top-left cell represents True Negatives (correctly predicting 'no'). The bottom-right represents True Positives (correctly predicting 'yes'). Our model shows a high number of both, indicating strong performance on our balanced data."
                        ]),
                        dcc.Graph(id="final-roc-curve", figure=go.Figure()),
                        
                        html.P([
                            "The ROC-AUC score of ", html.B(f"{round(roc_auc_score(y_test, final_model.predict_proba(X_test)[:, 1]), 4)}"),
                            " confirms that the Random Forest model is excellent at distinguishing between customers who will subscribe and those who will not."
                        ]),
                    ], className="p-4"
                )
            ]),
        ])
    ]
)

# 4. ACT Tab
act_tab = dcc.Markdown(
    """
    ### 🚀 **ACT** — Recommendations for the Bank

    Based on our comprehensive analysis, we provide the following actionable recommendations to improve the effectiveness of future telemarketing campaigns:

    #### 1. Target the Right Customers
    - **Demographics:** Focus on customers who are either **under 30 or over 60 years old**. This is the highest-potential group for term deposits.
    - **Occupation:** Prioritize calls to **students and retired individuals**, as they have shown the highest subscription rates.
    - **Financials:** Direct marketing efforts towards clients with a **positive or high balance**, especially those with over 5,000 euros.
    
    #### 2. Optimize Campaign Timing
    - Shift the focus of telemarketing campaigns from the summer months (May-August) to the **fall and spring** (March, September, October, and December), where the success rate is significantly higher.

    #### 3. Implement the Predictive Model
    - Integrate the trained **Random Forest Classifier** into the bank's system. The model can be run daily on the customer database to generate a list of high-potential clients to call. This will ensure that marketing resources are allocated to the customers most likely to convert, increasing efficiency and revenue.
    
    #### 4. Improve Customer Experience
    - For customers with a low predicted subscription likelihood, the bank should consider alternative, non-intrusive marketing methods, such as email campaigns or targeted in-app promotions. This respects their time and can improve overall customer satisfaction.
    """, className="p-4"
)

# Dashboard Layout
app.layout = dbc.Container(
    [
        header,
        dbc.Tabs(
            [
                dbc.Tab(ask_tab, label="Ask"),
                dbc.Tab(prepare_tab, label="Prepare"),
                dbc.Tab(analyze_tab, label="Analyze"),
                dbc.Tab(act_tab, label="Act"),
            ]
        ),
    ],
    fluid=True,
)

# --- Callbacks for Interactive Plots ---
@app.callback(
    Output("age-subscription-plot", "figure"),
    Output("balance-subscription-plot", "figure"),
    Output("month-subscription-plot", "figure"),
    Input("age-subscription-plot", "id") # Dummy input to trigger on load
)
def update_eda_plots(dummy):
    # Age plot
    d1_temp_age = d1_eda.copy()
    d1_temp_age.loc[d1_temp_age["age"] < 30, 'age_group'] = 20
    d1_temp_age.loc[(d1_temp_age["age"] >= 30) & (d1_temp_age["age"] <= 39), 'age_group'] = 30
    d1_temp_age.loc[(d1_temp_age["age"] >= 40) & (d1_temp_age["age"] <= 49), 'age_group'] = 40
    d1_temp_age.loc[(d1_temp_age["age"] >= 50) & (d1_temp_age["age"] <= 59), 'age_group'] = 50
    d1_temp_age.loc[d1_temp_age["age"] >= 60, 'age_group'] = 60
    
    count_age_response_pct = pd.crosstab(d1_temp_age['deposit'], d1_temp_age['age_group']).apply(lambda x: x / x.sum() * 100)
    count_age_response_pct = count_age_response_pct.transpose()
    age_plot = go.Figure(data=[
        go.Bar(name='Not Subscribed', x=count_age_response_pct.index, y=count_age_response_pct[0], marker_color='red'),
        go.Bar(name='Subscribed', x=count_age_response_pct.index, y=count_age_response_pct[1], marker_color='green')
    ])
    age_plot.update_layout(
        title='Subscription Rate by Age Group',
        xaxis_title='Age Group',
        yaxis_title='Percentage',
        barmode='group'
    )
    
    # Balance plot
    d1_temp_balance = d1_eda.copy()
    d1_temp_balance['balance'].replace(to_replace=-1, value=0, inplace=True)
    d1_temp_balance.loc[d1_temp_balance["balance"] <= 0, 'balance_group'] = 'no balance'
    d1_temp_balance.loc[(d1_temp_balance["balance"] > 0) & (d1_temp_balance["balance"] <= 1000), 'balance_group'] = 'low balance'
    d1_temp_balance.loc[(d1_temp_balance["balance"] > 1000) & (d1_temp_balance["balance"] <= 5000), 'balance_group'] = 'average balance'
    d1_temp_balance.loc[(d1_temp_balance["balance"] > 5000), 'balance_group'] = 'high balance'
    
    count_balance_response_pct = pd.crosstab(d1_temp_balance['deposit'], d1_temp_balance['balance_group']).apply(lambda x: x / x.sum() * 100)
    count_balance_response_pct = count_balance_response_pct.transpose()
    balance_plot = go.Figure(data=[
        go.Bar(name='Not Subscribed', x=count_balance_response_pct.index, y=count_balance_response_pct[0], marker_color='skyblue'),
        go.Bar(name='Subscribed', x=count_balance_response_pct.index, y=count_balance_response_pct[1], marker_color='royalblue')
    ])
    balance_plot.update_layout(
        title='Subscription Rate by Balance Level',
        xaxis_title='Balance Category',
        yaxis_title='Percentage',
        barmode='group'
    )

    # Month plot
    d1_temp_month = d1_eda.copy()
    month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    count_month_response_pct = pd.crosstab(d1_temp_month['deposit'], d1_temp_month['month']).apply(lambda x: x / x.sum() * 100).reindex(columns=month_order)
    
    month_plot = go.Figure(data=go.Scatter(
        x=count_month_response_pct.columns,
        y=count_month_response_pct.loc[1],
        mode='lines+markers',
        name='Subscription Rate'
    ))
    month_plot.update_layout(
        title='Subscription Rate by Month',
        xaxis_title='Month',
        yaxis_title='Subscription Rate (%)'
    )
    
    return age_plot, balance_plot, month_plot

@app.callback(
    Output("model-comparison-plot", "figure"),
    Output("final-confusion-matrix", "figure"),
    Output("final-roc-curve", "figure"),
    Input("model-comparison-plot", "id") # Dummy input
)
def update_model_performance_plots(dummy):
    # Model Comparison Plot
    df_metrics = metrics_train_df.set_index('Model').T
    model_comparison_fig = go.Figure()
    for model_name in df_metrics.columns:
        model_comparison_fig.add_trace(go.Bar(
            name=model_name,
            x=df_metrics.index,
            y=df_metrics[model_name]
        ))
    model_comparison_fig.update_layout(
        title='Comparison of Model Performance on Test Data',
        yaxis_title='Score',
        barmode='group'
    )
    
    # Final Model Confusion Matrix
    cm_labels = ['Not Subscribed (0)', 'Subscribed (1)']
    final_cm_fig = ff.create_annotated_heatmap(
        z=final_cm, 
        x=cm_labels, 
        y=cm_labels,
        colorscale='Blues',
        showscale=True
    )
    final_cm_fig.update_layout(
        title='Confusion Matrix for Final Random Forest Model',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        xaxis=dict(side="bottom")
    )

    # Final Model ROC Curve
    fpr, tpr, _ = roc_curve(y_test, final_model.predict_proba(X_test)[:, 1])
    final_roc_fig = go.Figure(data=[
        go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'),
        go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random Guess')
    ])
    final_roc_fig.update_layout(
        title='ROC Curve for Final Random Forest Model',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate'
    )

    return model_comparison_fig, final_cm_fig, final_roc_fig

if __name__ == '__main__':
    app.run(debug=True)