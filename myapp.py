import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Set page configuration
st.set_page_config(
    page_title='Consoleflare Analytics Portal',
    page_icon='📊',
)

# Title
st.title(":red[Data] Wizard Pro ")

# Updated subheader and description of the purpose of the app
st.subheader(":red[Seamless] Unlock the Power of Data Exploration, Transformation, and Visualization of Your Data",divider="rainbow")
st.write(":grey[The purpose of this Data Wizard Pro web application is to offer an intuitive and user-friendly platform for comprehensive data analysis. Whether you're a data analyst, scientist, or business professional, this tool will help you quickly understand, clean, and transform your datasets, and visualize meaningful insights.]")

# File upload description and instructions
st.subheader("Upload Your Dataset", divider="rainbow")
st.write(":grey[To start analyzing, simply upload a CSV, Excel, or JSON file containing the dataset you want to explore. This platform will assist you with a range of features, including data preview, cleaning, transformation, and visualizations.]")

# File uploader widget
file = st.file_uploader("Drop csv or excel or json file", type=['csv', 'xlsx', 'json'])

# File size validation and loading
if file:
    if file.size > 10 * 1024 * 1024:  # 10 MB limit
        st.warning("File size is too large. Please upload a file less than 10MB.")
    else:
        try:
            if file.name.endswith('csv'):
                data = pd.read_csv(file, encoding='ISO-8859-1')  # You can also try 'latin1' or 'utf-16'
            elif file.name.endswith('xlsx'):
                data = pd.read_excel(file)
            elif file.name.endswith('json'):
                data = pd.read_json(file)

            # Displaying the dataframe
            st.dataframe(data)
            st.info('File is successfully uploaded', icon='🚨')

            # Data Summary Tabs
            st.subheader(':rainbow[Basic Information of the Dataset]', divider='rainbow')
            tab1, tab2, tab3, tab4 = st.tabs(['Summary', 'Top and Bottom Rows', 'Data Types', 'Columns'])

            with tab1:
                st.write(f'There are {data.shape[0]} rows and {data.shape[1]} columns in the dataset.')
                st.subheader(':gray[Statistical Summary of the Dataset]')
                st.dataframe(data.describe())

            with tab2:
                st.subheader(':gray[Top Rows]')
                toprows = st.slider('Number of rows to display', 1, data.shape[0], key='topslider')
                st.dataframe(data.head(toprows))
                st.subheader(':gray[Bottom Rows]')
                bottomrows = st.slider('Number of rows to display', 1, data.shape[0], key='bottomslider')
                st.dataframe(data.tail(bottomrows))

            with tab3:
                st.subheader(':gray[Data Types of Columns]')
                st.dataframe(data.dtypes)

            with tab4:
                st.subheader('Column Names in Dataset')
                st.write(list(data.columns))

            # Basic Data Operations - Value Count
            st.subheader(':rainbow[Column Values to Count]', divider='rainbow')
            with st.expander('Value Count'):
                col1, col2 = st.columns(2)
                with col1:
                    column = st.selectbox('Choose Column Name', options=list(data.columns))
                with col2:
                    toprows = st.number_input('Top rows', min_value=1, step=1)

                count = st.button('Count')
                if count:
                    result = data[column].value_counts().reset_index().head(toprows)
                    st.dataframe(result)
                    st.subheader('Visualizations', divider='gray')

                    # Bar chart
                    fig = px.bar(data_frame=result, x=column, y='count', text='Count')
                    st.plotly_chart(fig)

                    # Line chart
                    fig = px.line(data_frame=result, x=column, y='count', text='Count')
                    st.plotly_chart(fig)

                    # Pie chart
                    fig = px.pie(data_frame=result, names=column, values='count')
                    st.plotly_chart(fig)

                    # Boxplot
                    fig = px.box(data_frame=result, y='count')
                    st.plotly_chart(fig)

            # Advanced Groupby Operations
            st.subheader(':rainbow[Groupby: Simplify Your Data Analysis]', divider='rainbow')
            st.write('The groupby method lets you summarize data by specific categories and groups.')

            with st.expander('Group By your columns'):
                col1, col2, col3 = st.columns(3)
                with col1:
                    groupby_cols = st.multiselect('Choose columns to group by', options=list(data.columns))
                with col2:
                    operation_col = st.selectbox('Choose column for operation', options=list(data.columns))
                with col3:
                    operation = st.selectbox('Choose operation', options=['sum', 'max', 'min', 'mean', 'count'])

                if groupby_cols:
                    result = data.groupby(groupby_cols).agg(
                        newcol=(operation_col, operation)
                    ).reset_index()

                    st.dataframe(result)

             # Data Visualization
                    st.subheader(":red[Data Visualization]", divider="rainbow")
                    graph_type = st.selectbox("Choose graph type", options=["line", "bar", "scatter", "pie", "sunburst", "histogram", "box", "heatmap"])

                    if graph_type == "line":
                        x_axis = st.selectbox("Choose X axis", options=list(result.columns))
                        y_axis = st.selectbox("Choose Y axis", options=list(result.columns))
                        color = st.selectbox("Choose color info", options=[None] + list(result.columns))
                        fig = px.line(data_frame=result, x=x_axis, y=y_axis, color=color, markers="o")
                        st.plotly_chart(fig)

                    elif graph_type == "bar":
                        x_axis = st.selectbox("Choose X axis", options=list(result.columns))
                        y_axis = st.selectbox("Choose Y axis", options=list(result.columns))
                        color = st.selectbox("Choose color info", options=[None] + list(result.columns))
                        facet_col = st.selectbox("Choose column info", options=[None] + list(result.columns))
                        fig = px.bar(data_frame=result, x=x_axis, y=y_axis, color=color, facet_col=facet_col, barmode="group")
                        st.plotly_chart(fig)

                    elif graph_type == "scatter":
                        x_axis = st.selectbox("Choose X axis", options=list(result.columns))
                        y_axis = st.selectbox("Choose Y axis", options=list(result.columns))
                        color = st.selectbox("Choose color info", options=[None] + list(result.columns))
                        size = st.selectbox("Choose size", options=[None] + list(result.columns))
                        fig = px.scatter(data_frame=result, x=x_axis, y=y_axis, color=color, size=size)
                        st.plotly_chart(fig)

                    elif graph_type == "pie":
                        names = st.selectbox("Choose labels", options=list(result.columns))
                        values = st.selectbox("Choose numerical values", options=list(result.columns))
                        fig = px.pie(data_frame=result, names=names, values=values)
                        st.plotly_chart(fig)

                    elif graph_type == "sunburst":
                        path = st.multiselect("Choose path", options=list(result.columns))
                        fig = px.sunburst(data_frame=result, path=path, values="newcol")
                        st.plotly_chart(fig)

                    elif graph_type == "histogram":
                        column = st.selectbox("Choose Column for Histogram", options=list(result.columns))
                        fig = px.histogram(data_frame=result, x=column)
                        st.plotly_chart(fig)

                    elif graph_type == "box":
                        fig = px.box(data_frame=result, y="newcol")
                        st.plotly_chart(fig)

                    elif graph_type == "heatmap":
                        corr = result.corr()
                        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="Viridis"))
                        st.plotly_chart(fig)

                    elif graph_type == "bubble":
                        x_axis = st.selectbox("Choose X axis for Bubble", options=list(result.columns))
                        y_axis = st.selectbox("Choose Y axis for Bubble", options=list(result.columns))
                        size = st.selectbox("Choose size for Bubble", options=[None] + list(result.columns))
                        color = st.selectbox("Choose color for Bubble", options=[None] + list(result.columns))
                        fig = px.scatter(data_frame=result, x=x_axis, y=y_axis, size=size, color=color)
                        st.plotly_chart(fig)

            # Data Cleaning Options (Optional)
            st.subheader(':rainbow[Data Cleaning]', divider='rainbow')
            with st.expander('Clean the Data'):
                if st.checkbox('Remove Duplicates'):
                    data_cleaned = data.drop_duplicates()
                    st.dataframe(data_cleaned)

                if st.checkbox('Handle Missing Values'):
                    missing_strategy = st.selectbox('Choose strategy', options=['Drop', 'Fill'])
                    if missing_strategy == 'Drop':
                        data_cleaned = data.dropna()
                    else:
                        fill_value = st.text_input('Fill with value', ' ')
                        data_cleaned = data.fillna(fill_value)
                    st.dataframe(data_cleaned)

                if st.checkbox('Replace Specific Values'):
                    column = st.selectbox('Choose column to replace values', options=list(data.columns))
                    old_value = st.text_input('Enter the value to replace')
                    new_value = st.text_input('Enter the new value')
                    data_cleaned[column] = data_cleaned[column].replace(old_value, new_value)
                    st.dataframe(data_cleaned)

                if st.checkbox('Drop Columns'):
                    columns_to_drop = st.multiselect('Choose columns to drop', options=list(data.columns))
                    data_cleaned = data.drop(columns=columns_to_drop)
                    st.dataframe(data_cleaned)

                if st.checkbox('Handle Outliers (Z-score method)'):
                    threshold = st.number_input('Z-score threshold', value=3.0, step=0.1)
                    z_scores = stats.zscore(data.select_dtypes(include=['float64', 'int64']))
                    abs_z_scores = abs(z_scores)
                    data_cleaned = data[(abs_z_scores < threshold).all(axis=1)]
                    st.dataframe(data_cleaned)

            # New Functionalities
            # Data Sampling
            st.subheader(":green[Data Sampling]", divider="green")
            with st.expander("Random Sampling"):
                sample_size = st.number_input("Sample size", min_value=1, max_value=data.shape[0], value=10, step=1)
                sampled_data = data.sample(n=sample_size)
                st.dataframe(sampled_data)

            # Correlation Matrix Heatmap
            st.subheader(":blue[Correlation Matrix Heatmap]", divider="blue")
            corr = data.corr()
            fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="Viridis"))
            st.plotly_chart(fig)

            # Interactive Data Filtering
            st.subheader(":orange[Interactive Data Filtering]", divider="orange")
            st.write("Use sliders and inputs to filter the dataset.")
            filter_column = st.selectbox("Select column to filter by", options=list(data.columns))
            min_value = st.number_input(f"Minimum {filter_column}", value=float(data[filter_column].min()))
            max_value = st.number_input(f"Maximum {filter_column}", value=float(data[filter_column].max()))
            filtered_data = data[(data[filter_column] >= min_value) & (data[filter_column] <= max_value)]
            st.dataframe(filtered_data)

            # Export Cleaned Data
            st.subheader(':blue[Download Processed Data]', divider='blue')
            file_format = st.selectbox('Select file format', options=['CSV', 'Excel', 'JSON'])
            if file_format == 'CSV':
                st.download_button('Download CSV', data=data_cleaned.to_csv(), file_name='processed_data.csv', mime='text/csv')
            elif file_format == 'Excel':
                st.download_button('Download Excel', data=data_cleaned.to_excel(), file_name='processed_data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            elif file_format == 'JSON':
                st.download_button('Download JSON', data=data_cleaned.to_json(), file_name='processed_data.json', mime='application/json')

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
