import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import numpy as np

st.set_page_config(page_title="Data Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center;'>Data Dashboard</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        st.success("File uploaded successfully!")

        required_cols = ['Age', 'vig', 'mod', 'walk', 'BriefBESTest', 'Gender', 'Subject']
        if not all(col in df.columns for col in required_cols):
            st.warning("Some required columns are missing from the file.")
        else:
            df_clean = df.dropna(subset=required_cols)
            df_clean = df_clean[df_clean['Gender'].isin([1, 2])]
            df_clean['GenderLabel'] = df_clean['Gender'].map({1: 'Male', 2: 'Female'})

            def categorize_age(age):
                if age < 30:
                    return 'Young'
                elif 30 <= age <= 50:
                    return 'Middle-aged'
                else:
                    return 'Old'

            df_clean['AgeCategory'] = df_clean['Age'].apply(categorize_age)

            # עיצוב כפתורים
            st.markdown("""
                <style>
                .circle-wrapper {
                    display: flex;
                    justify-content: center;
                    gap: 40px;
                    margin-top: 100px;
                    margin-bottom: 60px;
                }
                .circle-button {
                    width: 150px;
                    height: 150px;
                    border-radius: 25px;
                    background: linear-gradient(135deg, #43cea2, #185a9d);
                    color: white;
                    font-size: 24px;
                    font-weight: bold;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    text-align: center;
                    box-shadow: 0 10px 20px rgba(0,0,0,0.25);
                    transition: transform 0.3s, box-shadow 0.3s;
                    cursor: pointer;
                    border: none;
                }
                .circle-button:hover {
                    transform: scale(1.05);
                    box-shadow: 0 16px 30px rgba(0,0,0,0.4);
                }
                </style>
            """, unsafe_allow_html=True)

            if 'view_option' not in st.session_state:
                st.session_state.view_option = None

            st.markdown('<div class="circle-wrapper">', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔍\nDistribution", key="dist_btn_styled", help="Click to view the Distribution Data", use_container_width=True):
                    st.session_state.view_option = "Distribution"
            with col2:
                if st.button("🚶‍♀️\nWalking", key="walk_btn_styled", help="Click to view the Walking Pattern Data", use_container_width=True):
                    st.session_state.view_option = "Walking Pattern"
            st.markdown('</div>', unsafe_allow_html=True)

            if st.session_state.view_option == "Distribution":
                st.subheader("Participants")
                st.write(f"Number of unique participants: *{df_clean['Subject'].nunique()}*")

                age_pie = df_clean.groupby('Subject')['AgeCategory'].first().value_counts().reset_index()
                age_pie.columns = ['AgeCategory', 'Count']
                fig_age = px.pie(age_pie, names='AgeCategory', values='Count', title="Age Category Distribution")

                gender_pie = df_clean.groupby('Subject')['GenderLabel'].first().value_counts().reset_index()
                gender_pie.columns = ['Gender', 'Count']
                fig_gender = px.pie(gender_pie, names='Gender', values='Count', title="Gender Distribution")

                brief_pie = df_clean.groupby('Subject')['BriefBESTest'].first().value_counts().reset_index()
                brief_pie.columns = ['Score', 'Count']
                fig_brief = px.pie(brief_pie, names='Score', values='Count', title="BriefBESTest Score Distribution")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.plotly_chart(fig_age, use_container_width=True)
                with col2:
                    st.plotly_chart(fig_gender, use_container_width=True)
                with col3:
                    st.plotly_chart(fig_brief, use_container_width=True)

                st.markdown("### Parallel Coordinates")
                dimensions = ['Age', 'vig', 'mod', 'walk', 'BriefBESTest']
                scaler = MinMaxScaler()
                df_scaled = df_clean.copy()
                df_scaled[dimensions] = scaler.fit_transform(df_scaled[dimensions])

                dimensions_list = []
                for col in dimensions:
                    original_vals = df_clean[col]
                    scaled_vals = df_scaled[col]
                    unique_vals = sorted(original_vals.unique())

                    ticktext = [str(v) for v in unique_vals]
                    col_scaler = MinMaxScaler()
                    scaled_unique = col_scaler.fit_transform(np.array(unique_vals).reshape(-1, 1)).flatten()
                    tickvals = scaled_unique

                    dimensions_list.append(dict(
                        label=col,
                        values=scaled_vals,
                        tickvals=tickvals,
                        ticktext=ticktext
                    ))

                fig_parallel = go.Figure(data=go.Parcoords(
                    line=dict(
                        color=df_clean['Gender'],
                        colorscale=[[0, 'blue'], [1, 'red']],
                        showscale=False
                    ),
                    dimensions=dimensions_list
                ))

                fig_parallel.update_layout(
                    height=700,
                    margin=dict(t=80, l=40, r=40, b=40),
                    title="Parallel Coordinates - Gender Comparison (Blue = Male, Red = Female)",
                    font=dict(size=16, color='black'),
                    paper_bgcolor='white',
                    plot_bgcolor='white'
                )

                st.plotly_chart(fig_parallel, use_container_width=True)

            elif st.session_state.view_option == "Walking Pattern":
                st.subheader("Walking Pattern Analysis")

                gender_options = st.multiselect("Select Gender", options=df_clean['GenderLabel'].unique(), default=df_clean['GenderLabel'].unique())
                age_options = st.multiselect("Select Age Category", options=df_clean['AgeCategory'].unique(), default=df_clean['AgeCategory'].unique())
                axis_option = st.selectbox("Select Axis", options=['x', 'y', 'z', 't'])
                interval_step = st.selectbox("Select Time Interval (ms)", options=[10, 20, 40, 80, 100, 200])

                filtered_df = df_clean[
                    (df_clean['GenderLabel'].isin(gender_options)) &
                    (df_clean['AgeCategory'].isin(age_options)) &
                    (df_clean['Axis'] == axis_option)
                ]

                if filtered_df.empty:
                    st.warning("No data available for this selection.")
                else:
                    st.success(f"Showing data for {filtered_df['Subject'].nunique()} participants")

                    time_cols = [str(i) for i in range(0, 2000, interval_step)]
                    valid_cols = [col for col in time_cols if col in filtered_df.columns]

                    data_for_plot = []
                    for _, row in filtered_df.iterrows():
                        for t in valid_cols:
                            data_for_plot.append({
                                'Subject': row['Subject'],
                                'Time': int(t),
                                'Value': row[t]
                            })

                    plot_df = pd.DataFrame(data_for_plot)

                    # גרף כללי לפי ציר
                    fig_all = px.line(plot_df, x='Time', y='Value', color='Subject', title="Walking Pattern Over Time")
                    st.plotly_chart(fig_all, use_container_width=True)

                    # גרף עם הדגשת משתתף
                    highlight_subject = st.selectbox("Highlight specific subject (optional)", options=["None"] + sorted(df_clean['Subject'].unique().astype(str).tolist()))
                    highlight_subject = None if highlight_subject == "None" else int(highlight_subject)

                    fig = go.Figure()
                    for subject in plot_df['Subject'].unique():
                        subject_data = plot_df[plot_df['Subject'] == subject]
                        is_highlight = (highlight_subject is not None and subject == highlight_subject)

                        fig.add_trace(go.Scatter(
                            x=subject_data['Time'],
                            y=subject_data['Value'],
                            mode='lines',
                            name=f"Subject {subject}" if is_highlight else None,
                            line=dict(color='blue' if is_highlight else 'rgba(150,150,150,0.3)', width=3 if is_highlight else 1),
                            showlegend=bool(is_highlight)
                        ))

                    fig.update_layout(
                        title=f"Walking Pattern Over Time – Axis: {axis_option.upper()}",
                        xaxis_title="Time (ms)",
                        yaxis_title="Value",
                        legend_title="Highlighted Subject" if highlight_subject else "",
                        template="plotly_white"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # גרף לכל הצירים עבור משתתף מסוים
                st.markdown("---")
                st.markdown("### View All Axes for a Single Participant")

                subject_options = df_clean['Subject'].unique()
                selected_subject = st.selectbox("Select Subject to View All Axes", options=subject_options)

                subject_axes_data = df_clean[df_clean['Subject'] == selected_subject]

                if subject_axes_data.empty:
                    st.warning("No data for this subject.")
                else:
                    fig = go.Figure()
                    axis_colors = {'x': 'red', 'y': 'green', 'z': 'blue', 't': 'orange'}

                    for axis in ['x', 'y', 'z', 't']:
                        axis_row = subject_axes_data[subject_axes_data['Axis'] == axis]
                        if not axis_row.empty:
                            row = axis_row.iloc[0]
                            time_cols = [col for col in row.index if col.isdigit()]
                            time_vals = [int(col) for col in time_cols]
                            value_vals = [row[col] for col in time_cols]

                            fig.add_trace(go.Scatter(
                                x=time_vals,
                                y=value_vals,
                                mode='lines',
                                name=f"Axis {axis.upper()}",
                                line=dict(color=axis_colors[axis])
                            ))

                    fig.update_layout(
                        title=f"Walking Pattern for Subject {selected_subject} (All Axes)",
                        xaxis_title="Time (ms)",
                        yaxis_title="Value",
                        legend_title="Axis",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading file: {e}")
