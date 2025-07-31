import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="Gait Detection", layout="wide")
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>Gait Detection</h1>", unsafe_allow_html=True)

# בחר דף בסיידבר
view_option = st.sidebar.radio(
    "Choose View:",
    options=["", "Distribution", "Individual Walking Pattern", "Group Walking Pattern"],
    format_func=lambda x: "Main Menu" if x == "" else x
    
)

if view_option == "":
    # רק בדף הראשי תיבת העלאה ושמירת הנתונים בזיכרון
    uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            
            # ניקוי רווחים בשמות העמודות
            df.columns = df.columns.str.strip()

            required_cols = ['Age', 'vig', 'mod', 'walk', 'BriefBESTest', 'Gender', 'Subject', 'Axis']
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

                # שמירת DataFrame ב-session_state כדי לשמור לגלישה בין דפים
                st.session_state['df_clean'] = df_clean

                st.success("File uploaded and processed successfully!")
                st.markdown("Please choose a view from the sidebar to continue.")

        except Exception as e:
            st.error(f"Error loading file: {e}")

    else:
        st.markdown("### Please upload an Excel or CSV file with relevant data to start.")

elif view_option == "Distribution":
    # בודקים שיש נתונים ב-session_state
    if 'df_clean' not in st.session_state:
        st.warning("Please upload a valid data file in the Main Menu first.")
    else:
        df_clean = st.session_state['df_clean']

        st.subheader("Participants Overview")
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

        st.markdown("### Parallel Coordinates Plot")
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
            font=dict(size=16, color='#2c3e50'),
            paper_bgcolor='#f9f9f9',
            plot_bgcolor='#f9f9f9'
        )

        st.plotly_chart(fig_parallel, use_container_width=True)


elif view_option == "Individual Walking Pattern":
    if 'df_clean' not in st.session_state:
        st.warning("Please upload a valid data file in the Main Menu first.")
    else:
        df_clean = st.session_state['df_clean']

        # CSS לכחול לתוויות ולכותרות
        st.markdown("""
        <style>
        /* צבע הכניסה של התיבה והווי */
        .stCheckbox > div > div > input:checked + label::before {
            border-color: #0d6efd !important;
            background-color: #0d6efd !important;
        }
        
        /* צבע הווי (סימון √) */
        .stCheckbox > div > div > input:checked + label::after {
            border-color: white !important;  /* כדי שהווי יהיה לבן על רקע כחול */
            border-width: 2px !important;
        }
        
        /* צבע הטקסט של התיבה המסומנת */
        .stCheckbox > div > div > input:checked + label {
            color: #0d6efd !important;
            font-weight: 600;
        }
        </style>
        """, unsafe_allow_html=True)

        with st.sidebar.expander("Age Category Definitions"):
            st.markdown("""
            - **Young:** Participants aged 18–30 years  
            - **Middle-aged:** Participants aged 31–60 years  
            - **Old:** Participants aged 61 years and above  
            """)

        st.subheader("Walking Pattern Analysis")

        cols = st.columns(4)

        # Gender selection with checkboxes
        gender_unique = df_clean['GenderLabel'].unique()
        gender_selected = []
        with cols[0]:
            st.markdown("<b>Select Gender</b>", unsafe_allow_html=True)
            for gender in gender_unique:
                checked = st.checkbox(gender, value=True, key=f"gender_{gender}")
                if checked:
                    gender_selected.append(gender)

        # Age category selection with checkboxes
        age_unique = df_clean['AgeCategory'].unique()
        age_selected = []
        with cols[1]:
            st.markdown("<b>Select Age Category</b>", unsafe_allow_html=True)
            for age_cat in age_unique:
                checked = st.checkbox(age_cat, value=True, key=f"age_{age_cat}")
                if checked:
                    age_selected.append(age_cat)

        # Axis selection (שומר על selectbox רגיל)
        with cols[2]:
            axis_option = st.selectbox(
                "Select Axis",
                options=['x', 'y', 'z', 't'],
                index=0,
                key="axis_filter"
            )

        # Interval selection (שומר על selectbox רגיל)
        with cols[3]:
            interval_step = st.selectbox(
                "Time Interval (ms)",
                options=[10, 20, 40, 80, 100, 200],
                index=0,
                key="interval_filter"
            )

        stat_option = st.selectbox(
            "Statistic to Display",
            options=["Mean", "Median"],
            index=0,
            key="stat_filter"
        )
       
        time_min = 0
        time_max = 2000
      
        selected_range = st.slider(
            "Select Time Range to Display (ms)",
            min_value=time_min,
            max_value=time_max,
            value=(time_min, time_max),
            step=interval_step
        )

        filtered_df = df_clean[
            (df_clean['GenderLabel'].isin(gender_selected)) &
            (df_clean['AgeCategory'].isin(age_selected)) &
            (df_clean['Axis'] == axis_option)
        ]

        if filtered_df.empty:
            st.warning("No data available for this selection.")
        else:
            st.success(f"Showing data for {filtered_df['Subject'].nunique()} participants")

            time_cols = [str(i) for i in range(0, 2000, interval_step)]
            valid_cols = [col for col in time_cols if col in filtered_df.columns]
            valid_cols = [col for col in valid_cols if selected_range[0] <= int(col) <= selected_range[1]]

            def compute_stat(vals, stat_option):
                vals = vals.dropna()
                if stat_option == "Mean":
                    return vals.mean()
                elif stat_option == "Median":
                    return vals.median()
                return None

            subj_data = []
            for subject, group in filtered_df.groupby('Subject'):
                subj_vals = []
                for t in valid_cols:
                    vals = group[t]
                    stat_val = compute_stat(vals, stat_option)
                    subj_vals.append(stat_val)
                subj_data.append(pd.DataFrame({
                    'Time': [int(t) for t in valid_cols],
                    'Value': subj_vals,
                    'Subject': subject
                }))
            subj_df = pd.concat(subj_data, ignore_index=True)

            trend_vals = []
            for t in valid_cols:
                vals = subj_df[subj_df['Time'] == int(t)]['Value']
                trend_val = compute_stat(vals, stat_option)
                trend_vals.append(trend_val)

            fig_colorful = go.Figure()
            for subject in subj_df['Subject'].unique():
                subject_data = subj_df[subj_df['Subject'] == subject]
                fig_colorful.add_trace(go.Scatter(
                    x=subject_data['Time'],
                    y=subject_data['Value'],
                    mode='lines',
                    name=f"Subject {subject}",
                    line=dict(width=1),
                    showlegend=True
                ))

            fig_colorful.add_trace(go.Scatter(
                x=[int(t) for t in valid_cols],
                y=trend_vals,
                mode='lines',
                line=dict(color='black', width=2),
                name=f"{stat_option} Trend",
                hoverinfo='name+y',
                showlegend=True 
            ))

            fig_colorful.update_layout(
                title=f"Walking Pattern Over Time – Axis: {axis_option.upper()} (All Subjects - Colorful)",
                xaxis_title="Time (ms)",
                yaxis_title="Value",
                template="plotly_white",
                height=600,
                font=dict(color='#2c3e50')
            )
            st.plotly_chart(fig_colorful, use_container_width=True)

            st.markdown("---")
            st.markdown("### Highlight Specific Subject")

            subject_options = sorted(filtered_df['Subject'].unique())
            highlight_subject = st.selectbox(
                "Select Subject to Highlight",
                options=["None"] + [str(s) for s in subject_options],
                index=0
            )
            highlight_subject_val = None if highlight_subject == "None" else int(highlight_subject)

            fig_gray = go.Figure()
            for subject in subj_df['Subject'].unique():
                subject_data = subj_df[subj_df['Subject'] == subject]
                is_highlight = (highlight_subject_val is not None and subject == highlight_subject_val)

                fig_gray.add_trace(go.Scatter(
                    x=subject_data['Time'],
                    y=subject_data['Value'],
                    mode='lines',
                    name=f"Subject {subject}" if is_highlight else None,
                    line=dict(
                        color='blue' if is_highlight else 'rgba(150,150,150,0.3)',
                        width=3 if is_highlight else 1
                    ),
                    showlegend=bool(is_highlight),
                    hoverinfo='name+y'
                ))

            fig_gray.add_trace(go.Scatter(
                x=[int(t) for t in valid_cols],
                y=trend_vals,
                mode='lines',
                line=dict(color='black', width=2),
                name=f"{stat_option} Trend",
                hoverinfo='name+y'
            ))

            fig_gray.update_layout(
                title=f"Walking Pattern with Highlighted Subject – Axis: {axis_option.upper()}",
                xaxis_title="",  # הסרת תווית ציר X
                yaxis_title="Value",
                legend_title="Subjects",
                template="plotly_white",
                font=dict(color='#2c3e50'),
                height=600
            )
            st.plotly_chart(fig_gray, use_container_width=True)

            st.markdown("---")
            st.markdown("### View All Axes for a Single Participant")

            subject_options_all = df_clean['Subject'].unique()
            selected_subject = st.selectbox("Select Subject to View All Axes", options=subject_options_all)

            subject_axes_data = df_clean[df_clean['Subject'] == selected_subject]

            if subject_axes_data.empty:
                st.warning("No data for this subject.")
            else:
                fig_axes = go.Figure()
                axis_colors = {'x': 'red', 'y': 'green', 'z': 'blue', 't': 'orange'}

                for axis in ['x', 'y', 'z', 't']:
                    axis_row = subject_axes_data[subject_axes_data['Axis'] == axis]
                    if not axis_row.empty:
                        row = axis_row.iloc[0]
                        time_cols = [col for col in row.index if col.isdigit()]
                        time_vals = [int(col) for col in time_cols]
                        value_vals = [row[col] for col in time_cols]

                        fig_axes.add_trace(go.Scatter(
                            x=time_vals,
                            y=value_vals,
                            mode='lines',
                            name=f"Axis {axis.upper()}",
                            line=dict(color=axis_colors[axis]),
                            showlegend=True
                        ))

                fig_axes.update_layout(
                    title=f"Walking Pattern for Subject {selected_subject} (All Axes)",
                    xaxis_title="Time (ms)",
                    yaxis_title="Value",
                    legend_title="Axis",
                    template="plotly_white",
                    font=dict(color='#2c3e50')
                )
                st.plotly_chart(fig_axes, use_container_width=True)

                     
elif view_option == "Group Walking Pattern":
    if 'df_clean' not in st.session_state:
        st.warning("Please upload a valid data file in the Main Menu first.")
    else:
        df_clean = st.session_state['df_clean']

        # CSS לסימון כחול בצ'קבוקסים
        st.markdown("""
            <style>
            input[type=checkbox]:checked + div[data-testid="stMarkdownContainer"] svg {
                color: #1f77b4 !important;  /* כחול */
            }
            </style>
        """, unsafe_allow_html=True)

        with st.sidebar.expander("Age Category Definitions"):
            st.markdown("""
            - **Young:** Participants aged 18–30 years  
            - **Middle-aged:** Participants aged 31–60 years  
            - **Old:** Participants aged 61 years and above  
            """)

        with st.sidebar.expander("BriefBESTest Score Interpretation"):
            st.markdown("""
            - **Normal:** Score between 22 and 24 – Indicates typical balance ability  
            - **Abnormal:** Score between 17 and 21 – Indicates balance impairments  
            - **Other:** Score below 17 or missing – Considered as out of scope  
            """)

        st.subheader("Group Walking Analysis")

        def brief_category(score):
            if 22 <= score <= 24:
                return "Normal"
            elif 17 <= score <= 21:
                return "Abnormal"
            else:
                return "Other"

        df_clean['BriefBESTestCat'] = df_clean['BriefBESTest'].apply(brief_category)

        with st.form("group_filter_form"):
            cols = st.columns(4)

            with cols[0]:
                st.markdown("**Select Gender(s):**")
                gender_options_raw = df_clean['GenderLabel'].unique()
                gender_options = [g for g in gender_options_raw if st.checkbox(g, value=True, key=f"gender_{g}")]

            with cols[1]:
                st.markdown("**Select Age Category(ies):**")
                age_options_raw = df_clean['AgeCategory'].unique()
                age_options = [a for a in age_options_raw if st.checkbox(a, value=True, key=f"age_{a}")]

            with cols[2]:
                st.markdown("**Select BriefBESTest Category(ies):**")
                brief_options_raw = ["Normal", "Abnormal"]
                brief_options = [b for b in brief_options_raw if st.checkbox(b, value=True, key=f"brief_{b}")]

            with cols[3]:
                axis_option = st.selectbox(
                    "Select Axis",
                    options=['x', 'y', 'z', 't']
                )

            interval_step = st.selectbox(
                "Select Time Interval (ms)",
                options=[10, 20, 40, 80, 100, 200]
            )

            stat_option = st.selectbox(
                "Select Statistic to Display",
                options=["Mean", "Median"],
                index=0
            )

            submitted = st.form_submit_button("Update Graph")

        if submitted:
            def compute_stat(vals, stat_option):
                vals = vals.dropna()
                if stat_option == "Mean":
                    return vals.mean()
                elif stat_option == "Median":
                    return vals.median()
                else:
                    return None

            filtered_df = df_clean[
                (df_clean['GenderLabel'].isin(gender_options)) &
                (df_clean['AgeCategory'].isin(age_options)) &
                (df_clean['BriefBESTestCat'].isin(brief_options)) &
                (df_clean['Axis'] == axis_option)
            ]

            if filtered_df.empty:
                st.warning("No data available for the selected filters.")
            else:
                group_keys = filtered_df.groupby(['GenderLabel', 'AgeCategory', 'BriefBESTestCat'])

                time_cols = [str(i) for i in range(0, 2000, interval_step)]
                valid_cols = [col for col in time_cols if col in filtered_df.columns]

                if not valid_cols:
                    st.warning("No time columns matching the selected interval.")
                else:
                    fig = go.Figure()

                    for (gender, age_cat, brief_cat), group in group_keys:
                        if group.empty:
                            continue

                        stat_vals = [compute_stat(group[t], stat_option) for t in valid_cols]

                        if all(v is None for v in stat_vals):
                            continue

                        group_name = f"{gender}, {age_cat}, {brief_cat}"

                        fig.add_trace(go.Scatter(
                            x=[int(t) for t in valid_cols],
                            y=stat_vals,
                            mode='lines+markers',
                            name=group_name,
                            hoverinfo='name+y'
                        ))

                    if not fig.data:
                        st.warning("No valid data available for the selected groups after filtering.")
                    else:
                        fig.update_layout(
                            title=f"Group Walking Pattern Over Time – Axis: {axis_option.upper()}",
                            xaxis_title="Time (ms)",
                            yaxis_title="Value",
                            legend_title="Groups (Gender, Age, BriefBESTest)",
                            template="plotly_white",
                            font=dict(color='#2c3e50'),
                            height=600
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    st.markdown("---")
                    st.markdown("### Groups with No Data")

                    all_combinations = [
                        (g, a, b)
                        for g in gender_options
                        for a in age_options
                        for b in brief_options
                    ]

                    no_data_groups = []
                    for combo in all_combinations:
                        gender_c, age_c, brief_c = combo
                        check_df = filtered_df[
                            (filtered_df['GenderLabel'] == gender_c) &
                            (filtered_df['AgeCategory'] == age_c) &
                            (filtered_df['BriefBESTestCat'] == brief_c)
                        ]
                        if check_df.empty:
                            no_data_groups.append(combo)

                    if no_data_groups:
                        for g, a, b in no_data_groups:
                            st.write(f"- No data for group: **{g}**, **{a}**, **{b}**")
                    else:
                        st.write("All selected groups contain data.")










