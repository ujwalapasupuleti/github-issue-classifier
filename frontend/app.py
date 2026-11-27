import streamlit as st
import pandas as pd
import plotly.express as px

from backend.nlp.classifier import IssueClassifier
from backend.nlp.duplicate_detector import DuplicateDetector, ISSUES
from backend.fluid.flow_engine import calculate_flow_metrics


st.set_page_config(
    page_title="GitHub Issue Classifier",
    page_icon="⚙️",
    layout="wide"
)


@st.cache_resource
def load_models():
    return IssueClassifier(), DuplicateDetector()


classifier, detector = load_models()


st.markdown("""
<style>
.main {
    background-color: #0b1020;
}
[data-testid="stMetric"] {
    background: #151e32;
    padding: 18px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


st.title("GitHub Issue Classifier & Duplicate Bug Detector")

st.caption(
    "NLP-46 | Intelligent Issue Management | "
    "Fluid Mechanics Analytics | DevOps"
)

st.divider()


tab1, tab2, tab3 = st.tabs([
    "Issue Intelligence",
    "Fluid Mechanics Analytics",
    "System Overview"
])


# ---------------- TAB 1 ----------------

with tab1:

    st.subheader("Analyze a GitHub Issue")

    title = st.text_input(
        "Issue Title",
        placeholder="Application crashes during login"
    )

    description = st.text_area(
        "Issue Description",
        placeholder="Describe the problem experienced by the user...",
        height=150
    )

    if st.button("Analyze Issue", type="primary"):

        if not title.strip():
            st.warning("Please enter an issue title.")

        else:

            prediction = classifier.predict(
                title,
                description
            )

            matches = detector.find_duplicates(
                title,
                description,
                top_k=3
            )

            st.divider()

            st.subheader("NLP Classification")

            col1, col2 = st.columns(2)

            col1.metric(
                "Predicted Category",
                prediction["category"]
            )

            col2.metric(
                "Model Confidence",
                f"{prediction['confidence']}%"
            )

            st.write("Class probabilities")

            st.bar_chart(
                pd.DataFrame(
                    prediction["probabilities"].items(),
                    columns=["Category", "Confidence"]
                ).set_index("Category")
            )

            st.subheader("Duplicate Issue Detection")

            for match in matches:

                with st.container(border=True):

                    st.write(
                        f"**GitHub Issue #{match['id']}**"
                    )

                    st.write(match["title"])

                    st.progress(
                        min(match["similarity"] / 100, 1.0)
                    )

                    st.write(
                        f"Similarity: {match['similarity']}%"
                    )

                    if match["duplicate"]:
                        st.error(
                            "Potential duplicate detected"
                        )
                    else:
                        st.success(
                            "No strong duplicate match"
                        )

            if any(match["duplicate"] for match in matches):
                st.warning(
                    "Review existing issues before creating a new ticket."
                )
            else:
                st.success(
                    "No strong duplicate detected in the sample database."
                )


# ---------------- TAB 2 ----------------

with tab2:

    st.subheader("Software Issue Flow Analytics")

    col1, col2, col3 = st.columns(3)

    with col1:
        total = st.number_input(
            "Total Incoming Issues",
            min_value=1,
            value=100
        )

    with col2:
        pending = st.number_input(
            "Pending Issues",
            min_value=0,
            value=35
        )

    with col3:
        duplicates = st.number_input(
            "Duplicate Issues",
            min_value=0,
            value=15
        )

    capacity = st.slider(
        "Processing Capacity",
        min_value=10,
        max_value=200,
        value=50
    )

    history = [10, 15, 12, 40, 18, 50]

    metrics = calculate_flow_metrics(
        total,
        pending,
        duplicates,
        capacity,
        time_minutes=60,
        hourly_history=history
    )

    st.divider()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Issue Flow Rate",
        f"{metrics['flow_rate']} issues/min"
    )

    c2.metric(
        "Backlog Pressure",
        metrics["pressure"]
    )

    c3.metric(
        "Recirculation",
        f"{metrics['recirculation']}%"
    )

    c4, c5, c6 = st.columns(3)

    c4.metric(
        "Flow Efficiency",
        f"{metrics['efficiency']}%"
    )

    c5.metric(
        "Turbulence Index",
        f"{metrics['turbulence']}%"
    )

    c6.metric(
        "System Status",
        metrics["status"]
    )

    st.subheader("Issue Pipeline")

    pipeline = pd.DataFrame({
        "Stage": [
            "Incoming",
            "Processed",
            "Pending",
            "Duplicates"
        ],
        "Issues": [
            total,
            max(total - pending, 0),
            pending,
            duplicates
        ]
    })

    fig = px.bar(
        pipeline,
        x="Stage",
        y="Issues",
        color="Stage",
        title="Software Issue Pipeline"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Fluid Mechanics Interpretation: "
        "High backlog pressure indicates that incoming work "
        "is exceeding the team's processing capacity. "
        "High turbulence indicates unstable issue arrival rates."
    )


# ---------------- TAB 3 ----------------

with tab3:

    st.subheader("System Architecture")

    st.markdown("""
    ### NLP Engine
    - TF-IDF text vectorization
    - Logistic Regression classification
    - Cosine Similarity duplicate detection

    ### Fluid Mechanics Engine
    - Issue flow rate
    - Backlog pressure
    - Turbulence index
    - Duplicate recirculation
    - Flow efficiency

    ### DevOps Architecture
    - Git version control
    - Docker containerization
    - GitHub Actions CI/CD
    - Automated testing
    """)

    st.subheader("Sample Issue Database")

    df = pd.DataFrame(ISSUES)

    st.dataframe(
        df,
        use_container_width=True
    )

    st.success("NLP and analytics engines loaded successfully.")
