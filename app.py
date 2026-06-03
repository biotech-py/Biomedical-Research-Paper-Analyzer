import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import fitz
import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
load_dotenv()

GEMINI_API_KEY = (
    st.secrets.get("GEMINI_API_KEY")
    if "GEMINI_API_KEY" in st.secrets
    else os.getenv("GEMINI_API_KEY")
)
genai.configure(api_key=GEMINI_API_KEY)
if not GEMINI_API_KEY:
    st.error("Gemini API Key not found in .env file")
    st.stop()
def extract_text_from_pdf(uploaded_file):
    pdf_document = fitz.open(
        stream=uploaded_file.read(),
        filetype="pdf"
    )
    full_text = ""
    for page in pdf_document:
        full_text += page.get_text()
    return full_text
def create_pdf_report(result):
    pdf_file = "Research_Analysis_Report.pdf"
    doc = SimpleDocTemplate(pdf_file)
    styles = getSampleStyleSheet()
    content = []
    # Title
    content.append(
        Paragraph(
            "Biomedical Research Paper Analysis Report",
            styles["Title"]
        )
    )
    content.append(Spacer(1, 12))
    # Metadata
    content.append(Paragraph("Paper Metadata", styles["Heading1"]))
    content.append(Paragraph(f"<b>Title:</b> {result['title']}", styles["BodyText"]))
    content.append(Paragraph(f"<b>Authors:</b> {result['authors']}", styles["BodyText"]))
    content.append(Spacer(1, 10))
    # Scores
    content.append(Paragraph("Research Quality Assessment", styles["Heading1"]))
    content.append(
        Paragraph(
            f"""
            Novelty Score: {result['novelty_score']}/10<br/>
            Methodology Score: {result['methodology_score']}/10<br/>
            Impact Score: {result['impact_score']}/10<br/>
            Readability Score: {result['readability_score']}/10<br/>
            Overall Score: {result['overall_score']}/10
            """,
            styles["BodyText"]
        )
    )
    content.append(Spacer(1,10))
    # Recommendation
    content.append(Paragraph("Recommendation", styles["Heading1"]))
    content.append(
        Paragraph(result["recommendation"], styles["BodyText"])
    )
    content.append(Spacer(1,10))
    # Keywords
    content.append(Paragraph("Keywords", styles["Heading1"]))
    content.append(
        Paragraph(", ".join(result["keywords"]), styles["BodyText"])
    )
    content.append(Spacer(1,10))
    # Objective
    content.append(Paragraph("Research Objective", styles["Heading1"]))
    content.append(
        Paragraph(result["research_objective"], styles["BodyText"])
    )
    content.append(Spacer(1,10))
    # Techniques
    content.append(Paragraph("Techniques Used", styles["Heading1"]))
    for tech in result["techniques_used"]:
        content.append(
            Paragraph(f"• {tech}", styles["BodyText"])
        )
    content.append(Spacer(1,10))
    # Findings
    content.append(Paragraph("Key Findings", styles["Heading1"]))
    for finding in result["key_findings"]:
        content.append(
            Paragraph(f"• {finding}", styles["BodyText"])
        )
    content.append(Spacer(1,10))
    # Abstract
    content.append(Paragraph("Abstract Summary", styles["Heading1"]))
    content.append(
        Paragraph(result["abstract_summary"], styles["BodyText"])
    )
    content.append(Spacer(1,10))
    # AI Summary
    content.append(Paragraph("AI Summary", styles["Heading1"]))
    content.append(
        Paragraph(result["ai_summary"], styles["BodyText"])
    )
    content.append(Spacer(1,10))
    # Strengths
    content.append(Paragraph("Research Strengths", styles["Heading1"]))
    for point in result["strengths"]:
        content.append(
            Paragraph(f"• {point}", styles["BodyText"])
        )
    content.append(Spacer(1,10))
    # Limitations
    content.append(Paragraph("Research Limitations", styles["Heading1"]))
    for point in result["limitations"]:
        content.append(
            Paragraph(f"• {point}", styles["BodyText"])
        )
    content.append(Spacer(1,10))
    # Novelty
    content.append(Paragraph("Novelty Analysis", styles["Heading1"]))
    content.append(
        Paragraph(result["novelty_analysis"], styles["BodyText"])
    )
    content.append(Spacer(1,10))
    # Literature Review
    content.append(Paragraph("Literature Review", styles["Heading1"]))
    content.append(
        Paragraph(result["literature_review"], styles["BodyText"])
    )
    doc.build(content)
    return pdf_file
def analyze_paper(text): 
    model = genai.GenerativeModel("gemini-flash-lite-latest")
    prompt = f"""
Analyze this biomedical research paper.
Return ONLY valid JSON.
{{
"title":"",
"authors":"",
"abstract_summary":"",
"keywords":[],
"techniques_used":[],
"research_objective":"",
"key_findings":[],
"ai_summary":"",
"novelty_score":0,
"methodology_score":0,
"impact_score":0,
"reproducibility_score": 0,
"overall_score":0,
"recommendation":"",
"strengths":[],
"limitations":[],
"novelty_analysis":"",
"literature_review":"",
"research_gaps": [],
"future_work": [],
"clinical_translation_potential": "",
"commercialization_potential": "",
"phd_research_ideas":[]
}}
Rules:
- techniques_used must be a JSON list
- key_findings must be a JSON list
- keywords must be a JSON list
- No markdown
- No explanation
- JSON only
- novelty_score must be an integer from 1 to 10
- methodology_score must be an integer from 1 to 10
- impact_score must be an integer from 1 to 10
- readability_score must be an integer from 1 to 10
- reproducibility_score must be an integer from 1 to 10
- overall_score must be an integer from 1 to 10
- recommendation must be one of:
  Highly Recommended
  Moderately Recommended
  Needs Improvement
- strengths must be a JSON list with 3 to 5 points
- limitations must be a JSON list with 3 to 5 points
- novelty_analysis must be a concise paragraph explaining the innovation and uniqueness of the work
- literature_review must be a 150-250 word literature review written in academic style based on the uploaded paper
- research_gaps must contain 3 to 5 specific research gaps identified from the paper
- future_work must contain 3 to 5 future research directions
- clinical_translation_potential must explain the potential for clinical implementation in 50-100 words
- commercialization_potential must explain the potential for industrial/commercial development in 50-100 words
- phd_research_ideas must contain 3 innovative PhD-level research project ideas inspired by the uploaded paper
Generate PhD-level research ideas that extend the work described in the paper.

The ideas should:
- be novel
- be publishable
- address current limitations
- have biomedical significance
Additionally evaluate:

1. What limitations remain unsolved?
2. What research questions remain unanswered?
3. What future studies should be performed?
4. What experiments would strengthen the work?
5. What is the likelihood of translation into clinical practice?
6. What commercialization opportunities exist?
Research Paper:
{text[:5000]}
"""
    response = model.generate_content(prompt)
    clean_text = response.text.replace("```json","").replace("```","")
    return json.loads(clean_text)
def ask_paper_question(paper_text, question):
    model = genai.GenerativeModel("gemini-flash-lite-latest")
    prompt = f"""
    You are an expert biomedical research assistant.
    Answer ONLY using information from the research paper.
    If the answer is not present in the paper,
    respond with:
    "Information not found in the uploaded paper."
    Research Paper:
    {paper_text[:10000]}
    Question:
    {question}
    """
    response = model.generate_content(prompt)
    return response.text
st.set_page_config(
    page_title="Biomedical Research Paper Analyzer",
    page_icon="🧬",
    layout="wide"
)
st.title("🧬 Biomedical Research Paper Analyzer")
st.markdown("""
Upload a biomedical research paper and automatically extract:

- 📄 Title
- 👨‍🔬 Authors
- 📝 Abstract
- 🔑 Keywords
- 🧪 Techniques Used
- 🎯 Research Objective
- 📊 Key Findings
- 🧠 AI Summary
""")
num_papers = 0
if "result" not in st.session_state:
    st.session_state.result = None
    result = st.session_state.result
uploaded_files = st.file_uploader(
    "Upload Research Papers (Maximum 2 PDFs)",
    type=["pdf"],
    #accept_multiple_files=True
)
if uploaded_files:

    st.success("PDF uploaded successfully!")

    uploaded_file = uploaded_files[0]
    uploaded_file.seek(0)

    extracted_text = extract_text_from_pdf(uploaded_file)

    num_papers = len(uploaded_files)

    if num_papers == 2:

        st.info("📊 Comparison Mode Available")

        compare_button = st.button("📊 Compare Papers")

        if compare_button:

            uploaded_files[0].seek(0)
            uploaded_files[1].seek(0)

            paper1_text = extract_text_from_pdf(uploaded_files[0])
            paper2_text = extract_text_from_pdf(uploaded_files[1])

            with st.spinner("Analyzing both papers..."):

                paper1_result = analyze_paper(paper1_text)
                paper2_result = analyze_paper(paper2_text)

            st.success("Both papers analyzed successfully!")

            comparison_data = {
                "Metric": [
                    "Novelty",
                    "Methodology",
                    "Impact",
                    "Readability",
                    "Overall"
                ],
                "Paper 1": [
                    paper1_result["novelty_score"],
                    paper1_result["methodology_score"],
                    paper1_result["impact_score"],
                    paper1_result["readability_score"],
                    paper1_result["overall_score"]
                ],
                "Paper 2": [
                    paper2_result["novelty_score"],
                    paper2_result["methodology_score"],
                    paper2_result["impact_score"],
                    paper2_result["readability_score"],
                    paper2_result["overall_score"]
                ]
            }

            st.subheader("📊 Paper Comparison")
            st.table(comparison_data)

            paper1_total = (
                paper1_result["novelty_score"]
                + paper1_result["methodology_score"]
                + paper1_result["impact_score"]
                + paper1_result["readability_score"]
                + paper1_result["overall_score"]
            )

            paper2_total = (
                paper2_result["novelty_score"]
                + paper2_result["methodology_score"]
                + paper2_result["impact_score"]
                + paper2_result["readability_score"]
                + paper2_result["overall_score"]
            )

            if paper1_total > paper2_total:
                st.success(f"🏆 Winner: {paper1_result['title']}")

            elif paper2_total > paper1_total:
                st.success(f"🏆 Winner: {paper2_result['title']}")

            else:
                st.info("🤝 Both papers performed equally")

            st.subheader("📈 Visual Comparison")

            metrics = [
                "Novelty",
                "Methodology",
                "Impact",
                "Readability",
                "Overall"
            ]

            paper1_scores = [
                paper1_result["novelty_score"],
                paper1_result["methodology_score"],
                paper1_result["impact_score"],
                paper1_result["readability_score"],
                paper1_result["overall_score"]
            ]

            paper2_scores = [
                paper2_result["novelty_score"],
                paper2_result["methodology_score"],
                paper2_result["impact_score"],
                paper2_result["readability_score"],
                paper2_result["overall_score"]
            ]

            angles = np.linspace(
                0,
                2 * np.pi,
                len(metrics),
                endpoint=False
            ).tolist()

            paper1_scores += paper1_scores[:1]
            paper2_scores += paper2_scores[:1]
            angles += angles[:1]

            fig, ax = plt.subplots(
                figsize=(2.5, 2.5),
                subplot_kw=dict(polar=True)
            )

            ax.plot(
                angles,
                paper1_scores,
                linewidth=2,
                label="Paper 1"
            )

            ax.fill(
                angles,
                paper1_scores,
                alpha=0.1
            )

            ax.plot(
                angles,
                paper2_scores,
                linewidth=2,
                label="Paper 2"
            )

            ax.fill(
                angles,
                paper2_scores,
                alpha=0.1
            )

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)

            ax.set_ylim(0, 10)

            ax.legend(loc="upper right")

            col1, col2, col3 = st.columns([1,2,1])

            with col2:
             st.pyplot(fig)

            st.subheader("🧠 AI Comparison Summary")

            comparison_prompt = f"""
Compare these two biomedical research papers.

Paper 1:
Title: {paper1_result['title']}
Abstract: {paper1_result['abstract_summary']}

Paper 2:
Title: {paper2_result['title']}
Abstract: {paper2_result['abstract_summary']}

Provide:

1. Main differences
2. Which paper is more innovative?
3. Which paper has stronger methodology?
4. Which paper has higher clinical impact?
5. Final recommendation

Keep the response concise and professional.
"""

            comparison_model = genai.GenerativeModel(
                "gemini-flash-lite-latest"
            )

            comparison_response = comparison_model.generate_content(
                comparison_prompt
            )

            st.markdown(comparison_response.text)

    else:

        st.subheader("📄 Extracted Text Preview")

        st.text_area(
            "First 3000 Characters",
            extracted_text[:3000],
            height=250
        )

        if st.button("🔬 Analyze Paper"):
            with st.spinner("Analyzing with Gemini..."):
                st.session_state.result = analyze_paper(
                    extracted_text
                )

if st.session_state.result is not None:

    result = st.session_state.result
    if result is None:
     st.stop()

    # RESEARCH METRICS
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📄 Words", len(extracted_text.split()))

    with col2:
        author_count = len(
            [a.strip() for a in result["authors"].replace(" and ", ",").split(",") if a.strip()]
        )
        st.metric("👥 Authors", author_count)

    with col3:
        st.metric("🔑 Keywords", len(result["keywords"]))

    with col4:
        st.metric("📊 Findings", len(result["key_findings"]))

    st.markdown("---")

    # RESEARCH QUALITY ASSESSMENT
    st.subheader("📈 Research Quality Assessment")

    score1, score2, score3, score4, score5 = st.columns(5)

    with score1:
        st.metric("💡 Novelty", f"{result['novelty_score']}/10")

    with score2:
        st.metric("🧪 Methodology", f"{result['methodology_score']}/10")

    with score3:
        st.metric("🌍 Impact", f"{result['impact_score']}/10")

    with score4:
        st.metric("📖 Readability", f"{result['readability_score']}/10"
    )

    with score5:
        st.metric("⭐ Overall", f"{result['overall_score']}/10")

    if result["recommendation"] == "Highly Recommended":
        st.success(f"✅ {result['recommendation']}")

    elif result["recommendation"] == "Moderately Recommended":
        st.info(f"📌 Recommendation: {result['recommendation']}")

    else:
     st.error(f"❌ {result['recommendation']}")
    # PAPER METADATA
    col1, col2 = st.columns(2)
    with col1:
            st.subheader("📄 Title")
            st.info(result["title"])
    with col2:
            st.subheader("👨‍🔬 Authors")
            st.info(result["authors"])
        # ABSTRACT
    st.subheader("📝 Abstract Summary")
    st.success(result["abstract_summary"])
    # ==========================================
    # KEYWORDS
    # ==========================================
    st.subheader("🔑 Keywords")
    st.markdown(
        " • ".join(result["keywords"])
    )
    # ==========================================
    # EXPANDABLE SECTIONS
    # ==========================================
    with st.expander("🧪 Techniques Used"):
        for technique in result["techniques_used"]:
            st.markdown(f"✅ {technique}")
    with st.expander("🎯 Research Objective"):
        st.write(result["research_objective"])
    with st.expander("📊 Key Findings"):
        for i, finding in enumerate(result["key_findings"], start=1):
            st.markdown(f"**{i}.** {finding}")
    # ==========================================
    # AI SUMMARY
    # ==========================================
    st.subheader("🧠 AI Summary")
    st.success(result["ai_summary"])
    with st.expander("💪 Research Strengths"):
        for point in result["strengths"]:
            st.success(point)

    with st.expander("⚠️ Research Limitations"):
        for point in result["limitations"]:
            st.warning(point)
    st.subheader("🧬 Novelty Analysis")
    st.info(result["novelty_analysis"])
    st.subheader("🔍 Research Gap Analysis")

    for gap in result.get("research_gaps", []):
        st.warning(gap)

    st.subheader("📚 Literature Review")
    st.markdown(result["literature_review"])

    st.subheader("🚀 Future Research Directions")

    for item in result.get("future_work", []):
        st.success(item)

    st.subheader("🏥 Clinical Translation Potential")

    st.info(
        result.get(
            "clinical_translation_potential",
            "Not Available"
        )
    )

    st.subheader("💰 Commercialization Potential")

    st.info(
        result.get(
            "commercialization_potential",
            "Not Available"
        )
    )

    st.subheader("🎓 Potential PhD Research Ideas")

    for i, idea in enumerate(
        result.get("phd_research_ideas", []),
        start=1
    ):
     st.success(f"{i}. {idea}")
    report = f"""
    TITLE
    {result['title']}
    AUTHORS
    {result['authors']}
    ABSTRACT SUMMARY
    {result['abstract_summary']}
    KEYWORDS
    {", ".join(result["keywords"])}
    RESEARCH OBJECTIVE
    {result['research_objective']}
    KEY FINDINGS
    {"; ".join(result["key_findings"])}AI SUMMARY
    {result['ai_summary']}
    """
    pdf_file = create_pdf_report(result)
    with open(pdf_file, "rb") as pdf:

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf,
            file_name="Research_Analysis_Report.pdf",
            mime="application/pdf"
        )
    st.divider()
    question = st.text_input(
        "Ask anything about this paper",
        placeholder="What techniques were used?"
    )
    if st.button("🚀 Get Answer"):
        if question.strip():
            with st.spinner("Searching paper..."):
                answer = ask_paper_question(
                    extracted_text,
                    question
                )
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            st.session_state.chat_history.append(
                {
                    "question": question,
                    "answer": answer
                }
            )
    if "chat_history" in st.session_state:
        if st.button("🗑 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()   
    if "chat_history" in st.session_state:
        st.markdown("## 💬 Chat History")
        for chat in reversed(st.session_state.chat_history):
            st.markdown(
                f"**🧑 Question:** {chat['question']}"
            )
            st.markdown(
                f"**🤖 Answer:** {chat['answer']}"
            )
            st.divider()