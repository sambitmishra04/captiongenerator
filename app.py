"""
Redesigned Streamlit UI for image caption generation.
"""

from __future__ import annotations

from io import BytesIO

import streamlit as st
from PIL import Image

from cnn_lstm_captioner import CNNLSTMCaptioner


st.set_page_config(
    page_title="Image Caption Studio",
    page_icon="*",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --- Configuration ---
with st.sidebar:
    st.title("Settings")
    st.info("The app automatically searches for models in your project folder and Google Drive.")
    manual_model_path = st.text_input("Manual Model Path (Optional)", value="", help="Point to a specific .h5 file if discovery fails.")
    manual_tokenizer_path = st.text_input("Manual Tokenizer Path (Optional)", value="", help="Point to a specific .pkl file.")

@st.cache_resource
def load_captioner(m_path: str | None = None, t_path: str | None = None) -> CNNLSTMCaptioner:
    # Use defaults if manual paths are empty
    m_p = m_path if m_path and m_path.strip() else "mymodel.h5"
    t_p = t_path if t_path and t_path.strip() else "tokenizer.pkl"
    
    return CNNLSTMCaptioner(
        model_path=m_p,
        tokenizer_path=t_p,
        max_caption_length=34,
        image_size=(224, 224),
    )


captioner = load_captioner(manual_model_path, manual_tokenizer_path)
runtime = captioner.get_runtime_info()
backbone_label = (runtime["backbone"] or "unknown").upper()
feature_dim = runtime["feature_dim"] or "unknown"
resolved_model_path = runtime.get("model_path", "unknown")
model_location = "Google Drive" if "drive" in str(resolved_model_path).lower() else "Local Project"


st.markdown(
    """
<style>
    :root {
        --bg: #f6f7fb;
        --panel: #ffffff;
        --panel-soft: #f9fbff;
        --text: #18212f;
        --muted: #5f6b7a;
        --line: #dde5f0;
        --accent: #1769ff;
        --accent-soft: #e9f1ff;
        --success: #0f9d58;
        --shadow: 0 20px 45px rgba(24, 33, 47, 0.08);
        --radius-xl: 26px;
        --radius-lg: 18px;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(23, 105, 255, 0.10), transparent 30%),
            radial-gradient(circle at top right, rgba(66, 133, 244, 0.10), transparent 25%),
            linear-gradient(180deg, #f8fbff 0%, var(--bg) 55%, #eef3fb 100%);
        color: var(--text);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    div[data-testid="stVerticalBlock"] > div:has(.hero-card),
    div[data-testid="stVerticalBlock"] > div:has(.panel-card),
    div[data-testid="stVerticalBlock"] > div:has(.result-card),
    div[data-testid="stVerticalBlock"] > div:has(.metric-chip) {
        width: 100%;
    }

    .hero-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 60%, #eef5ff 100%);
        border: 1px solid rgba(255, 255, 255, 0.8);
        box-shadow: var(--shadow);
        border-radius: var(--radius-xl);
        padding: 2.2rem 2.4rem;
        margin-bottom: 1.2rem;
    }

    .eyebrow {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .hero-title {
        margin: 0.85rem 0 0.35rem 0;
        font-size: 3rem;
        line-height: 1.02;
        font-weight: 800;
        color: var(--text);
    }

    .hero-subtitle {
        margin: 0;
        font-size: 1.02rem;
        line-height: 1.7;
        color: var(--muted);
        max-width: 760px;
    }

    .chip-row {
        display: flex;
        gap: 0.7rem;
        flex-wrap: wrap;
        margin-top: 1.25rem;
    }

    .metric-chip {
        background: rgba(255, 255, 255, 0.75);
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 0.75rem 0.95rem;
        color: var(--text);
        font-size: 0.92rem;
        box-shadow: 0 10px 25px rgba(24, 33, 47, 0.04);
    }

    .metric-chip b {
        color: var(--accent);
    }

    .panel-card {
        background: var(--panel);
        border: 1px solid rgba(221, 229, 240, 0.95);
        border-radius: var(--radius-xl);
        box-shadow: var(--shadow);
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
    }

    .section-title {
        margin: 0 0 0.25rem 0;
        font-size: 1.2rem;
        font-weight: 750;
        color: var(--text);
    }

    .section-subtitle {
        margin: 0;
        color: var(--muted);
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .step-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.9rem;
        margin-top: 1.2rem;
    }

    .step-card {
        background: var(--panel-soft);
        border: 1px solid var(--line);
        border-radius: var(--radius-lg);
        padding: 1rem;
        min-height: 130px;
    }

    .step-number {
        width: 2rem;
        height: 2rem;
        border-radius: 999px;
        background: var(--accent);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        margin-bottom: 0.85rem;
    }

    .step-title {
        margin: 0 0 0.35rem 0;
        font-size: 1rem;
        font-weight: 700;
        color: var(--text);
    }

    .step-copy {
        margin: 0;
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.55;
    }

    .result-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid var(--line);
        border-radius: var(--radius-xl);
        box-shadow: var(--shadow);
        padding: 1.35rem 1.45rem;
        margin: 0.75rem 0 1rem 0;
    }

    .caption-pill {
        display: inline-block;
        margin-bottom: 0.85rem;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: #edf7f0;
        color: var(--success);
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .caption-text {
        font-size: 1.22rem;
        line-height: 1.7;
        color: var(--text);
        margin: 0;
        font-weight: 600;
    }

    .mini-note {
        color: var(--muted);
        font-size: 0.9rem;
        margin-top: 0.9rem;
    }

    .gallery-card {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: var(--radius-lg);
        padding: 0.85rem;
        box-shadow: 0 14px 28px rgba(24, 33, 47, 0.05);
        margin-bottom: 0.85rem;
    }

    div[data-testid="stFileUploader"] {
        background: linear-gradient(180deg, #ffffff 0%, #f7faff 100%);
        border-radius: var(--radius-lg);
        border: 1px dashed #b8c7dd;
        padding: 0.5rem;
    }

    div[data-testid="stFileUploader"] section {
        border: none !important;
        background: transparent !important;
    }

    div[data-testid="stFileUploader"] button,
    .stDownloadButton button,
    .stButton button {
        border-radius: 999px !important;
        border: none !important;
        background: linear-gradient(135deg, #1769ff 0%, #4a8dff 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 0.65rem 1.15rem !important;
        box-shadow: 0 12px 24px rgba(23, 105, 255, 0.22);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    div[data-testid="stFileUploader"] button:hover,
    .stDownloadButton button:hover,
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 28px rgba(23, 105, 255, 0.28);
    }

    div[data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.7);
        padding: 0.35rem;
        border-radius: 999px;
        border: 1px solid var(--line);
    }

    button[data-baseweb="tab"] {
        border-radius: 999px !important;
        color: var(--muted) !important;
        font-weight: 700 !important;
        padding: 0.7rem 1rem !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: #ffffff !important;
        color: var(--accent) !important;
        box-shadow: 0 8px 20px rgba(24, 33, 47, 0.08);
    }

    div[data-testid="stMetric"] {
        background: var(--panel-soft);
        border: 1px solid var(--line);
        border-radius: var(--radius-lg);
        padding: 0.9rem 1rem;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    div[data-testid="stMetricValue"] {
        color: var(--text);
    }

    .footer-note {
        color: var(--muted);
        text-align: center;
        font-size: 0.9rem;
        margin-top: 1.4rem;
    }

    @media (max-width: 900px) {
        .hero-title {
            font-size: 2.2rem;
        }

        .step-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


def image_bytes(uploaded_file) -> bytes:
    uploaded_file.seek(0)
    return uploaded_file.read()


def show_preview(image_data: bytes, caption: str) -> None:
    image = Image.open(BytesIO(image_data))
    st.image(image, use_container_width=True)
    st.markdown(
        f"""
        <div class="result-card">
            <div class="caption-pill">Generated Caption</div>
            <p class="caption-text">{caption}</p>
            <p class="mini-note">Caption created from the detected visual feature backbone and token sequence model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="eyebrow">Light Mode Studio</div>
            <h1 class="hero-title">Turn images into clean, fast captions.</h1>
            <p class="hero-subtitle">
                A polished captioning workspace with a lighter visual system, clearer steps,
                richer feedback, and smoother single-image and batch flows.
            </p>
            <div class="chip-row">
                <div class="metric-chip"><b>Backbone</b> {backbone_label}</div>
                <div class="metric-chip"><b>Feature Size</b> {feature_dim}</div>
                <div class="metric-chip"><b>Caption Length</b> {runtime['max_caption_length']} tokens max</div>
                <div class="metric-chip"><b>Model Source</b> {model_location}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_steps() -> None:
    st.markdown(
        """
        <div class="panel-card">
            <h3 class="section-title">A simpler flow</h3>
            <p class="section-subtitle">
                Upload, generate, and export without hunting through the screen.
            </p>
            <div class="step-grid">
                <div class="step-card">
                    <div class="step-number">1</div>
                    <h4 class="step-title">Drop images in</h4>
                    <p class="step-copy">Use single mode for focused captioning or batch mode for quick multi-image runs.</p>
                </div>
                <div class="step-card">
                    <div class="step-number">2</div>
                    <h4 class="step-title">Generate instantly</h4>
                    <p class="step-copy">The app keeps the backend loaded once and gives immediate visual feedback while processing.</p>
                </div>
                <div class="step-card">
                    <div class="step-number">3</div>
                    <h4 class="step-title">Review and export</h4>
                    <p class="step-copy">Inspect image-by-image results, then download the generated captions as plain text.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_single_mode() -> None:
    st.markdown(
        """
        <div class="panel-card">
            <h3 class="section-title">Single Image</h3>
            <p class="section-subtitle">Best when you want one polished caption with a large preview and a focused result panel.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.92, 1.08], gap="large")

    with left:
        uploaded_image = st.file_uploader(
            "Upload one image",
            type=["jpg", "jpeg", "png"],
            key="single_image_uploader",
            help="Drag and drop a JPG or PNG image here.",
        )

        st.markdown(
            """
            <div class="panel-card">
                <h3 class="section-title">What happens next</h3>
                <p class="section-subtitle">
                    Once you click generate, the app extracts visual features,
                    feeds them into the caption model, and renders the final text
                    beside the image.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        generate = st.button(
            "Generate caption",
            key="generate_single_caption",
            use_container_width=True,
            disabled=uploaded_image is None,
        )

    with right:
        if uploaded_image is None:
            st.markdown(
                """
                <div class="panel-card">
                    <h3 class="section-title">Preview</h3>
                    <p class="section-subtitle">
                        Add an image to see a full preview and caption card here.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            image_data = image_bytes(uploaded_image)
            preview = Image.open(BytesIO(image_data))
            st.markdown('<div class="gallery-card">', unsafe_allow_html=True)
            st.image(preview, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if generate:
                with st.spinner("Creating caption..."):
                    caption = captioner.generate_caption(BytesIO(image_data))

                if caption and caption.strip():
                    show_preview(image_data, caption)
                else:
                    st.error("The model failed to generate a caption for this image. This usually happens if the visual features don't match the expected patterns.")


def render_batch_mode() -> None:
    st.markdown(
        """
        <div class="panel-card">
            <h3 class="section-title">Batch Processing</h3>
            <p class="section-subtitle">Upload several images, preview the set, then generate captions in one run.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_image_uploader",
        help="Ideal for quickly captioning a small collection.",
    )

    if not uploaded_files:
        st.markdown(
            """
            <div class="panel-card">
                <h3 class="section-title">Batch preview</h3>
                <p class="section-subtitle">
                    Add multiple files to unlock a visual gallery, progress summary, and downloadable output.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    image_payloads = []
    for file in uploaded_files:
        image_payloads.append(
            {
                "name": file.name,
                "data": image_bytes(file),
            }
        )

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Uploaded", len(image_payloads))
    metric_col2.metric("Mode", "Batch")
    metric_col3.metric("Backbone", backbone_label)

    preview_columns = st.columns(min(3, len(image_payloads)))
    for index, payload in enumerate(image_payloads[:3]):
        with preview_columns[index % len(preview_columns)]:
            st.markdown('<div class="gallery-card">', unsafe_allow_html=True)
            st.image(Image.open(BytesIO(payload["data"])), use_container_width=True)
            st.caption(payload["name"])
            st.markdown("</div>", unsafe_allow_html=True)

    if len(image_payloads) > 3:
        st.caption(f"{len(image_payloads) - 3} additional image(s) are queued.")

    generate_batch = st.button(
        "Generate captions for all images",
        key="generate_batch_captions",
        use_container_width=True,
    )

    if not generate_batch:
        return

    progress_bar = st.progress(0, text="Preparing batch...")
    results = []

    for index, payload in enumerate(image_payloads):
        current_num = index + 1
        total_num = len(image_payloads)
        progress_bar.progress(
            int(index / total_num * 100),
            text=f"Processing {current_num} of {total_num}: {payload['name']}",
        )
        caption = captioner.generate_caption(BytesIO(payload["data"]))
        is_success = caption is not None and bool(caption.strip())
        results.append(
            {
                "index": current_num,
                "name": payload["name"],
                "data": payload["data"],
                "caption": caption if is_success else None,
                "success": is_success,
            }
        )

    progress_bar.progress(100, text="Batch complete.")

    success_count = sum(1 for result in results if result["success"])
    failure_count = len(results) - success_count
    st.markdown(
        """
        <div class="panel-card">
            <h3 class="section-title">Batch results</h3>
            <p class="section-subtitle">Review every caption below and export the successful ones when you are done.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("Successful", success_count)
    summary_col2.metric("Failed", failure_count)
    summary_col3.metric("Total", len(results))

    export_text = []
    for result in results:
        image_col, text_col = st.columns([0.7, 1.3], gap="large")
        with image_col:
            st.markdown('<div class="gallery-card">', unsafe_allow_html=True)
            st.image(Image.open(BytesIO(result["data"])), use_container_width=True)
            st.caption(result["name"])
            st.markdown("</div>", unsafe_allow_html=True)
        with text_col:
            if result["success"]:
                export_text.append(f"{result['name']}: {result['caption']}")
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="caption-pill">Image {result['index']}</div>
                        <p class="caption-text">{result['caption']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.error(f"Image {result['index']} could not be captioned.")

    if export_text:
        st.download_button(
            label="Download successful captions",
            data="\n".join(export_text),
            file_name="captions.txt",
            mime="text/plain",
            use_container_width=True,
        )


render_header()

if not captioner.is_ready():
    st.error("The captioning backend is not ready.")
    for error in captioner.get_errors():
        st.error(error)
    st.info(
        "Make sure `mymodel.h5` and `tokenizer.pkl` are valid, then retrain from `image-captioner.ipynb` if needed."
    )
    st.stop()


render_steps()

single_tab, batch_tab, info_tab = st.tabs(["Single", "Batch", "Model Info"])

with single_tab:
    render_single_mode()

with batch_tab:
    render_batch_mode()

with info_tab:
    st.markdown(
        """
        <div class="panel-card">
            <h3 class="section-title">Model details</h3>
            <p class="section-subtitle">
                The app inspects the saved caption model and chooses a supported feature extractor automatically.
                That keeps inference aligned with the model artifact currently present in the repo.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("Detected backbone", backbone_label)
    info_col2.metric("Feature size", feature_dim)
    info_col3.metric("Max caption length", runtime["max_caption_length"])
    st.markdown(
        """
        <div class="panel-card">
            <h3 class="section-title">Why this UI is different</h3>
            <p class="section-subtitle">
                This version uses a full light visual system, stronger hierarchy, cleaner upload surfaces,
                smoother buttons, richer preview cards, and clearer single-versus-batch workflows.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="footer-note">Image Caption Studio - light mode - smoother review flow - export-ready output</div>',
    unsafe_allow_html=True,
)
