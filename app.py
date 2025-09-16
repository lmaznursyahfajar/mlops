import streamlit as st

from crud.absa_inference_models import absa_inference_models_page
from crud.models_mapping import models_mapping_page
from crud.object_to_aspect import object_to_aspect_page
from crud.projects import projects_page

# ======================
# 🎨 Custom Styling
# ======================
st.set_page_config(
    page_title="Sentiment MLOps Dashboard", page_icon="📊", layout="wide"
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        .css-1d391kg {padding-top: 0rem;} /* Hilangin padding atas sidebar */
    </style>
""",
    unsafe_allow_html=True,
)

# ======================
# 🧭 Sidebar: Pilih Tabel & Aksi
# ======================
st.sidebar.title("📊 Sentiment MLOps")

# Dropdown untuk tabel
menu_table = st.sidebar.selectbox(
    "Pilih Tabel:",
    ["Projects", "Absa Inference Models", "Models Mappings", "Object to Aspect"],
)

# Dropdown untuk aksi CRUD
menu_action = st.sidebar.selectbox(
    "Pilih Aksi:", ["Create", "Read", "Update", "Delete"], index=1
)

# ======================
# 🔗 Routing
# ======================
st.title("🚀 Sentiment MLOps Dashboard")
st.caption(
    "Kelola tabel database dengan mudah: Projects, Models, Mappings, dan Object to Aspect"
)

col1, col2 = st.columns([3, 2])

with col1:
    st.info(f"📂 **Tabel aktif:** {menu_table}")
with col2:
    st.success(f"🛠️ **Aksi:** {menu_action}")

st.markdown("---")

# Panggil halaman sesuai tabel
if menu_table == "Projects":
    projects_page(menu_action)
elif menu_table == "Absa Inference Models":
    absa_inference_models_page(menu_action)
elif menu_table == "Models Mappings":
    models_mapping_page(menu_action)
elif menu_table == "Object to Aspect":
    object_to_aspect_page(menu_action)
