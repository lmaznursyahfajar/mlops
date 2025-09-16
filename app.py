import streamlit as st
<<<<<<< HEAD

from crud.absa_inference_models import absa_inference_models_page
from crud.models_mapping import models_mapping_page
from crud.object_to_aspect import object_to_aspect_page
from crud.projects import projects_page
=======
from crud.projects import projects_page
from crud.absa_inference_models import absa_inference_models_page
from crud.models_mapping import models_mapping_page
from crud.object_to_aspect import object_to_aspect_page
>>>>>>> ba0c469a (initial commit)

# ======================
# 🎨 Custom Styling
# ======================
st.set_page_config(
<<<<<<< HEAD
    page_title="Sentiment MLOps Dashboard", page_icon="📊", layout="wide"
)

st.markdown(
    """
=======
    page_title="Sentiment MLOps Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
>>>>>>> ba0c469a (initial commit)
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
<<<<<<< HEAD
""",
    unsafe_allow_html=True,
)
=======
""", unsafe_allow_html=True)
>>>>>>> ba0c469a (initial commit)

# ======================
# 🧭 Sidebar: Pilih Tabel & Aksi
# ======================
st.sidebar.title("📊 Sentiment MLOps")

# Dropdown untuk tabel
menu_table = st.sidebar.selectbox(
    "Pilih Tabel:",
<<<<<<< HEAD
    ["Projects", "Absa Inference Models", "Models Mappings", "Object to Aspect"],
=======
    ["Projects", "Absa Inference Models", "Models Mappings", "Object to Aspect"]
>>>>>>> ba0c469a (initial commit)
)

# Dropdown untuk aksi CRUD
menu_action = st.sidebar.selectbox(
<<<<<<< HEAD
    "Pilih Aksi:", ["Create", "Read", "Update", "Delete"], index=1
=======
    "Pilih Aksi:",
    ["Create", "Read", "Update", "Delete"],
    index=1
>>>>>>> ba0c469a (initial commit)
)

# ======================
# 🔗 Routing
# ======================
st.title("🚀 Sentiment MLOps Dashboard")
<<<<<<< HEAD
st.caption(
    "Kelola tabel database dengan mudah: Projects, Models, Mappings, dan Object to Aspect"
)
=======
st.caption("Kelola tabel database dengan mudah: Projects, Models, Mappings, dan Object to Aspect")
>>>>>>> ba0c469a (initial commit)

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
