import streamlit as st
from db import run_query, get_dataframe

def absa_inference_models_page(action):
    st.header("🤖 ABSA Inference Models")

    if action == "Create":
        with st.form("add_model"):
            version = st.text_input("Version")
            name = st.text_input("Model Name")
            description = st.text_area("Description")
            path = st.text_input("Path (ONNX)")
            submitted = st.form_submit_button("Add Model")
            if submitted:
                run_query(
                    """
                    INSERT INTO absa_inference_models 
                    (version, name, description, path, created_at, updated_at) 
                    VALUES (%s,%s,%s,%s,NOW(),NOW())
                    """,
                    (version, name, description, path)
                )
                st.success("✅ Model added")

    elif action == "Read":
        df = get_dataframe("SELECT * FROM absa_inference_models ORDER BY created_at DESC")
        st.dataframe(df)

    elif action == "Update":
        st.subheader("🔍 Cari Model untuk Update")
        keyword = st.text_input("Masukkan nama model / version")
        
        if keyword:
            df = get_dataframe(
                "SELECT * FROM absa_inference_models WHERE name LIKE %s OR version LIKE %s",
                (f"%{keyword}%", f"%{keyword}%")
            )

            if df.empty:
                st.warning("⚠️ Tidak ada model ditemukan.")
            else:
                row = df.iloc[0]  # ambil hasil pertama
                st.info(f"Model ditemukan: {row['name']} (v{row['version']})")

                with st.form(f"update_{row['uuid']}"):
                    new_ver = st.text_input("Version", row["version"])
                    new_name = st.text_input("Model Name", row["name"])
                    new_desc = st.text_area("Description", row["description"])
                    new_path = st.text_input("Path", row["path"])
                    submitted = st.form_submit_button("Update")
                    if submitted:
                        run_query(
                            """
                            UPDATE absa_inference_models 
                            SET version=%s, name=%s, description=%s, path=%s, updated_at=NOW() 
                            WHERE uuid=%s
                            """,
                            (new_ver, new_name, new_desc, new_path, row["uuid"])
                        )
                        st.success("✅ Updated")

    elif action == "Delete":
        st.subheader("🔍 Cari Model untuk Delete")
        keyword = st.text_input("Masukkan nama model / version")
        
        if keyword:
            df = get_dataframe(
                "SELECT * FROM absa_inference_models WHERE name LIKE %s OR version LIKE %s",
                (f"%{keyword}%", f"%{keyword}%")
            )

            if df.empty:
                st.warning("⚠️ Tidak ada model ditemukan.")
            else:
                row = df.iloc[0]
                st.error(f"Anda akan menghapus: {row['name']} (v{row['version']})")

                if st.button("❌ Konfirmasi Delete"):
                    run_query("DELETE FROM absa_inference_models WHERE uuid=%s", (row["uuid"],))
                    st.warning("❌ Deleted")
