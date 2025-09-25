import streamlit as st
from db import run_query, get_dataframe

def models_mapping_page(action):
    st.header("🔗 Models Mapping")

    # CREATE
    if action == "Create":
        with st.form("add_mapping"):
            project_id = st.text_input("Project ID")
            model_name = st.text_input("Model Name")
            version = st.text_input("Version")
            submitted = st.form_submit_button("Add Mapping")
            if submitted:
                run_query(
                    """
                    INSERT INTO models_mapping (project_id, model_name, version) 
                    VALUES (%s,%s,%s)
                    """,
                    (project_id, model_name, version)
                )
                st.success("✅ Mapping added")

    # READ
    elif action == "Read":
        df = get_dataframe("SELECT * FROM models_mapping ORDER BY project_id DESC")
        st.dataframe(df)

    # UPDATE
    elif action == "Update":
        st.subheader("🔍 Cari Mapping untuk Update")
        keyword = st.text_input("Masukkan Project ID / Model Name / Version")

        if keyword:
            df = get_dataframe(
                """
                SELECT * FROM models_mapping 
                WHERE project_id LIKE %s OR model_name LIKE %s OR version LIKE %s
                """,
                (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%")
            )

            if df.empty:
                st.warning("⚠️ Tidak ada mapping ditemukan.")
            else:
                # biar user bisa pilih mapping kalau ada lebih dari satu
                selected_idx = st.selectbox(
                    "Pilih mapping yang ingin diupdate:",
                    options=df.index,
                    format_func=lambda i: f"Project: {df.loc[i, 'project_id']} "
                                          f"- Model: {df.loc[i, 'model_name']} "
                                          f"(v{df.loc[i, 'version']})"
                )

                row = df.loc[selected_idx]

                with st.form(f"update_{row['project_id']}_{row['model_name']}"):
                    new_proj = st.text_input("Project ID", row["project_id"])
                    new_name = st.text_input("Model Name", row["model_name"])
                    new_ver = st.text_input("Version", row["version"])
                    submitted = st.form_submit_button("Update")
                    if submitted:
                        run_query(
                            """
                            UPDATE models_mapping 
                            SET project_id=%s, model_name=%s, version=%s 
                            WHERE project_id=%s AND model_name=%s AND version=%s
                            """,
                            (new_proj, new_name, new_ver,
                             row["project_id"], row["model_name"], row["version"])
                        )
                        st.success("✅ Updated")
                        st.rerun()

    # DELETE
    elif action == "Delete":
        st.subheader("🔍 Cari Mapping untuk Delete")
        keyword = st.text_input("Masukkan Project ID / Model Name / Version")

        if keyword:
            df = get_dataframe(
                """
                SELECT * FROM models_mapping 
                WHERE project_id LIKE %s OR model_name LIKE %s OR version LIKE %s
                """,
                (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%")
            )

            if df.empty:
                st.warning("⚠️ Tidak ada mapping ditemukan.")
            else:
                selected_idx = st.selectbox(
                    "Pilih mapping yang ingin dihapus:",
                    options=df.index,
                    format_func=lambda i: f"Project: {df.loc[i, 'project_id']} "
                                          f"- Model: {df.loc[i, 'model_name']} "
                                          f"(v{df.loc[i, 'version']})"
                )

                row = df.loc[selected_idx]

                st.error(f"Anda akan menghapus → Project: **{row['project_id']}**, Model: **{row['model_name']}**, Version: {row['version']}")

                if st.button("❌ Konfirmasi Delete"):
                    run_query(
                        """
                        DELETE FROM models_mapping 
                        WHERE project_id=%s AND model_name=%s AND version=%s
                        """,
                        (row["project_id"], row["model_name"], row["version"])
                    )
                    st.warning("❌ Deleted")
                    st.rerun()
