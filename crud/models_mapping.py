import streamlit as st
<<<<<<< HEAD

from db import get_dataframe, run_query

=======
from db import run_query, get_dataframe
>>>>>>> ba0c469a (initial commit)

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
<<<<<<< HEAD
                    (project_id, model_name, version),
=======
                    (project_id, model_name, version)
>>>>>>> ba0c469a (initial commit)
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
<<<<<<< HEAD
                (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"),
=======
                (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%")
>>>>>>> ba0c469a (initial commit)
            )

            if df.empty:
                st.warning("⚠️ Tidak ada mapping ditemukan.")
            else:
                row = df.iloc[0]
<<<<<<< HEAD
                st.info(
                    f"Mapping ditemukan → Project: **{row['project_id']}**, Model: **{row['model_name']}**, Version: {row['version']}"
                )
=======
                st.info(f"Mapping ditemukan → Project: **{row['project_id']}**, Model: **{row['model_name']}**, Version: {row['version']}")
>>>>>>> ba0c469a (initial commit)

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
                            WHERE project_id=%s AND model_name=%s
                            """,
<<<<<<< HEAD
                            (
                                new_proj,
                                new_name,
                                new_ver,
                                row["project_id"],
                                row["model_name"],
                            ),
=======
                            (new_proj, new_name, new_ver, row["project_id"], row["model_name"])
>>>>>>> ba0c469a (initial commit)
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
<<<<<<< HEAD
                (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"),
=======
                (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%")
>>>>>>> ba0c469a (initial commit)
            )

            if df.empty:
                st.warning("⚠️ Tidak ada mapping ditemukan.")
            else:
                row = df.iloc[0]
<<<<<<< HEAD
                st.error(
                    f"Anda akan menghapus → Project: **{row['project_id']}**, Model: **{row['model_name']}**, Version: {row['version']}"
                )

                if st.button("❌ Konfirmasi Delete"):
                    run_query(
                        "DELETE FROM models_mapping WHERE project_id=%s AND model_name=%s",
                        (row["project_id"], row["model_name"]),
=======
                st.error(f"Anda akan menghapus → Project: **{row['project_id']}**, Model: **{row['model_name']}**, Version: {row['version']}")

                if st.button("❌ Konfirmasi Delete"):
                    run_query(
                        "DELETE FROM models_mapping WHERE project_id=%s AND model_name=%s", 
                        (row["project_id"], row["model_name"])
>>>>>>> ba0c469a (initial commit)
                    )
                    st.warning("❌ Deleted")
                    st.rerun()
