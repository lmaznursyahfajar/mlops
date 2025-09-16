import streamlit as st
<<<<<<< HEAD

from db import get_dataframe, run_query

=======
from db import run_query, get_dataframe
>>>>>>> ba0c469a (initial commit)

def projects_page(action):
    st.header("📂 Projects")

    # Ambil daftar distinct project_type dari DB
    types_df = get_dataframe("SELECT DISTINCT project_type FROM projects")
    project_types = types_df["project_type"].dropna().tolist()

    # Fallback jika kosong
    if not project_types:
        project_types = ["Other"]

    # CREATE
    if action == "Create":
        with st.form("add_project"):
            project_name = st.text_input("Project Name")
            project_name_model = st.text_input("Project Model Name")
            project_type = st.selectbox("Project Type", project_types)
            aspect_by = st.text_input("Aspect By")
            submitted = st.form_submit_button("Add Project")
            if submitted:
                run_query(
                    """
                    INSERT INTO projects 
                    (project_name, project_name_model, project_type, aspect_by) 
                    VALUES (%s,%s,%s,%s)
                    """,
<<<<<<< HEAD
                    (project_name, project_name_model, project_type, aspect_by),
=======
                    (project_name, project_name_model, project_type, aspect_by)
>>>>>>> ba0c469a (initial commit)
                )
                st.success("✅ Project added")
                st.rerun()

    # READ
    elif action == "Read":
        df = get_dataframe("SELECT * FROM projects ORDER BY project_id DESC")
        st.dataframe(df)

    # UPDATE
    elif action == "Update":
        st.subheader("🔍 Cari Project untuk Update")
        keyword = st.text_input("Masukkan Project Name / Model Name")

        if keyword:
            df = get_dataframe(
                """
                SELECT * FROM projects 
                WHERE project_name LIKE %s OR project_name_model LIKE %s
                """,
<<<<<<< HEAD
                (f"%{keyword}%", f"%{keyword}%"),
=======
                (f"%{keyword}%", f"%{keyword}%")
>>>>>>> ba0c469a (initial commit)
            )

            if df.empty:
                st.warning("⚠️ Tidak ada project ditemukan.")
            else:
                row = df.iloc[0]
<<<<<<< HEAD
                st.info(
                    f"Project ditemukan → **{row['project_name']}** (Model: {row['project_name_model']})"
                )
=======
                st.info(f"Project ditemukan → **{row['project_name']}** (Model: {row['project_name_model']})")
>>>>>>> ba0c469a (initial commit)

                with st.form(f"update_{row['project_id']}"):
                    new_name = st.text_input("Project Name", row["project_name"])
                    new_model = st.text_input("Model Name", row["project_name_model"])
                    new_type = st.selectbox(
<<<<<<< HEAD
                        "Project Type",
                        project_types,
                        index=(
                            project_types.index(row["project_type"])
                            if row["project_type"] in project_types
                            else 0
                        ),
=======
                        "Project Type", 
                        project_types,
                        index=project_types.index(row["project_type"]) if row["project_type"] in project_types else 0
>>>>>>> ba0c469a (initial commit)
                    )
                    new_aspect = st.text_input("Aspect By", row["aspect_by"])
                    submitted = st.form_submit_button("Update")
                    if submitted:
                        run_query(
                            """
                            UPDATE projects 
                            SET project_name=%s, project_name_model=%s, project_type=%s, aspect_by=%s 
                            WHERE project_id=%s
                            """,
<<<<<<< HEAD
                            (
                                new_name,
                                new_model,
                                new_type,
                                new_aspect,
                                row["project_id"],
                            ),
=======
                            (new_name, new_model, new_type, new_aspect, row["project_id"])
>>>>>>> ba0c469a (initial commit)
                        )
                        st.success("✅ Updated")
                        st.rerun()

    # DELETE
    elif action == "Delete":
        st.subheader("🔍 Cari Project untuk Delete")
        keyword = st.text_input("Masukkan Project Name / Model Name")

        if keyword:
            df = get_dataframe(
                """
                SELECT * FROM projects 
                WHERE project_name LIKE %s OR project_name_model LIKE %s
                """,
<<<<<<< HEAD
                (f"%{keyword}%", f"%{keyword}%"),
=======
                (f"%{keyword}%", f"%{keyword}%")
>>>>>>> ba0c469a (initial commit)
            )

            if df.empty:
                st.warning("⚠️ Tidak ada project ditemukan.")
            else:
                row = df.iloc[0]
<<<<<<< HEAD
                st.error(
                    f"Anda akan menghapus → **{row['project_name']}** (Model: {row['project_name_model']})"
                )

                if st.button("❌ Konfirmasi Delete"):
                    run_query(
                        "DELETE FROM projects WHERE project_id=%s", (row["project_id"],)
                    )
=======
                st.error(f"Anda akan menghapus → **{row['project_name']}** (Model: {row['project_name_model']})")

                if st.button("❌ Konfirmasi Delete"):
                    run_query("DELETE FROM projects WHERE project_id=%s", (row["project_id"],))
>>>>>>> ba0c469a (initial commit)
                    st.warning("❌ Deleted")
                    st.rerun()
