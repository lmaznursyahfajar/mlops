import streamlit as st
from db import run_query, get_dataframe

def object_to_aspect_page(action):
    st.header("🧩 Object to Aspect")

    # CREATE
    if action == "Create":
        with st.form("add_object_aspect"):
            project_id = st.text_input("Project ID")
            object_id = st.text_input("Object ID")
            object_name = st.text_input("Object Name")
            aspect = st.text_input("Aspect")
            submitted = st.form_submit_button("Add Mapping")
            if submitted:
                run_query(
                    """
                    INSERT INTO object_to_aspect 
                    (project_id, object_id, object_name, aspect) 
                    VALUES (%s,%s,%s,%s)
                    """,
                    (project_id, object_id, object_name, aspect)
                )
                st.success("✅ Mapping added")

    # READ
    elif action == "Read":
        df = get_dataframe("SELECT * FROM object_to_aspect ORDER BY project_id, object_id")
        st.dataframe(df)

    # UPDATE
    elif action == "Update":
        st.subheader("🔍 Cari Object untuk Update")
        keyword = st.text_input("Masukkan Object Name / Project ID / Object ID")

        if keyword:
            df = get_dataframe(
                """
                SELECT * FROM object_to_aspect 
                WHERE object_name LIKE %s OR project_id LIKE %s OR object_id LIKE %s
                """,
                (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%")
            )

            if df.empty:
                st.warning("⚠️ Tidak ada data ditemukan.")
            else:
                row = df.iloc[0]
                st.info(f"Data ditemukan → **{row['object_name']}** (Project: {row['project_id']}, Object ID: {row['object_id']})")

                with st.form(f"update_{row['object_id']}"):
                    new_object_name = st.text_input("Object Name", row["object_name"])
                    new_aspect = st.text_input("Aspect", row["aspect"])
                    submitted = st.form_submit_button("Update")
                    if submitted:
                        run_query(
                            """
                            UPDATE object_to_aspect 
                            SET object_name=%s, aspect=%s 
                            WHERE project_id=%s AND object_id=%s
                            """,
                            (new_object_name, new_aspect, row["project_id"], row["object_id"])
                        )
                        st.success("✅ Updated")
                        st.rerun()

    # DELETE
    elif action == "Delete":
        st.subheader("🔍 Cari Object untuk Delete")
        keyword = st.text_input("Masukkan Object Name / Project ID / Object ID")

        if keyword:
            df = get_dataframe(
                """
                SELECT * FROM object_to_aspect 
                WHERE object_name LIKE %s OR project_id LIKE %s OR object_id LIKE %s
                """,
                (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%")
            )

            if df.empty:
                st.warning("⚠️ Tidak ada data ditemukan.")
            else:
                row = df.iloc[0]
                st.error(f"Anda akan menghapus → **{row['object_name']}** (Project: {row['project_id']}, Object ID: {row['object_id']})")

                if st.button("❌ Konfirmasi Delete"):
                    run_query(
                        "DELETE FROM object_to_aspect WHERE project_id=%s AND object_id=%s",
                        (row["project_id"], row["object_id"])
                    )
                    st.warning("❌ Deleted")
                    st.rerun()
