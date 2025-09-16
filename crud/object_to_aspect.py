import streamlit as st
<<<<<<< HEAD

from db import get_dataframe, run_query

=======
from db import run_query, get_dataframe
>>>>>>> ba0c469a (initial commit)

def object_to_aspect_page(action):
    st.header("🧩 Object to Aspect")

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
<<<<<<< HEAD
                    (project_id, object_id, object_name, aspect),
=======
                    (project_id, object_id, object_name, aspect)
>>>>>>> ba0c469a (initial commit)
                )
                st.success("✅ Mapping added")

    elif action == "Read":
<<<<<<< HEAD
        df = get_dataframe(
            "SELECT * FROM object_to_aspect ORDER BY project_id, object_id"
        )
=======
        df = get_dataframe("SELECT * FROM object_to_aspect ORDER BY project_id, object_id")
>>>>>>> ba0c469a (initial commit)
        st.dataframe(df)

    elif action == "Update":
        st.subheader("🔍 Cari Object untuk Update")
        keyword = st.text_input("Masukkan Object Name / Project ID / Object ID")

        if keyword:
            df = get_dataframe(
                """
                SELECT * FROM object_to_aspect 
                WHERE object_name LIKE %s OR project_id LIKE %s OR object_id LIKE %s
                """,
<<<<<<< HEAD
                (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"),
=======
                (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%")
>>>>>>> ba0c469a (initial commit)
            )

            if df.empty:
                st.warning("⚠️ Tidak ada data ditemukan.")
            else:
                row = df.iloc[0]
<<<<<<< HEAD
                st.info(
                    f"Data ditemukan → **{row['object_name']}** (Project: {row['project_id']}, Object ID: {row['object_id']})"
                )
=======
                st.info(f"Data ditemukan → **{row['object_name']}** (Project: {row['project_id']}, Object ID: {row['object_id']})")
>>>>>>> ba0c469a (initial commit)

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
<<<<<<< HEAD
                            (
                                new_object_name,
                                new_aspect,
                                row["project_id"],
                                row["object_id"],
                            ),
=======
                            (new_object_name, new_aspect, row["project_id"], row["object_id"])
>>>>>>> ba0c469a (initial commit)
                        )
                        st.success("✅ Updated")
                        st.rerun()

    elif action == "Delete":
        st.subheader("🔍 Cari Object untuk Delete")
        keyword = st.text_input("Masukkan Object Name / Project ID / Object ID")

        if keyword:
            df = get_dataframe(
                """
                SELECT * FROM object_to_aspect 
                WHERE object_name LIKE %s OR project_id LIKE %s OR object_id LIKE %s
                """,
<<<<<<< HEAD
                (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"),
=======
                (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%")
>>>>>>> ba0c469a (initial commit)
            )

            if df.empty:
                st.warning("⚠️ Tidak ada data ditemukan.")
            else:
                row = df.iloc[0]
<<<<<<< HEAD
                st.error(
                    f"Anda akan menghapus → **{row['object_name']}** (Project: {row['project_id']}, Object ID: {row['object_id']})"
                )
=======
                st.error(f"Anda akan menghapus → **{row['object_name']}** (Project: {row['project_id']}, Object ID: {row['object_id']})")
>>>>>>> ba0c469a (initial commit)

                if st.button("❌ Konfirmasi Delete"):
                    run_query(
                        "DELETE FROM object_to_aspect WHERE project_id=%s AND object_id=%s",
<<<<<<< HEAD
                        (row["project_id"], row["object_id"]),
=======
                        (row["project_id"], row["object_id"])
>>>>>>> ba0c469a (initial commit)
                    )
                    st.warning("❌ Deleted")
                    st.rerun()
