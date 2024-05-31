import streamlit as st
import pandas as pd
from config import RAGAS_RESULTS
from st_aggrid import AgGrid, GridOptionsBuilder


st.set_page_config(layout="wide")


@st.cache_data()
def load_data():
    df_ragas = pd.read_csv(RAGAS_RESULTS)
    return df_ragas


def renderExpDashboard(df):
    gb_exp = GridOptionsBuilder.from_dataframe(df)
    gb_exp.configure_default_column(editable=False, wrapText=True, autoHeight=True)
    gb_exp.configure_pagination(paginationAutoPageSize=True)
    gb_grid_options = gb_exp.build()

    st.title('Experiments Dashboard')
    AgGrid(df,  allow_unsafe_jscode=True, gridOptions=gb_grid_options, height=750 )


# Streamlit app
def main():
    placeholder = st.empty()
    with placeholder.container():

        # Load data
        df_ragas = load_data()

        # Check Params
        params = st.query_params
        page = "default"
        if "page" in params:
            page = params["page"][0]

        # show ragas results
        if page == "default":
            renderExpDashboard(df_ragas)


if __name__ == '__main__':
    main()
