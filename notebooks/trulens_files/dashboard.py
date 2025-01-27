import streamlit as st
import plotly.express as px

def main():
    st.title("Example Dashboard")
    st.write("This dashboard visualizes key metrics.")
    df = px.data.gapminder()  # Example dataset
    fig = px.scatter(
        df, x="gdpPercap", y="lifeExp", color="continent", hover_name="country",
        log_x=True, size_max=60
    )
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
