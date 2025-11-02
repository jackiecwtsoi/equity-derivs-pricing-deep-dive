import streamlit as st
import ui.volatility_surface as volsfc

def main():
    st.set_page_config(
        page_title="Volatility and Options Pricing"
    )
    volsfc.render()


if __name__ == "__main__":
    main()