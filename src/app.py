import streamlit as st
import ui.volatility_and_options_pricing as voap

def main():
    st.set_page_config(
        page_title="Volatility and Options Pricing"
    )
    voap.render()


if __name__ == "__main__":
    main()