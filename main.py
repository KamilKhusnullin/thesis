import streamlit as st
from PIL import Image
from pix2tex.cli import LatexOCR


def main():
    # Устанавливаем конфигурации страницы
    st.set_page_config(page_title="Дипломная Работа", layout="wide")

    # Пользовательские стили CSS
    st.markdown(
        """
        <style>
        .big-font {
            font-size:30px !important;
            font-weight: bold;
        }
        h1 {
            color: #ff6347;
        }
        .stApp {
            background-image: url(https://www.toptal.com/designers/subtlepatterns/patterns/memphis-mini.png);
            background-size: cover;
        }
        .reportview-container .markdown-text-container {
            font-family: monospace;
        }
        .sidebar .sidebar-content {
            background-color: #f0f0f0;
        }
        </style>
        """, unsafe_allow_html=True)

    # Заголовок приложения и информация о проекте
    st.title("Модель распознавания формул и преобразования в LaTeX код")
    st.markdown("Хуснуллин Камиль Шавкатович, студент группы 11-002", unsafe_allow_html=True)
    st.markdown("Научный руководитель: Агафонов Александр Алексеевич", unsafe_allow_html=True)

    # Sidebar for optional settings or additional information
    with st.sidebar:
        st.header("Настройки")
        st.info("Загрузите изображение формулы для преобразования в LaTeX.")

    # Main content
    uploaded_image = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"], key="1")

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Загруженное изображение", use_column_width=True)
        latex_formula = process_image(uploaded_image)
        st.subheader("Формула LaTeX:")
        st.text_area("Результат", latex_formula, height=150)
        parsed_md = parse_to_md(latex_formula)
        st.subheader("Преобразованный Markdown:")
        st.latex(f"\n{latex_formula}\n")


def process_image(image):
    img = Image.open(image)
    model = LatexOCR()
    latex_formula = model(img)
    return latex_formula


def parse_to_md(latex_formula):
    parsed_md = f"**Преобразованная формула:** *{latex_formula}*"
    return parsed_md


if __name__ == "__main__":
    main()
