import streamlit as st
import chess
from scripts.engine import Engine
from scripts.deepchess import DeepChess
from scripts.autoencoder import AE
from scripts.utils import *


# todo: implement gameplay
# todo: implement openings or mimicing player moves ar first few moves
# todo: add mechanism to go revert to previous move
# todo: add comments and docstrings

MODEL_PATH = "runs/deepchess_bs512_lr0.01/epoch-100_loss-0.1638_train_acc-0.9270_dev_acc-0.9008.pth"


class Slider:
    """
    A class to create a slider and a numeric input that are synced together.

    Syncing is done by referencing the session state of the slider
    to the session state of the numeric input and vice versa.
    """

    def __init__(self, name, min_value, max_value, step, default, index=0, columns=(3, 1)):
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.default = default
        self.index = index
        self.columns = columns


    @property
    def value(self):
        col1, col2 = st.columns(self.columns)
        with col1:
            slider_val = self.get_slider_value()
        with col2:
            numeric_val = self.get_numeric_value()
        assert slider_val == numeric_val, "Slider and numeric values are not equal"
        return slider_val
    

    def get_slider_value(self):
        return st.slider(
            self.name, 
            self.min_value, 
            self.max_value, 
            self.default, 
            self.step, 
            key=f"slider_{self.index}",
            on_change=self.update_numeric
        )
    

    def get_numeric_value(self):
        return st.number_input(
            self.name,
            self.min_value, 
            self.max_value, 
            self.default, 
            self.step, 
            key=f"numeric_{self.index}",
            on_change=self.update_slider,
            label_visibility="hidden"
        )
    

    def update_slider(self):
        st.session_state[f"slider_{self.index}"] = st.session_state[f"numeric_{self.index}"]


    def update_numeric(self):
        st.session_state[f"numeric_{self.index}"] = st.session_state[f"slider_{self.index}"]


@st.cache_resource()
def load_model():
    """Load the model and the autoencoder."""

    model = DeepChess(AE().encoder)
    _ = load_state_dict(MODEL_PATH, model)
    return model


def main():
    css = '''
    <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:1.2rem;
        }
    </style>
    '''

    st.markdown(css, unsafe_allow_html=True)
    st.title("♟️ DeepChess")

    tab1, tab2, tab3 = st.tabs(["Play", "Settings", "About"])

    with tab1:
        st.subheader("Play against the engine")
        start = st.button("Start new game")

        if start:
            pass



    with tab2:
        search_depth = Slider(
            name = "Search Depth",
            min_value = 1,
            max_value = 10,
            step = 1,
            default = 3,
            index = 0
        )

        color = st.radio(
            "Player color",
            ("White", "Black"),
            index=0,
            key="color"
        )

        device = st.radio(
            "Device",
            ("CPU", "GPU"),
            index=0,
            key="device"
        )

        verbose = st.radio(
            "Logging level",
            ("Verbose", "Silent"),
            index=0,
            key="verbose"
        )


        color = True if color == "White" else False
        device = "cpu" if device == "CPU" else "cuda"
        verbose = True if verbose == "Verbose" else False
        search_depth = search_depth.value

        if search_depth == 1:
            st.info("Search depth of 1 is not recommended. The model will only evaluate the current position.")

        elif search_depth > 4:
            st.info("Search depth higher than 4 is not recommended. The search speed will decrease significantly.")

    with tab3:
        st.markdown(
            """
            This is a simple chess engine that uses a deep neural network to evaluate chess positions.
            The model was implemented in PyTorch and follows the architecture described in the [DeepChess paper](https://arxiv.org/pdf/1711.09667.pdf).
            """
        )

        st.markdown(
            """
            The model was trained for 100 epochs on over 500,000 games with Elo over 2,000 from the [computerchess.org](https://computerchess.org.uk/ccrl/4040/).
            From each game 15 positions were sampled and the model was trained to predict which position was better.
            """
        )

        st.markdown(
            """
            The model searches for the best move using the minimax algorithm with alpha-beta pruning. 
            If the search depth is set to 1, the model will only evaluate the current position.
            Despite hashing the board positions, the search is still very slow and time required to make a move increases exponentially with the search depth.
            Thus, it is recommended to set the search depth to 3 or lower.
            """
        )




if __name__ == "__main__":
    main()
