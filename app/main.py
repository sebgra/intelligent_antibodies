import streamlit as st

st.set_page_config(page_title="Intelligent Antibodies Dashboard", layout="wide")

# Inject custom CSS to style the tabs

# This CSS styles all st.tabs components on the page
st.markdown(
    """
    <style>
        /* Center the tab headers */
        .stTabs [data-baseweb="tab-list"] {
            justify-content: center;
            gap: 20px;
        }

        /* Color the active (selected) tab text */
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] p {
            color: #008080;
        }

        /* Color the inactive tab text */
        .stTabs [data-baseweb="tab-list"] button[aria-selected="false"] p {
            color: #808080;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def count_a_in_sequence(sequence):
    """Counts the number of 'A's in a given string."""
    count = sequence.count('A')
    st.write(f"The sequence '{sequence}' has {count} 'A's.")
    return count

# Group the subheader and caption together in a container
col1, col2, col3 = st.columns([0.2,0.15, 0.85], gap="small", vertical_alignment="center")

# Use a container to group the image and subheader, then a separate call for the caption
with st.container():

    with col2:
        st.image("graphics/logo.png", width=200)

    with col3:
        st.markdown("<h2 style='color: #008080;'>Intelligent Antibodies</h2>", unsafe_allow_html=True)

# --- Your tabs with the new styling ---
with st.container():
    tab1, tab2, tab3 = st.tabs(["Generate antibodies", "Display antibodies", "About"])

#Input container
with tab1:
    with st.container(border = True):
        antibody_input = st.text_input(label='Antigen input', placeholder="Enter your amino acid sequence, e.g. ARYTTTA")
        st.text("or select local fasta file")
        uploaded_file = st.file_uploader(key='antibody_search_uploader', label = "Choose a file", accept_multiple_files=False)
        anitbody_search_button = st.button(key='antibody_search_button', label = "Launch generation")
        
        # Check the button's state in a separate if statement
        if anitbody_search_button:
            if antibody_input:
                count_a_in_sequence(antibody_input)
            else:
                st.warning("Please enter a sequence.")
                
with tab2:
    title = st.text_input(label='Antibody input', placeholder="Enter your amino acid sequence, e.g. ARYTTTA")
    st.text("or select local fasta file")
    uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=False)
    complex_trigger_button = st.button(label = "Launch generation")
      
with tab3:
    st.text("""Therapeutic (monoclonal) antibodies are one of the most effective therapies available today for the treatment of chronic inflammatory diseases such as Crohn's disease, lupus and multiple sclerosis. To treat the latter, monoclonal antibodies can target certain proteins involved in these pathologies with a view to neutralizing them, and can also be used to limit the supply of factors essential to tumor growth or disruptors of the tumor microenvironment. Monoclonal antibody-based serotherapy can also compensate for treatment shortfalls in the case of fulminant epidemics where the pathogens involved have a high mutability rate, such as COVID-19.

Although promising and a major product on the pharmaceutical market, only around thirty monoclonal antibodies are currently available for chronic inflammatory diseases, and around ten for the treatment of cancer. This lack of comprehensiveness is due to the many difficulties inherent in the in-vitro and in-silico design of these therapeutic molecules. Antibody design and/or optimization remains a real challenge, not least because of the need to produce molecules that are effective, target-specific and deliverable to the organs being treated. The difficulties are also linked to long and costly development times.

In order to accelerate the development of therapeutic antibodies, in-silico methods have been developed to reduce modeling times for these molecules, while exploring design possibilities more exhaustively. Although advantageous, these methods currently rely essentially on estimating the affinity between the antibody and its target by calculating the binding energy, which remains difficult to estimate and extremely time-consuming from an experimental point of view.
            """)
    # st.markdown('<div style="text-align: justify;">Therapeutic (monoclonal) antibodies are one of the most effective therapies available today for the treatment of chronic inflammatory diseases such as Crohn\'s disease, lupus and multiple sclerosis. To treat the latter, monoclonal antibodies can target certain proteins involved in these pathologies with a view to neutralizing them, and can also be used to limit the supply of factors essential to tumor growth or disruptors of the tumor microenvironment. Monoclonal antibody-based serotherapy can also compensate for treatment shortfalls in the case of fulminant epidemics where the pathogens involved have a high mutability rate, such as COVID-19.\n Although promising and a major product on the pharmaceutical market, only around thirty monoclonal antibodies are currently available for chronic inflammatory diseases, and around ten for the treatment of cancer. This lack of comprehensiveness is due to the many difficulties inherent in the in-vitro and in-silico design of these therapeutic molecules. Antibody design and/or optimization remains a real challenge, not least because of the need to produce molecules that are effective, target-specific and deliverable to the organs being treated. The difficulties are also linked to long and costly development times. In order to accelerate the development of therapeutic antibodies, in-silico methods have been developed to reduce modeling times for these molecules, while exploring design possibilities more exhaustively. Although advantageous, these methods currently rely essentially on estimating the affinity between the antibody and its target by calculating the binding energy, which remains difficult to estimate and extremely time-consuming from an experimental point of view.</div>', unsafe_allow_html=True)
    
    
