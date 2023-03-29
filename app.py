import pickle
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from scipy.stats import poisson, nbinom

# Load the saved XGBoost model
with open('xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)


# Define a function to preprocess the input data
def preprocess_data(home_xG, away_xG):
    # Create a dictionary with the input data
    input_data = {'home_xG': [home_xG], 'away_xG': [away_xG]}
    # Convert the dictionary to a pandas DataFrame
    input_df = pd.DataFrame.from_dict(input_data)
    # Return the preprocessed data
    return input_df


def zero_inflated_poisson_model(home_xg, away_xg, theta=0.08):
    # Define the maximum number of goals to consider
    max_goals = 10
    # Create a list of possible goals from 0 to max_goals
    goals = np.arange(max_goals + 1)
    # Create a matrix of home and away goal probabilities
    home_probs = [(1 - theta) * poisson.pmf(i, home_xg) + theta * nbinom.pmf(i, n=theta, p=home_xg / (home_xg + theta))
                  for i in goals]
    away_probs = [(1 - theta) * poisson.pmf(i, away_xg) + theta * nbinom.pmf(i, n=theta, p=away_xg / (away_xg + theta))
                  for i in goals]
    # Calculate the joint probabilities of each scoreline
    scoreline_probs = np.outer(home_probs, away_probs)
    # Create a DataFrame to store the scoreline probabilities
    scorelines = pd.DataFrame(scoreline_probs, index=goals, columns=goals)
    # Add text labels to the heatmap
    fig, ax = plt.subplots()
    heatmap = ax.imshow(scorelines, cmap='Reds', vmin=0, vmax=0.15)
    ax.set_xlabel('Away Goals')
    ax.set_ylabel('Home Goals')
    ax.set_xticks(np.arange(max_goals + 1))
    ax.set_yticks(np.arange(max_goals + 1))
    ax.set_xticklabels(np.arange(max_goals + 1), fontsize=8)
    ax.set_yticklabels(np.arange(max_goals + 1), fontsize=8)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.colorbar(heatmap, ax=ax, label='Probability')
    # Add text labels to the heatmap
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            text = ax.text(j, i, round(scorelines.iloc[i, j], 3),
                           ha='center', va='center', color='black', fontsize=8, rotation=45)
    # Add a title to the plot
    ax.set_title('Scoreline Probabilities')
    # Display the plot
    st.pyplot(fig)

    # calculate the sums of home wins, draws, and away wins
    home_wins = np.sum(np.tril(scorelines, -1))
    draws = np.sum(np.diag(scorelines))
    away_wins = np.sum(np.triu(scorelines, 1))

    st.write(f'Zero Inflated Poisson match probabilities:', 'Home', round(home_wins, 4), 'Draw', round(draws, 4),
             'Away', round(away_wins, 4))

    sum_probs = round(home_wins, 4) + round(draws, 4) + round(away_wins, 4)
    sum_probs_rounded = round(sum_probs, 4)
    adjustment = 1 - sum_probs_rounded

    st.write(f'Zero Inflated Poisson match probabilities summed:', sum_probs_rounded + adjustment)
    # Return the scorelines DataFrame
    return scorelines


# Define the Streamlit app
def main():
    # Set the app title
    st.title('xG Match Result Prediction App')
    # Add a text box to the app
    text = st.text('Using known xG figures for a match, enter them into the boxes below.\n'
                   'The first prediction is made from an XGBoost Regressor trained on a dataset of 17000 rows from '
                   'understat.\n'
                   'There is a clear element of home advantage derived from the model as can be seen with similiar xG values.\n'
                   'The second prediction uses a ZIF Poisson model with a theta of 0.08 to account for the 0-0.')

    # Add input fields for home and away xG
    home_xg = st.number_input('Home xG', min_value=0.0, max_value=6.0, step=0.05)
    away_xg = st.number_input('Away xG', min_value=0.0, max_value=6.0, step=0.05)
    # Add a button to trigger the prediction
    if st.button('Predict'):
        if home_xg == 0.0 or away_xg == 0.0:
            st.error('Please enter an xG value for both teams.')
        else:
            # Preprocess the input data
            input_data = preprocess_data(home_xg, away_xg)
            # Make the prediction
            prediction = xgb_model.predict(input_data)[0]
            # Normalize the prediction
            prediction_normalized = prediction / sum(prediction)
            st.success("The predicted match outcomes are:\nHome team: {:.4f}\nDraw: {:.4f}\nAway team: {:.4f}".format(
                prediction_normalized[0], prediction_normalized[1], prediction_normalized[2]))
            st.success("The summed match outcomes are: {:.4f}".format(sum(prediction_normalized)))

        # Make the Poisson model prediction
        poisson_prediction = zero_inflated_poisson_model(home_xg, away_xg)


# Run the app
if __name__ == '__main__':
    main()
