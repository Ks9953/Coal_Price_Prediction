import pandas as pd
import numpy as np
import streamlit as st
from scipy.optimize import minimize

# Sample dataset
data = {
    'Mines Name': ['Jagannath', 'Ananta', 'Bhubaneswari', 'Kaniha', 'Lingaraj', 'Balram', 'Hingula', 'Bharatpur'],
    'Landed Cost': [2733, 2723, 2728, 2776, 2886, 2669, 2531, 2742],
    'Expected GCV for Auction': [3400, 3050, 3200, 3200, 3300, 3150, 2700, 3200]
}

# Load data into a DataFrame
df = pd.DataFrame(data)

# Streamlit app
st.title("Coal Auction Target Quantity Optimizer")

# Input fields
total_quantity = st.number_input(
    'Enter the total quantity to be purchased (in tons):', min_value=1000, value=10000)
target_rs_per_gcv = st.number_input(
    'Enter the target Rs per GCV:', min_value=0.0, value=0.5)
selected_mines = st.multiselect(
    'Select Mines for distribution', df['Mines Name'].tolist(), default=df['Mines Name'].tolist())

# Filter selected mines
filtered_df = df[df['Mines Name'].isin(selected_mines)].reset_index(drop=True)

# Allow the user to modify Landed Cost and Expected GCV for each selected mine
st.subheader('Adjust Landed Cost and Expected GCV for Auction for each mine:')
for idx, row in filtered_df.iterrows():
    st.write(f"**{row['Mines Name']}**")
    new_landed_cost = st.number_input(
        f"Landed Cost for {row['Mines Name']}", value=float(row['Landed Cost']), key=f'lc_{row["Mines Name"]}')
    new_expected_gcv = st.number_input(
        f"Expected GCV for Auction for {row['Mines Name']}", value=float(row['Expected GCV for Auction']), key=f'gcv_{row["Mines Name"]}')
    # Update the DataFrame with the new values
    filtered_df.at[idx, 'Landed Cost'] = new_landed_cost
    filtered_df.at[idx, 'Expected GCV for Auction'] = new_expected_gcv

# Recalculate Rs per GCV with the updated values
filtered_df['Rs per GCV'] = filtered_df['Landed Cost'] / filtered_df['Expected GCV for Auction']

# Extract costs and GCV values
costs = filtered_df['Landed Cost'].values
gcv_values = filtered_df['Expected GCV for Auction'].values
rs_per_gcv = filtered_df['Rs per GCV'].values

# Objective: Minimize the average landed cost while meeting the target Rs per GCV
def objective(x):
    """
    Objective function to minimize the average landed cost.
    """
    # Calculate the total cost and the total quantities distributed
    total_cost = np.dot(x, costs)
    avg_landed_cost = total_cost / np.sum(x)  # Average landed cost
    return avg_landed_cost

# Constraints
constraints = [
    {'type': 'eq', 'fun': lambda x: np.sum(x) - total_quantity},
    {'type': 'eq', 'fun': lambda x: (np.dot(x, rs_per_gcv) / np.sum(x)) - target_rs_per_gcv}
]

# Bounds: each mine's allocation must be non-negative and cannot exceed total_quantity
bounds = [(0, total_quantity) for _ in range(len(filtered_df))]

# Initial guess: equal distribution of quantities among the mines
initial_guess = [total_quantity / len(filtered_df)] * len(filtered_df)

# Run the optimization
result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)

# Display the result
if result.success:
    optimized_quantities = result.x
    avg_landed_cost = np.dot(optimized_quantities, costs) / np.sum(optimized_quantities)
    avg_rs_per_gcv = np.dot(optimized_quantities, rs_per_gcv) / np.sum(optimized_quantities)

    # Display the optimized quantities
    st.subheader("Optimized Target Quantities per Mine:")
    for mine, quantity in zip(filtered_df['Mines Name'], optimized_quantities):
        st.write(f"{mine}: {quantity:.2f} tons")

    # Display the average landed cost and Rs per GCV
    st.subheader(f"Average Landed Cost: ₹{avg_landed_cost:.2f} per ton")
    st.subheader(f"Achieved Rs per GCV: ₹{avg_rs_per_gcv:.5f} (Rs/GCV)")
else:
    st.error("Optimization failed. Please try again with different inputs.")
