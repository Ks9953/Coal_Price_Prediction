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

# Streamlit app UI
st.title("Coal Auction Target Quantity Optimizer")

# Step 1: Inputs for total quantity and cost constraints
st.header("Step 1: Enter Auction Parameters")

total_quantity = st.number_input(
    'Total quantity to be purchased (in tons):', min_value=1000, value=10000)
target_rs_per_gcv = st.number_input(
    'Target Rs per GCV (maximum):', min_value=0.0, value=0.5)
desired_avg_landed_cost = st.number_input(
    'Desired average landed cost (maximum):', min_value=0.0, value=2700.0)

# Step 2: Mine selection and adjustment
st.header("Step 2: Select Mines and Adjust Costs")

selected_mines = st.multiselect(
    'Select Mines for distribution:', df['Mines Name'].tolist(), default=df['Mines Name'].tolist())

filtered_df = df[df['Mines Name'].isin(selected_mines)].reset_index(drop=True)

# Organize mine cost and GCV adjustment in two columns
st.subheader('Adjust Landed Cost, Expected GCV, and view Rs per GCV:')
for idx, row in filtered_df.iterrows():
    st.write(f"**{row['Mines Name']}**")
    col1, col2, col3 = st.columns(3)
    with col1:
        new_landed_cost = st.number_input(
            f"Landed Cost for {row['Mines Name']}", value=float(row['Landed Cost']), key=f'lc_{row["Mines Name"]}')
    with col2:
        new_expected_gcv = st.number_input(
            f"Expected GCV for {row['Mines Name']}", value=float(row['Expected GCV for Auction']), key=f'gcv_{row["Mines Name"]}')
    
    # Update the DataFrame with the new values
    filtered_df.at[idx, 'Landed Cost'] = new_landed_cost
    filtered_df.at[idx, 'Expected GCV for Auction'] = new_expected_gcv

    # Calculate Rs per GCV for the mine
    rs_per_gcv = new_landed_cost / new_expected_gcv

    with col3:
        st.write(f"Rs per GCV: ₹{rs_per_gcv:.5f}")

# Step 3: Optimization and results
st.header("Step 3: Optimize Target Quantities")

# Extract costs and GCV values
costs = filtered_df['Landed Cost'].values
gcv_values = filtered_df['Expected GCV for Auction'].values
rs_per_gcv = filtered_df['Landed Cost'].values / filtered_df['Expected GCV for Auction'].values

# Objective: Minimize Rs per GCV (Rs/GCV)
def objective(x):
    return np.dot(x, rs_per_gcv) / np.sum(x)

# Constraints
constraints = [
    {'type': 'eq', 'fun': lambda x: np.sum(x) - total_quantity},  # Total quantity constraint
    {'type': 'ineq', 'fun': lambda x: desired_avg_landed_cost - (np.dot(x, costs) / np.sum(x))},  # Landed cost <= desired
    {'type': 'ineq', 'fun': lambda x: target_rs_per_gcv - (np.dot(x, rs_per_gcv) / np.sum(x))}  # Rs/GCV <= target
]

# Set bounds for each mine
max_limits = {
    'Hingula': 500000,
    'Balram': 300000,
    'Bharatpur': 250000
}

bounds = [(0, min(max_limits.get(mine, total_quantity), total_quantity)) for mine in filtered_df['Mines Name']]

# Initial guess: equal distribution of quantities among the mines
initial_guess = [total_quantity / len(filtered_df)] * len(filtered_df)

# Run the optimization
result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)

# Show the optimized quantities and results
if result.success:
    optimized_quantities = result.x
    avg_landed_cost = np.dot(optimized_quantities, costs) / np.sum(optimized_quantities)
    avg_rs_per_gcv_after_opt = np.dot(optimized_quantities, rs_per_gcv) / np.sum(optimized_quantities)

    # Display the optimized quantities
    st.subheader("Optimized Quantities per Mine:")
    for mine, quantity in zip(filtered_df['Mines Name'], optimized_quantities):
        st.write(f"{mine}: {quantity:.2f} tons")

    # Display the average landed cost and Rs per GCV after optimization
    st.subheader(f"Optimized Wt.Average Landed Cost: ₹{avg_landed_cost:.2f} per ton (<= ₹{desired_avg_landed_cost})")
    st.subheader(f"Optimized Wt.Rs/MCAL: ₹{avg_rs_per_gcv_after_opt:.5f} (<= ₹{target_rs_per_gcv})")
else:
    st.error("Optimization failed. Please try again with different inputs.")
