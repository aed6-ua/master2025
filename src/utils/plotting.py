import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import mlflow
import io

# =============================================================================
# == FUNCTIONS FOR VISUALIZING THE FINAL TRAINED STATE (called from run_experiment.py)
# =============================================================================

def create_final_solution_plots(df, run_name, plot_amplitude):
    """
    Creates and logs plots comparing the final model prediction to the ground truth.
    This function is called once at the end of a training run.

    Args:
        df (pd.DataFrame): DataFrame containing columns ['x', 'time', 'model', 'ground_truth', 'difference'].
        run_name (str): The name of the MLflow run.
        plot_amplitude (float): The approximate max amplitude for setting y-axis ranges.
    """
    print("Creating and logging final solution visualizations...")

    # --- Animated Line Plot: Model vs. Ground Truth ---
    fig_line = px.line(
        df, x='x', y=['model', 'ground_truth'], animation_frame='time',
        title=f'Final Solution: Model vs. Ground Truth ({run_name})'
    )
    fig_line.update_layout(yaxis_range=[-plot_amplitude * 1.2, plot_amplitude * 1.2], yaxis_title="u(x,t) or Amplitude")
    mlflow.log_figure(fig_line, 'final_plots/solution_vs_time.html')

    # --- Heatmap Grid ---
    _create_final_heatmap_grid(df, run_name)

    # --- Save the final data as a CSV artifact ---
    df_path = io.StringIO()
    df.to_csv(df_path, index=True)
    mlflow.log_text(df_path.getvalue(), artifact_file=f"data_artifacts/final_predictions.csv")
    print(f"Logged final predictions dataframe to MLflow artifact: data_artifacts/final_predictions.csv")

def _create_final_heatmap_grid(df, run_name):
    """Helper function to create a 2x1 grid of heatmaps: Solution and Difference."""
    x_unique = sorted(df['x'].unique())
    t_unique = sorted(df['time'].unique())

    # Create 2D grids for plotting
    model_grid = df.pivot(index='time', columns='x', values='model').reindex(index=t_unique, columns=x_unique).values
    gt_grid = df.pivot(index='time', columns='x', values='ground_truth').reindex(index=t_unique, columns=x_unique).values
    diff_grid = df.pivot(index='time', columns='x', values='difference').reindex(index=t_unique, columns=x_unique).values

    # Determine a shared color scale for the solutions
    abs_max_sol = np.nanmax(np.abs(np.concatenate([model_grid.flatten(), gt_grid.flatten()])))
    if abs_max_sol == 0: abs_max_sol = 1.0

    # Determine a color scale for the difference map
    abs_max_diff = np.nanmax(np.abs(diff_grid))
    if abs_max_diff == 0: abs_max_diff = 0.1 * abs_max_sol

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Ground Truth', 'Model Prediction', 'Difference (Pred - GT)'),
        shared_yaxes=True
    )

    # Add Ground Truth Heatmap
    fig.add_trace(go.Heatmap(z=gt_grid, x=x_unique, y=t_unique, colorscale='RdBu_r',
                             zmin=-abs_max_sol, zmax=abs_max_sol), row=1, col=1)

    # Add Model Prediction Heatmap
    fig.add_trace(go.Heatmap(z=model_grid, x=x_unique, y=t_unique, colorscale='RdBu_r',
                             zmin=-abs_max_sol, zmax=abs_max_sol, showscale=False), row=1, col=2)

    # Add Difference Heatmap
    fig.add_trace(go.Heatmap(z=diff_grid, x=x_unique, y=t_unique, colorscale='RdBu_r',
                             zmin=-abs_max_diff, zmax=abs_max_diff, zmid=0, colorbar=dict(title="Error")), row=1, col=3)

    fig.update_layout(
        title_text=f'Final Solution and Difference Heatmaps ({run_name})',
        height=400, width=1200
    )
    fig.update_yaxes(autorange="reversed")
    mlflow.log_figure(fig, 'final_plots/solution_and_difference_heatmaps.html')


# =============================================================================
# == FUNCTIONS FOR OFFLINE ANALYSIS (called from a Jupyter notebook)
# =============================================================================

def plot_training_animation_from_artifact(run_id, ground_truth_df=None, y_range=None):
    """
    Loads prediction history from an MLflow run artifact and creates an animation.

    Args:
        run_id (str): The MLflow run ID.
        ground_truth_df (pd.DataFrame, optional): DataFrame with final ground truth for comparison.
        y_range (list, optional): The y-axis range, e.g., [-10, 10].

    Returns:
        A Plotly Figure object.
    """
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    run_name = run.data.tags.get('mlflow.runName', run_id)
    
    # Find the artifact path
    artifacts = client.list_artifacts(run_id, "data_artifacts")
    history_artifact = next((a for a in artifacts if "prediction_history" in a.path), None)

    if not history_artifact:
        print(f"Error: Could not find prediction_history artifact for run {run_id}")
        return None

    # Download and load the CSV
    local_path = client.download_artifacts(run_id, history_artifact.path)
    df_hist = pd.read_csv(local_path)
    
    # Merge with ground truth if provided
    if ground_truth_df is not None:
        # We only need the x, time, and ground_truth columns
        gt_slice = ground_truth_df[['x', 'time', 'ground_truth']]
        df_hist = pd.merge(df_hist, gt_slice, on=['x', 'time'], how='left')
        y_cols = ['model', 'ground_truth']
    else:
        y_cols = ['model']
        
    # Create the animated plot
    fig = px.line(
        df_hist, x='x', y=y_cols, animation_frame='step',
        title=f'Training Evolution ({run_name})',
        labels={'value': 'u(x,t)'}
    )
    
    if y_range:
        fig.update_layout(yaxis_range=y_range)
        
    return fig


def plot_parameter_evolution_from_artifact(run_id, param_name='k_n'):
    """
    Loads a specific parameter's history from an MLflow run artifact and plots its evolution.

    Args:
        run_id (str): The MLflow run ID.
        param_name (str): The name of the parameter to plot (e.g., 'k_n', 'w_n').

    Returns:
        A Plotly Figure object.
    """
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    run_name = run.data.tags.get('mlflow.runName', run_id)
    
    # Find the artifact path
    artifacts = client.list_artifacts(run_id, "data_artifacts")
    param_artifact = next((a for a in artifacts if f"{param_name}_history" in a.path), None)

    if not param_artifact:
        print(f"Error: Could not find {param_name}_history artifact for run {run_id}")
        return None
        
    local_path = client.download_artifacts(run_id, param_artifact.path)
    df_params = pd.read_csv(local_path, index_col='step')
    
    # Melt dataframe for plotting with Plotly Express
    df_params.reset_index(inplace=True)
    df_melt = df_params.melt(id_vars='step', var_name='param_label', value_name='value')
    
    fig = px.line(
        df_melt, x='step', y='value', color='param_label',
        title=f'Evolution of Learned Parameter: {param_name} ({run_name})'
    )
    return fig