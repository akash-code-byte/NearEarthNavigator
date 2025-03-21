import os
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime

def export_data(df, format='csv', content='data'):
    """
    Export data in the specified format
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with asteroid data
    format : str
        Export format: 'csv', 'json', 'excel', 'html', 'pdf'
    content : str
        Content to export: 'data', 'visualization', 'model_results', 'all'
        
    Returns:
    --------
    str
        Path to the exported file
    """
    # Create exports directory if it doesn't exist
    os.makedirs('exports', exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Generate base filename
    base_filename = f"neo_data_{timestamp}"
    
    # Export based on format and content
    if content == 'data' or content == 'all':
        if format == 'csv':
            filepath = f"exports/{base_filename}.csv"
            df.to_csv(filepath, index=False)
            return filepath
            
        elif format == 'json':
            filepath = f"exports/{base_filename}.json"
            df.to_json(filepath, orient='records')
            return filepath
            
        elif format == 'excel':
            filepath = f"exports/{base_filename}.xlsx"
            df.to_excel(filepath, index=False)
            return filepath
            
        elif format == 'html':
            filepath = f"exports/{base_filename}.html"
            df.to_html(filepath, index=False)
            return filepath
            
        elif format == 'pdf':
            # For PDF, we need to use a different approach, e.g. using plotly
            filepath = f"exports/{base_filename}.pdf"
            
            # Create a basic table figure
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=list(df.columns),
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[df[col] for col in df.columns],
                    fill_color='lavender',
                    align='left'
                )
            )])
            
            fig.update_layout(
                title=f"NEO Data - {timestamp}",
                width=1200,
                height=800
            )
            
            pio.write_image(fig, filepath, scale=2)
            return filepath
    
    # Return the path to the last exported file
    return filepath

def export_visualization(df, metrics=None, feature_importance=None, format='pdf'):
    """
    Export visualization in PDF format
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with asteroid data
    metrics : dict
        Dictionary of model evaluation metrics
    feature_importance : dict
        Dictionary of feature importance values
    format : str
        Export format (only 'pdf' supported for now)
        
    Returns:
    --------
    str
        Path to the exported file
    """
    # Only PDF format supported for now
    if format != 'pdf':
        raise ValueError(f"Unsupported format for visualization export: {format}")
    
    # Create exports directory if it doesn't exist
    os.makedirs('exports', exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Generate filename
    filepath = f"exports/neo_visualization_{timestamp}.pdf"
    
    # Create a figure with multiple subplots
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    # Prepare a figure with multiple subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Asteroid Distribution by Size",
            "Potentially Hazardous vs. Safe Asteroids",
            "Feature Importance",
            "Model Performance",
            "Miss Distance vs. Relative Velocity",
            "Orbital Stability Index Distribution"
        )
    )
    
    # 1. Asteroid Distribution by Size
    hist_data = go.Histogram(
        x=df['estimated_diameter_max'],
        marker_color='#2196F3',
        name="Asteroid Size"
    )
    fig.add_trace(hist_data, row=1, col=1)
    
    # 2. Hazardous vs. Safe pie chart
    hazardous_count = df['is_potentially_hazardous'].sum()
    safe_count = len(df) - hazardous_count
    
    pie_data = go.Pie(
        labels=['Safe', 'Potentially Hazardous'],
        values=[safe_count, hazardous_count],
        marker_colors=['#2196F3', '#F44336'],
        hole=0.4
    )
    fig.add_trace(pie_data, row=1, col=2)
    
    # 3. Feature Importance (if available)
    if feature_importance:
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_features)
        
        bar_data = go.Bar(
            x=features,
            y=importance,
            marker_color='#4CAF50'
        )
        fig.add_trace(bar_data, row=2, col=1)
    
    # 4. Model Performance (if available)
    if metrics:
        performance_data = go.Bar(
            x=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            y=[
                metrics.get('accuracy', 0),
                metrics.get('precision', 0),
                metrics.get('recall', 0),
                metrics.get('f1', 0)
            ],
            marker_color='#9C27B0'
        )
        fig.add_trace(performance_data, row=2, col=2)
    
    # 5. Miss Distance vs. Relative Velocity
    scatter_data = go.Scatter(
        x=df['miss_distance'],
        y=df['relative_velocity'],
        mode='markers',
        marker=dict(
            size=8,
            color=df['is_potentially_hazardous'].map({True: '#F44336', False: '#2196F3'}),
            opacity=0.6
        ),
        text=df['name']
    )
    fig.add_trace(scatter_data, row=3, col=1)
    
    # 6. Orbital Stability Index Distribution
    if 'orbital_stability_index' in df.columns:
        hist_orbital = go.Histogram(
            x=df['orbital_stability_index'],
            marker_color='#FF9800',
            name="Orbital Stability Index"
        )
        fig.add_trace(hist_orbital, row=3, col=2)
    
    # Update layout
    fig.update_layout(
        title_text=f"NEO Analysis and Threat Assessment - {timestamp}",
        showlegend=False,
        height=1200,
        width=1000
    )
    
    # Update axes
    fig.update_xaxes(title_text="Diameter (km)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    
    if feature_importance:
        fig.update_xaxes(title_text="Feature", row=2, col=1)
        fig.update_yaxes(title_text="Importance", row=2, col=1)
    
    if metrics:
        fig.update_xaxes(title_text="Metric", row=2, col=2)
        fig.update_yaxes(title_text="Value", row=2, col=2)
    
    fig.update_xaxes(title_text="Miss Distance (km)", type="log", row=3, col=1)
    fig.update_yaxes(title_text="Relative Velocity (km/s)", row=3, col=1)
    
    if 'orbital_stability_index' in df.columns:
        fig.update_xaxes(title_text="Orbital Stability Index", type="log", row=3, col=2)
        fig.update_yaxes(title_text="Count", row=3, col=2)
    
    # Save to PDF
    pio.write_image(fig, filepath, scale=2)
    
    return filepath

def create_summary_report(df, metrics=None, format='html'):
    """
    Create a summary report of the NEO analysis
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with asteroid data
    metrics : dict
        Dictionary of model evaluation metrics
    format : str
        Export format: 'html', 'pdf'
        
    Returns:
    --------
    str
        Path to the exported report
    """
    # Create exports directory if it doesn't exist
    os.makedirs('exports', exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Generate filename based on format
    if format == 'html':
        filepath = f"exports/neo_report_{timestamp}.html"
    elif format == 'pdf':
        filepath = f"exports/neo_report_{timestamp}.pdf"
    else:
        raise ValueError(f"Unsupported format for report: {format}")
    
    # Prepare report content
    if format == 'html':
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NEO Analysis Report - {timestamp}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2E86C1; }}
                .section {{ margin-bottom: 30px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                          background-color: #f5f5f5; border-radius: 5px; 
                          box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 24px; font-weight: bold; }}
                .metric-label {{ font-size: 14px; color: #777; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .hazardous {{ color: red; }}
                .safe {{ color: green; }}
            </style>
        </head>
        <body>
            <h1>Near-Earth Object Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Overview</h2>
                <div class="metric">
                    <div class="metric-value">{len(df)}</div>
                    <div class="metric-label">Total NEOs</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{df['is_potentially_hazardous'].sum()}</div>
                    <div class="metric-label">Potentially Hazardous</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{df['miss_distance'].mean() / 1000000:.2f} M</div>
                    <div class="metric-label">Avg. Miss Distance (km)</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{df['relative_velocity'].mean():.2f}</div>
                    <div class="metric-label">Avg. Velocity (km/s)</div>
                </div>
            </div>
            
            <div class="section">
                <h2>Potentially Hazardous Asteroids</h2>
        """
        
        # Add table of hazardous asteroids
        hazardous_df = df[df['is_potentially_hazardous']].sort_values('diameter_velocity_ratio', ascending=False)
        
        if len(hazardous_df) > 0:
            hazardous_df = hazardous_df[['name', 'close_approach_date_display', 'estimated_diameter_max', 
                                        'relative_velocity', 'miss_distance', 'energy_proxy']]
            
            html_content += "<table>"
            html_content += "<tr><th>Name</th><th>Approach Date</th><th>Diameter (km)</th><th>Velocity (km/s)</th><th>Miss Distance (km)</th><th>Energy Proxy</th></tr>"
            
            for _, row in hazardous_df.iterrows():
                html_content += f"""
                <tr>
                    <td>{row['name']}</td>
                    <td>{row['close_approach_date_display']}</td>
                    <td>{row['estimated_diameter_max']:.4f}</td>
                    <td>{row['relative_velocity']:.2f}</td>
                    <td>{row['miss_distance']/1000000:.2f} million</td>
                    <td>{row['energy_proxy']:.2e}</td>
                </tr>
                """
            
            html_content += "</table>"
        else:
            html_content += "<p>No potentially hazardous asteroids found in the dataset.</p>"
        
        # Add model metrics if available
        if metrics:
            html_content += f"""
            <div class="section">
                <h2>Model Performance</h2>
                <div class="metric">
                    <div class="metric-value">{metrics.get('accuracy', 0):.4f}</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{metrics.get('precision', 0):.4f}</div>
                    <div class="metric-label">Precision</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{metrics.get('recall', 0):.4f}</div>
                    <div class="metric-label">Recall</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{metrics.get('f1', 0):.4f}</div>
                    <div class="metric-label">F1 Score</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{metrics.get('auc', 0):.4f}</div>
                    <div class="metric-label">AUC-ROC</div>
                </div>
            </div>
            """
        
        # Close HTML document
        html_content += """
            <div class="section">
                <h2>Notes</h2>
                <p>This report was automatically generated by the Comprehensive Space Threat Assessment and Prediction System.</p>
                <p>Data Source: NASA Near Earth Object API</p>
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(filepath, 'w') as f:
            f.write(html_content)
    
    elif format == 'pdf':
        # For PDF generation, we would typically use a library like reportlab
        # But for simplicity, we'll create a Plotly figure with tables and export it
        
        # Create a figure with multiple subplots for the report
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        # Create figure with subplots
        fig = make_subplots(
            rows=4, cols=1,
            row_heights=[0.2, 0.3, 0.3, 0.2],
            subplot_titles=(
                "Near-Earth Object Analysis Report",
                "Overview Statistics",
                "Potentially Hazardous Asteroids",
                "Model Performance"
            ),
            vertical_spacing=0.1
        )
        
        # Add overview statistics
        statistics = go.Table(
            header=dict(
                values=["Metric", "Value"],
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[
                    ["Total NEOs", "Potentially Hazardous", "Avg. Miss Distance", "Avg. Velocity", "Max Diameter"],
                    [
                        len(df),
                        df['is_potentially_hazardous'].sum(),
                        f"{df['miss_distance'].mean() / 1000000:.2f} million km",
                        f"{df['relative_velocity'].mean():.2f} km/s",
                        f"{df['estimated_diameter_max'].max()*1000:.1f} m"
                    ]
                ],
                fill_color='lavender',
                align='left'
            )
        )
        fig.add_trace(statistics, row=2, col=1)
        
        # Add hazardous asteroids table
        hazardous_df = df[df['is_potentially_hazardous']].sort_values('diameter_velocity_ratio', ascending=False)
        
        if len(hazardous_df) > 0:
            hazardous_df = hazardous_df[['name', 'close_approach_date_display', 'estimated_diameter_max', 
                                       'relative_velocity', 'miss_distance']]
            
            hazardous_table = go.Table(
                header=dict(
                    values=["Name", "Approach Date", "Diameter (km)", "Velocity (km/s)", "Miss Distance"],
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[
                        hazardous_df['name'].head(10),
                        hazardous_df['close_approach_date_display'].head(10),
                        hazardous_df['estimated_diameter_max'].head(10).round(4),
                        hazardous_df['relative_velocity'].head(10).round(2),
                        (hazardous_df['miss_distance'].head(10) / 1000000).round(2).astype(str) + " million"
                    ],
                    fill_color='lavender',
                    align='left'
                )
            )
            fig.add_trace(hazardous_table, row=3, col=1)
        
        # Add model metrics if available
        if metrics:
            metrics_table = go.Table(
                header=dict(
                    values=["Metric", "Value"],
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[
                        ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"],
                        [
                            f"{metrics.get('accuracy', 0):.4f}",
                            f"{metrics.get('precision', 0):.4f}",
                            f"{metrics.get('recall', 0):.4f}",
                            f"{metrics.get('f1', 0):.4f}",
                            f"{metrics.get('auc', 0):.4f}"
                        ]
                    ],
                    fill_color='lavender',
                    align='left'
                )
            )
            fig.add_trace(metrics_table, row=4, col=1)
        
        # Update layout
        fig.update_layout(
            title_text=f"NEO Analysis Report - {timestamp}",
            showlegend=False,
            height=1200,
            width=800
        )
        
        # Save to PDF
        pio.write_image(fig, filepath, scale=2)
    
    return filepath
