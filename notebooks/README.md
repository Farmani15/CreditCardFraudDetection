# Jupyter Notebooks

This directory contains Jupyter notebooks for exploratory data analysis, model experimentation, and visualization.

## Notebooks

- `exploratory_data_analysis.ipynb`: Explores the credit card fraud detection dataset to understand its characteristics and prepare for model building.

## Running Notebooks

To run these notebooks, you need to have Jupyter installed. You can install it via pip:

```bash
pip install jupyter
```

Then, start the Jupyter notebook server:

```bash
jupyter notebook
```

This will open a browser window where you can navigate to and open the notebooks.

## Dependencies

The notebooks require the same dependencies as the main project. Make sure you have installed all the requirements:

```bash
pip install -r ../requirements.txt
```

## Creating New Notebooks

When creating new notebooks for this project, please follow these guidelines:

1. Use a clear and descriptive name for the notebook.
2. Include markdown cells to explain the purpose and methodology.
3. Document your code with comments.
4. Keep the notebook focused on a specific task or analysis.
5. Clean up the notebook before committing (clear outputs, remove unnecessary cells).

## Best Practices

- Import the project's configuration and utility modules to maintain consistency.
- Save visualizations to the `models/plots` directory for use in the web application.
- Use relative imports to access project modules.
- Document insights and findings clearly. 