import wandb

# Set your API key
wandb.login()

# Set the source and destination projects
src_entity = "roy-mahlab-ben-gurion-university-of-the-negev"
src_project = "anygraph_svd"
dst_entity = "roy-mahlab-ben-gurion-university-of-the-negev"
dst_project = "anygraph_no_adj_features"

run_to_copy = "no_input_features_use"
# Initialize the wandb API
api = wandb.Api()

# Get the runs from the source project
runs = api.runs(f"{src_entity}/{src_project}")

# Iterate through the runs and copy them to the destination project
print(len(runs))

for run in runs:
    if run.name != run_to_copy:
        continue
    # Get the run history and files
    history = run.history()
    files = run.files()

    # Create a new run in the destination project
    new_run = wandb.init(project=dst_project, entity=dst_entity, config=run.config, name=run.name,resume="allow")
    
    # Log the history to the new run
    for index, row in history.iterrows():
        new_run.log(row.to_dict())

    # Upload the files to the new run
    for file in files:
        file.download(replace=True)
        new_run.save(file.name,policy = "now")

    # Finish the new run
    new_run.finish()