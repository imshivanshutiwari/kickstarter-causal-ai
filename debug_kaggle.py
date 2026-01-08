from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

datasets = api.dataset_list(search="kickstarter-projects")
if datasets:
    ds = datasets[0]
    print(f"Type: {type(ds)}")
    print("Attributes:")
    for attr in dir(ds):
        if not attr.startswith('_'):
            try:
                val = getattr(ds, attr)
                print(f"  {attr}: {val} (type: {type(val)})")
            except:
                print(f"  {attr}: <error reading>")
else:
    print("No datasets found")
