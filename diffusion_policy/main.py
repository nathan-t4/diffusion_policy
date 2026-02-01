import zarr

def main():
    data_version = "v2"
    dataset_path = f"data/pusht_cchi_{data_version}.zarr"

    dataset = zarr.open(dataset_path, mode="r")

    print("Dataset keys:", list(dataset['data'].keys()))

    train_image_data = dataset['data']['img']

    print("Image data shape:",train_image_data.shape)

    train_action_data = dataset['data']['action']
    print("Action data shape:", train_action_data.shape)

    train_state_data = dataset['data']['state']
    print("State data shape:", train_state_data.shape)

    episode_ends = dataset['meta']['episode_ends'][:]
    print("Episode ends:", episode_ends)


if __name__ == "__main__":
    main()
