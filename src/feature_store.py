from src.config import (
    HOPSWORKS_PROJECT_NAME,
    HOPSWORKS_API_KEY,
    HOPSWORKS_HOST,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
)


def store_features_in_hopsworks(feature_df):
    """
    Stores processed feature data into Hopsworks Feature Store.
    If Hopsworks keys are not configured, the function skips Hopsworks
    and keeps the local CSV output only.
    """

    if not HOPSWORKS_PROJECT_NAME or not HOPSWORKS_API_KEY:
        print("Hopsworks is not configured.")
        print("Features were saved locally only.")
        return None

    import pandas as pd
    import hopsworks

    feature_df = feature_df.copy()

    # Hopsworks expects a primary key timestamp column.
    # Our pipeline uses event_time, so we create timestamp from event_time.
    if "timestamp" not in feature_df.columns:
        feature_df["timestamp"] = pd.to_datetime(feature_df["event_time"])

    feature_df["timestamp"] = pd.to_datetime(feature_df["timestamp"]).dt.tz_localize(None)

    project = hopsworks.login(
        project=HOPSWORKS_PROJECT_NAME,
        api_key_value=HOPSWORKS_API_KEY,
        host=HOPSWORKS_HOST,
    )

    fs = project.get_feature_store()

    feature_group = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Karachi AQI feature pipeline with weather and pollution features",
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        online_enabled=True,
    )

    feature_group.insert(
        feature_df,
        write_options={"wait_for_job": True},
    )

    print(f"Inserted {len(feature_df)} row(s) into Hopsworks feature group.")

    return feature_group
