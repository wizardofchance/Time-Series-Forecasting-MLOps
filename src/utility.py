
import os
from dotenv import load_dotenv, find_dotenv
import hopsworks




#########################################################################################
# CONNECT TO HOPSWORKS PROJECT
#########################################################################################

def taxi_connect_to_project():

    project = hopsworks.login(
        project = os.environ['HOPSWORKS_PROJECT_NAME'],
        api_key_value = os.environ['HOPSWORKS_API_KEY']        
        )
    
    return project





#########################################################################################
# INSERT DATA FROM FEATURE STORE
#########################################################################################

def taxi_insert_data_into_feature_store(project, df_demand, feat_group_name, feat_group_ver):    

    feature_store = project.get_feature_store()

    feature_group = feature_store.get_or_create_feature_group(

        name = feat_group_name,
        version = feat_group_ver,
        description = "Timeseries data at hourly frequency for location 90",
        primary_key = ['date_time'],
        event_time = 'date_time'
    )

    try:
        feature_group.insert(df_demand)
        print(f'data successfully inserted in feature store')
    except Exception as error:
        print(f"Error: {error}")





#########################################################################################
# FETCH DATA FROM FEATURE STORE
#########################################################################################

def taxi_create_feature_view(project, feat_view_name, feat_view_ver, 
                             parent_feat_group_name, parent_feat_group_ver):  

    feature_store = project.get_feature_store()

    parent_feature_group = feature_store.get_feature_group(        
        name = parent_feat_group_name,
        version = parent_feat_group_ver      
        )

    try:
        feature_store.create_feature_view(
            name = feat_view_name,
            version = feat_view_ver,
            query = parent_feature_group.select_all()            
            )  

        response = f"""feature view {feat_view_name} created """    

    except:
        response = f"""Feature view {feat_view_name} already exists."""
    
    return response




def taxi_fetch_data_from_feature_store(project, feat_view_name, feat_view_ver):

    feature_store = project.get_feature_store()
        
    feature_view = feature_store.get_feature_view(
        name = feat_view_name, 
        version = feat_view_ver
        )

    df_demand, _ = feature_view.training_data( 
        description = "Timeseries data at hourly frequency for location 90")
    
    df_demand = df_demand.sort_values(by = 'date_time')
    return df_demand.astype({'date_time': 'datetime64[ns]'})







