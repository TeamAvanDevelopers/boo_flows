import src.connect_database as cd
import src.json_format as jf
import src.user_information as ui


def df_to_list(df):
    return df.to_string()


rdf = cd.ResultDataFrame()
get_users_list_query = "SELECT user_id FROM db_recommendation_test.new_table group by user_id"
users_df = rdf.result_table(get_users_list_query)
users_list = list(users_df['user_id'])

get_posts_list_query = "SELECT post_id FROM db_recommendation_test.new_table where extracted_tags != \"\" group by post_id" 
posts_df =(rdf.result_table(get_posts_list_query))
posts_list = list(posts_df['post_id'])

get_extracted_tags_list_query = "SELECT extracted_tags FROM db_recommendation_test.new_table where extracted_tags != \"\""
extracted_df = rdf.result_table(get_extracted_tags_list_query)
extracted_list = ui.deduplicated_all_tags(extracted_df['extracted_tags'])

    
user_info_dict_list = dict()
for uid in users_list:
    user_info = ui.UserInfo(uid).get_user_info(users_list, posts_list, extracted_list)
    user_info_dict = ui.format_user_info_for_recommendation(user_info)
    file_name = str(user_info_dict['user'])
    jf.save_to_json(user_info_dict, "C:\\Users\\User\\Desktop\\공동_작업_깃헙\\boo_flows\\DataProcessing\\res", file_name)
