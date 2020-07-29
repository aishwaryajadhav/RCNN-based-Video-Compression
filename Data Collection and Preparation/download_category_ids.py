
def get_video_id_category_tuples(categories, max_num_videos_per_cat=3000):
    video_id_category_tuples = []
    for category in categories:
        ids = get_youtube_ids_of_category(category)
        if len(ids) > max_num_videos_per_cat:
        ids = random.sample(ids, max_num_videos_per_cat)
        with open('D:\\Program Files\\meais-sf\\youtube-8m-videos-frames-master\\category_ids_dir\\'+category+'.txt', 'w') as filehandle:  
        for listitem in ids:
            filehandle.write('%s\n' % listitem)


def download_youtube_categories(categories, max_num_videos_per_cat=100):
    get_video_id_category_tuples(categories, max_num_videos_per_cat=max_num_videos_per_cat)



def main():
    categories_to_download=[]
    with open('D:\\Program Files\\meais-sf\\youtube-8m-videos-frames-master\\cat_list.txt') as f:
        content = f.readlines()
        categories_to_download.append(content.rstrip())
  
    categories_to_download = [x.rstrip() for x in content]   
    print(categories_to_download)    
    download_youtube_categories(categories_to_download)
    print(done)


if __name__ == '__main__':
    main()