import csv
import os 

def convert_to_csv():
    with open('data_training.csv', mode='w',newline='') as index_file:
        fieldnames = ['name', 'label']
        writer = csv.DictWriter(index_file, fieldnames=fieldnames)

        writer.writeheader()

        my_folder_path = r'./data/images'
        list_of_list = os.listdir(my_folder_path)

        i = 0
        j = 1
        for my_file in list_of_list:
            if 'fake' in my_file:
                writer.writerow({'name':'./data/images/'+ my_file, 'label':i})
            else:
                writer.writerow({'name':'./data/images/'+ my_file, 'label': j})
                

if __name__ == '__main__':
    convert_csv = convert_to_csv()